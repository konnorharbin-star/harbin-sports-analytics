"""Unit tests for free same-book, before-kickoff *movement*, never fabricated CLV."""
import csv
from datetime import datetime, timezone
import io
import json
import math

from platform_ops.samebook_movement import (
    markdown_report, latest_movement_report, parse_cfb_market_rows,
    parse_nfl_market_rows, select_later_same_book,
)

KICKOFF="2026-10-09T00:15:00+00:00"
OBSERVED="2026-10-08T22:30:00+00:00"
ENTRY_QUOTE="2026-10-08T22:20:00+00:00"
CAPTURED="2026-10-08T23:40:00+00:00"
QUOTE_AT="2026-10-08T23:35:00+00:00"


def candidate(market="spread",side="away",line=9.5,odds=-110,book="ActionNetwork book 69"):
    return {
        "league":"CFB","game_id":"g1","market":market,"side":side,
        "home_team":"Home","away_team":"Away","line":line,"odds":odds,
        "book":book,"kickoff_utc":KICKOFF,"observed_at_utc":OBSERVED,
        "quoted_at_utc":ENTRY_QUOTE,"blockers":["model_not_production_validated"],
    }


def cfb_csv(quotes,*,capture=CAPTURED,kickoff=KICKOFF):
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=[
        "game_id","captured_at","kickoff","market_quotes_json",
    ])
    writer.writeheader()
    writer.writerow({"game_id":"g1","captured_at":capture,"kickoff":kickoff,
                     "market_quotes_json":json.dumps(quotes)})
    return stream.getvalue()


def cfb_quote(*,provider="ActionNetwork book 69",updated=QUOTE_AT,spread=-8.5,
              total=47.0,source="action_network",home_price=-110,away_price=-120):
    return {
        "provider":provider,"source":source,"last_update":updated,
        "captured_at":CAPTURED,
        "home_spread":spread,"home_spread_price":home_price,
        "away_spread_price":away_price,
        "market_total":total,"over_price":-115,"under_price":-110,
        "home_ml":-250,"away_ml":+210,
    }


def nfl_csv(row):
    buff=io.StringIO()
    fields=["captured_at","kickoff","game_id","market_type","provider","book",
            "source_event_id","snapshot_id","first_side","first_line",
            "first_american_odds","second_side","second_line",
            "second_american_odds"]
    w=csv.DictWriter(buff,fieldnames=fields)
    w.writeheader()
    w.writerow(row)
    return buff.getvalue()


def nfl_row(**kwargs):
    row={
        "captured_at":CAPTURED,"kickoff":KICKOFF,"game_id":"2026_05_TB_DAL",
        "market_type":"spread","provider":"espn",
        "book":"Example Sportsbook","source_event_id":"401000001",
        "snapshot_id":"snap-1","first_side":"home","first_line":"-8.5",
        "first_american_odds":"-110","second_side":"away",
        "second_line":"8.5","second_american_odds":"-115",
    }
    row.update(kwargs)
    return row


def test_cfb_extracts_actual_same_book_quote_and_side_line():
    close,stats=parse_cfb_market_rows(cfb_csv([cfb_quote()]))
    assert stats["accepted_side_quotes"]==6
    result=select_later_same_book(candidate(),close)
    assert result["status"]=="OBSERVED_SAME_BOOK_NEAR_KICKOFF"
    assert math.isclose(result["line_advantage_points"],1.0)
    assert result["same_line_implied_probability_move"] is None
    assert result["closing_american_odds"]==-120
    assert result["official_book_closing_verified"] is False
    assert result["wager_executed"] is False


def test_wrong_book_is_never_substituted_even_when_better():
    close,_=parse_cfb_market_rows(cfb_csv([cfb_quote(provider="Other Book")]))
    result=select_later_same_book(candidate(),close)
    assert result["status"]=="NO_VERIFIABLE_LATER_SAME_BOOK_SNAPSHOT"
    assert result["closing_line"] is None


def test_identical_point_line_compares_american_odds_break_even_only():
    close,_=parse_cfb_market_rows(cfb_csv([cfb_quote(spread=-9.5,away_price=-120)]))
    result=select_later_same_book(candidate(),close)
    assert result["line_advantage_points"]==0
    assert result["same_line_implied_probability_move"]>0
    assert not result["price_executability_verified"]


def test_moneyline_break_even_price_change_without_fake_no_vig():
    close,_=parse_cfb_market_rows(cfb_csv([cfb_quote()]))
    entry=candidate(market="moneyline",side="Away",line=None,odds=+220)
    result=select_later_same_book(entry,close)
    assert result["line_advantage_points"] is None
    assert result["same_line_implied_probability_move"]>0


def test_total_over_under_advantage_direction():
    close,_=parse_cfb_market_rows(cfb_csv([cfb_quote(total=47)]))
    under=candidate(market="total",side="U",line=49.5,odds=-110)
    over=candidate(market="total",side="O",line=45.5,odds=-110)
    assert select_later_same_book(under,close)["line_advantage_points"]==2.5
    assert select_later_same_book(over,close)["line_advantage_points"]==1.5


def test_nfl_paired_snapshot_reads_both_sides_without_source_timestamp():
    csv_text=nfl_csv(nfl_row())
    quotes,stats=parse_nfl_market_rows(csv_text)
    assert stats["accepted_side_quotes"]==2
    assert set(x["role"] for x in quotes)=={"home","away"}
    assert all(x["time_provenance"]=="public_capture_only" for x in quotes)
    entry=candidate(book="Example Sportsbook",line=9.5)
    entry.update(league="NFL",game_id="2026_05_TB_DAL")
    result=select_later_same_book(entry,quotes)
    assert result["status"]=="OBSERVED_SAME_BOOK_NEAR_KICKOFF"
    assert result["line_advantage_points"]==1.0
    assert not result["official_book_closing_verified"]


def test_nfl_incomplete_pair_fails_closed():
    cases=[
        nfl_row(second_side="home"),
        nfl_row(second_american_odds=""),
        nfl_row(first_line=""),
        nfl_row(captured_at="2026-10-09T00:20:00+00:00"),
        nfl_row(source_event_id=""),
    ]
    for row in cases:
        quotes,_=parse_nfl_market_rows(nfl_csv(row))
        assert not quotes


def test_cfb_synthetic_only_timestamp_or_paid_source_is_rejected():
    for q in [
        cfb_quote(updated=None),
        cfb_quote(source="odds_api"),
        cfb_quote(updated="2026-10-09T00:25:00+00:00"),
        cfb_quote(updated="2026-10-08T20:00:00+00:00"),
    ]:
        quotes,_=parse_cfb_market_rows(cfb_csv([q]))
        assert not quotes


def test_cfb_bad_json_and_early_capture_cannot_be_a_close():
    cases=[
        "not json",
        cfb_csv([cfb_quote()],capture="2026-10-08T21:50:00+00:00"),
        cfb_csv([cfb_quote()],capture="2026-10-09T00:16:00+00:00"),
    ]
    # Bad source content and illegal timing never create a comparable quote.
    first,_=parse_cfb_market_rows(
        "game_id,captured_at,kickoff,market_quotes_json\n"
        'g1,2026-10-08T23:40:00+00:00,2026-10-09T00:15:00+00:00,"not json"\n'
    )
    assert not first
    for src in cases[1:]:
        qs,_=parse_cfb_market_rows(src)
        assert not qs


def test_ambiguous_latest_same_book_is_not_picked_winner_favorably():
    qs,_=parse_cfb_market_rows(cfb_csv([
        cfb_quote(spread=-8.5),cfb_quote(spread=-7.5)
    ]))
    result=select_later_same_book(candidate(),qs)
    assert result["status"]=="AMBIGUOUS_LATEST_SAME_BOOK_QUOTES"
    assert result["line_advantage_points"] is None


def test_only_most_recent_proven_pre_kickoff_snapshot_used():
    earlier=cfb_csv([cfb_quote(spread=-9.0,updated="2026-10-08T23:03:00+00:00")],
                    capture="2026-10-08T23:10:00+00:00")
    newer=cfb_csv([cfb_quote(spread=-8.5)])
    q1,_=parse_cfb_market_rows(earlier)
    q2,_=parse_cfb_market_rows(newer)
    comparison=select_later_same_book(candidate(),q1+q2)
    assert comparison["closing_line"]==8.5
    assert comparison["line_advantage_points"]==1.0


def test_no_unverified_row_will_have_invented_close_or_risk_claim():
    a=candidate()
    bad=select_later_same_book(a,[])
    assert bad["status"].startswith("NO_VERIFIABLE")
    assert bad["closing_line"] is None
    report=latest_movement_report([a],{"CFB":[],"NFL":[]})
    assert report["summary"]["unverified_or_unmatched"]==1
    assert report["official_closing_prices_verified"] is False
    assert report["automatic_betting_enabled"] is False
    assert report["paid_sources_used"] is False
    assert "not official closing-line value" in markdown_report(report).lower()


def test_near_kickoff_time_requirement_not_true_final_market_close():
    close,_=parse_cfb_market_rows(cfb_csv([cfb_quote()],
                                        capture="2026-10-08T22:34:00+00:00"))
    # Source update at 23:35 is newer than observation; reject future capture.
    assert not close


def test_raw_bad_quote_price_never_promoted():
    bad=candidate(odds=-25)
    report=select_later_same_book(bad,[])
    assert report["status"]=="INVALID_OR_UNTIMED_ORIGINAL_QUOTE"
