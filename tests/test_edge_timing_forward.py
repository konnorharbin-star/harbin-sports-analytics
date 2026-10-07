"""Prospective timing strategy validation: do not retroactively invent signals."""
import json

import pandas as pd

from harbin.edge_timing_forward import (
    append_timing_decisions,
    grade_timing_forward,
    _movement,
)

DECISION = "2026-10-07T20:00:00Z"
KICKOFF = "2026-10-10T20:00:00Z"


def edge(**changes):
    row = {
        "game_id": "g1", "date": KICKOFF, "home_team": "Home",
        "away_team": "Away", "market": "spread", "side": "Home",
        "book": "Book A", "line": -9.5, "odds": -106,
        "quote_at": "2026-10-07T19:58:00Z",
        "edge_priority": "ROBUST_CORE",
        "edge_reliability_status": "SUPPORTED_SUBGROUP",
        "price_evidence_status": "CONFIRMED",
        "probability": .661, "ev": .28, "edge": 6.8,
        "bet_to_line": -10, "conservative_price_margin": .10,
        "timing_action": "BET_NOW_RESEARCH",
        "timing_reason": "OBSERVED_SAME_BOOK_DETERIORATION",
        "timing_status": "RESEARCH_ONLY", "timing_evidence_source": "captured_market_quote",
        "timing_previous_line": -9.0, "timing_previous_odds": -110,
        "timing_last_observed_at": "2026-10-07T17:00:00Z",
    }
    row.update(changes)
    return row


def capture(stamp, line, odds=-106, book="Book A", kickoff=KICKOFF,
            game="g1", home="Home", away="Away"):
    return {
        "captured_at": stamp, "game_id": game, "kickoff": kickoff,
        "home_team": home, "away_team": away,
        "market_quotes_json": json.dumps([
            {"provider": book, "home_spread": line, "home_spread_price": odds,
             "away_spread_price": -110}
        ]),
    }


def make_ledger(tmp_path, **kw):
    path = tmp_path / "decisions.csv"
    result = append_timing_decisions(pd.DataFrame([edge(**kw)]), path, DECISION)
    assert result["appended"] == 1
    return path


def grade(tmp_path, ledger, markets=(), as_of="2026-10-10T21:00:00Z"):
    market_file = tmp_path / "market.csv"
    if markets:
        pd.DataFrame(markets).to_csv(market_file, index=False)
    report = grade_timing_forward(
        ledger, market_file, tmp_path / "no_candidates.csv",
        reports_dir=tmp_path / "reports", as_of=as_of
    )
    frame = pd.read_csv(tmp_path / "reports" / "edge_timing_forward_graded.csv")
    return report, frame


def test_first_clean_decision_is_immutable_across_reruns_and_books(tmp_path):
    path = tmp_path / "ledger.csv"
    first = append_timing_decisions(pd.DataFrame([edge()]), path, DECISION)
    next_run = append_timing_decisions(
        pd.DataFrame([edge(book="Book B", line=-9, odds=-110,
                           quote_at="2026-10-07T20:30:00Z",
                           timing_action="WAIT_MONITOR", bet_to_line=-10)]),
        path, "2026-10-07T21:00:00Z"
    )
    assert first["appended"] == 1
    assert next_run["appended"] == 0
    observed = pd.read_csv(path)
    assert len(observed) == 1
    assert observed.iloc[0]["book"] == "Book A"
    assert observed.iloc[0]["entry_line"] == -9.5
    assert observed.iloc[0]["timing_action"] == "BET_NOW_RESEARCH"
    assert observed.iloc[0]["decision_at"] == "2026-10-07T20:00:00+00:00"


def test_rejects_expired_noncore_and_future_quoted_decisions(tmp_path):
    bad = [
        edge(edge_priority="CORE_LINE_THIN"),
        edge(price_evidence_status="PLAUSIBLE"),
        edge(timing_action="NO_TIMING_SIGNAL"),
        edge(quote_at="2026-10-07T20:06:00Z"),
        edge(quote_at="2026-10-07T17:00:00Z"),
        edge(line=-10.5),
        edge(conservative_price_margin=0),
        edge(date="2026-10-07T19:00:00Z"),
    ]
    result = append_timing_decisions(
        pd.DataFrame(bad), tmp_path / "ledger.csv", DECISION
    )
    assert result["appended"] == 0
    assert not (tmp_path / "ledger.csv").exists()


def test_primary_first_future_same_book_observation_and_final_hour(tmp_path):
    ledger = make_ledger(tmp_path)
    report, graded = grade(tmp_path, ledger, [
        capture("2026-10-08T01:00:00Z", -11),   # before +6h: excluded
        capture("2026-10-08T02:15:00Z", -10),   # FIRST qualifying quote
        capture("2026-10-08T03:15:00Z", -8),    # later better: must not cherry pick
        capture("2026-10-08T02:05:00Z", -15, book="Book B"),
        capture("2026-10-10T19:05:00Z", -10.5),
        capture("2026-10-10T19:45:00Z", -10),
    ])
    assert report["status"] == "EARLY_FORWARD"
    assert report["overall"]["primary_observed"] == 1
    assert report["overall"]["correct"] == 1
    assert report["overall"]["accuracy"] == 1.0
    assert graded.iloc[0]["primary_line"] == -10
    assert graded.iloc[0]["primary_movement"] == "WORSE"
    assert graded.iloc[0]["primary_observed_at"] == "2026-10-08T02:15:00+00:00"
    assert graded.iloc[0]["near_kickoff_line"] == -10
    assert graded.iloc[0]["near_kickoff_observed_at"] == "2026-10-10T19:45:00+00:00"
    assert report["overall"]["near_kickoff_conclusive"] == 1
    assert report["overall"]["near_kickoff_correct"] == 1
    assert report["overall"]["near_kickoff_accuracy"] == 1.0
    assert report["overall"]["primary_quote_coverage"] == 1.0
    assert report["no_official_close_claim"]
    assert not report["betting_authorized"]


def test_wait_correct_when_later_line_better(tmp_path):
    ledger = make_ledger(tmp_path, timing_action="WAIT_MONITOR",
                         timing_reason="OBSERVED_SAME_BOOK_IMPROVEMENT")
    report, graded = grade(tmp_path, ledger, [
        capture("2026-10-08T02:20:00Z", -9),
    ])
    assert graded.iloc[0]["primary_movement"] == "BETTER"
    assert report["by_action"]["WAIT_MONITOR"]["correct"] == 1
    assert report["by_action"]["BET_NOW_RESEARCH"]["decisions"] == 0


def test_missing_quote_not_treated_as_win_and_no_future_leakage(tmp_path):
    ledger = make_ledger(tmp_path)
    prices = [capture("2026-10-08T02:30:00Z", -10)]
    early, _ = grade(tmp_path, ledger, prices, "2026-10-08T01:00:00Z")
    assert early["overall"]["primary_observed"] == 0
    assert early["overall"]["primary_pending"] == 1
    assert early["overall"]["accuracy"] is None
    late, out = grade(tmp_path, ledger, [
        capture("2026-10-08T05:30:00Z", -10)  # too late for +6h to +9h window
    ], "2026-10-10T21:00:00Z")
    assert out.iloc[0]["primary_status"] == "NO_HORIZON_QUOTE"
    assert late["overall"]["conclusive"] == 0
    assert late["overall"]["accuracy"] is None


def test_tradeoff_line_vs_price_not_counted_as_win(tmp_path):
    ledger = make_ledger(tmp_path, timing_action="WAIT_MONITOR")
    report, out = grade(tmp_path, ledger, [
        capture("2026-10-08T02:25:00Z", -9, -145),
    ])
    assert out.iloc[0]["primary_movement"] == "MIXED_LINE_PRICE"
    assert report["overall"]["primary_mixed"] == 1
    assert report["overall"]["conclusive"] == 0
    assert report["overall"]["accuracy"] is None


def test_missing_last_hour_is_not_closing_line(tmp_path):
    ledger = make_ledger(tmp_path)
    report, out = grade(tmp_path, ledger, [
        capture("2026-10-10T18:00:00Z", -11),  # not in final hour
        capture("2026-10-10T20:30:00Z", -11),  # after kickoff
    ])
    assert out.iloc[0]["near_kickoff_status"] == "NO_FINAL_HOUR_QUOTE"
    assert report["overall"]["near_kickoff_observed"] == 0
    assert report["no_official_close_claim"]


def test_empty_ledger_yields_pending_and_empty_csv(tmp_path):
    report, out = grade(tmp_path, tmp_path / "missing.csv")
    assert report["status"] == "PENDING_FORWARD"
    assert report["raw_decision_rows"] == 0
    assert out.empty


def test_mixed_price_direction_is_ambiguous_even_if_line_better():
    movement, line_delta, price_delta = _movement(-9.5, -106, -9, -140)
    assert movement == "MIXED_LINE_PRICE"
    assert line_delta == .5
    assert price_delta > 0


def test_no_historical_backfill_from_candidate_quotes_at_same_time(tmp_path):
    ledger = make_ledger(tmp_path)
    candidates = tmp_path / "candidate_history.csv"
    pd.DataFrame([{
        "game_id": "g1", "market": "spread", "side": "Home",
        "book": "Book A", "line": -10, "odds": -106,
        "snapshot_at": "2026-10-08T01:59:00Z",
        "quote_at": "2026-10-08T02:10:00Z",  # future relative to snapshot, not confirmed
        "date": KICKOFF,
    }]).to_csv(candidates, index=False)
    result = grade_timing_forward(
        ledger, tmp_path / "missing_market.csv", candidates, tmp_path / "reports",
        as_of="2026-10-10T21:00:00Z"
    )
    assert result["overall"]["primary_observed"] == 0
