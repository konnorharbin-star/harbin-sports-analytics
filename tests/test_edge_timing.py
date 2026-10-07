import pandas as pd

from harbin.edge_timing import enrich_edge_timing, summarize_observed_survival


NOW = pd.Timestamp("2026-10-07T20:00:00Z")
KICK = "2026-10-10T20:00:00Z"


def candidate(**kw):
    row = dict(
        game_id="g1", date=KICK, home_team="Home", away_team="Away",
        market="spread", side="Home", book="Book A", line=-9.5,
        odds=-106, quote_at="2026-10-07T19:58:00Z",
        edge_priority="ROBUST_CORE", bet_to_line=-10,
        conservative_price_margin=0.10, probability=0.66, ev=0.25
    )
    row.update(kw)
    return row


def observation(snapshot, quote, line=-9, odds=-110, **kw):
    row = dict(candidate(line=line, odds=odds, quote_at=quote),
               snapshot_at=snapshot, regime_band="6-8", edge=6.8)
    row.update(kw)
    return row


def evaluate(tmp_path, current, histories):
    path = tmp_path / "edges.csv"
    if histories:
        pd.DataFrame(histories).to_csv(path, index=False)
    return enrich_edge_timing(pd.DataFrame([current]), path, as_of=NOW).iloc[0]


def test_missing_history_fails_closed(tmp_path):
    r = evaluate(tmp_path, candidate(), [])
    assert r.timing_action == "NO_TIMING_SIGNAL"
    assert r.timing_reason == "NO_COMPARABLE_HISTORY"


def test_same_book_deteriorating_line_is_research_bet_now(tmp_path):
    r = evaluate(tmp_path, candidate(), [
        observation("2026-10-07T17:00:00Z", "2026-10-07T16:59:00Z", line=-9)
    ])
    assert r.timing_action == "BET_NOW_RESEARCH"
    assert r.timing_line_move_pts == -0.5
    assert r.timing_status == "RESEARCH_ONLY"


def test_same_book_improving_line_is_wait(tmp_path):
    r = evaluate(tmp_path, candidate(line=-9.5, bet_to_line=-10), [
        observation("2026-10-07T17:00:00Z", "2026-10-07T16:59:00Z", line=-10)
    ])
    assert r.timing_action == "WAIT_MONITOR"


def test_different_books_and_future_snapshots_are_not_history(tmp_path):
    r = evaluate(tmp_path, candidate(), [
        observation("2026-10-07T17:00:00Z", "2026-10-07T16:59:00Z", book="Book B"),
        observation("2026-10-07T21:00:00Z", "2026-10-07T20:59:00Z", line=-9),
    ])
    assert r.timing_action == "NO_TIMING_SIGNAL"
    assert r.timing_observations == 0


def test_stale_and_post_kickoff_quotes_never_promote(tmp_path):
    histories = [observation("2026-10-07T17:00:00Z", "2026-10-07T16:59:00Z", line=-9)]
    stale = evaluate(tmp_path, candidate(quote_at="2026-10-07T17:00:00Z"), histories)
    assert stale.timing_action == "PASS"
    assert stale.timing_reason == "STALE_OR_FUTURE_QUOTE"
    started = evaluate(tmp_path, candidate(date="2026-10-07T19:00:00Z"), histories)
    assert started.timing_reason == "GAME_STARTED_OR_BAD_KICKOFF"


def test_price_and_bet_to_gates(tmp_path):
    histories = [observation("2026-10-07T17:00:00Z", "2026-10-07T16:59:00Z", line=-9)]
    beyond = evaluate(tmp_path, candidate(line=-10.5), histories)
    assert beyond.timing_reason == "OUTSIDE_BET_TO_LINE"
    thin = evaluate(tmp_path, candidate(conservative_price_margin=0), histories)
    assert thin.timing_reason == "NO_CONSERVATIVE_PRICE_CUSHION"
    not_core = evaluate(tmp_path, candidate(edge_priority="CORE_LINE_THIN"), histories)
    assert not_core.timing_reason == "NOT_PRIORITY_CORE"


def test_price_deterioration_same_line_triggers_research_only(tmp_path):
    r = evaluate(tmp_path, candidate(line=-9.5, odds=-125), [
        observation("2026-10-07T17:00:00Z", "2026-10-07T16:59:00Z", line=-9.5, odds=-110)
    ])
    assert r.timing_action == "BET_NOW_RESEARCH"
    assert r.timing_price_move_pp > 1


def test_survival_descriptive_not_official_closing_line(tmp_path):
    path = tmp_path / "history.csv"
    observations = [
        observation("2026-10-01T17:00:00Z", "2026-10-01T16:59:00Z",
                    date="2026-10-05T20:00:00Z", line=-9.5, edge=6.8),
        observation("2026-10-05T16:00:00Z", "2026-10-05T15:59:00Z",
                    date="2026-10-05T20:00:00Z", line=-10, edge=6.8),
    ]
    pd.DataFrame(observations).to_csv(path, index=False)
    report = summarize_observed_survival(path, as_of=NOW)
    assert report["comparable_series"] == 1
    assert report["survived_bet_to"] == 1
    assert report["closing_line_claim"] is False
    assert report["betting_authorized"] is False
