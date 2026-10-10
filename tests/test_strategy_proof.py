import pandas as pd
import pytest

from scripts.strategy_proof import evaluate


def row(season, profit, verified=False):
    return {
        "market": "spread", "market_role": "favorite", "side_location": "home",
        "edge_bucket": "6-8", "season": season,
        "result": 1 if profit > 0 else -1, "profit": profit,
        "entry_quote_verified": verified,
    }


def test_retrospective_win_streak_is_not_proven_profit():
    bets = pd.DataFrame([row(2023, .91), row(2024, .91), row(2025, .91)])
    board = pd.DataFrame([{"game_id": "7", "evidence_tier": "ROBUST_CORE",
                           "away_team": "Away", "home_team": "Home",
                           "side": "Home", "line": "-6.5", "american_odds": "-110",
                           "decision": "RECHECK_PRICE",
                           "reasons": "UNRESOLVED_BOOK",
                           "bet_approved": "False"}])
    report = evaluate(bets, {"clean_entries": 0, "graded_bets": 0},
                      {"production_eligible": False}, board)
    assert report["historical_sample"]["roi"] > 0
    assert report["historical_sample"]["verified_entry_quotes"] == 0
    assert report["retrospective_2025_holdout"]["bets"] == 1
    assert report["research_candidates"][0]["quote_verified"] is False
    assert report["profitability_proven"] is False
    assert report["bet_approved_by_this_report"] is False


def test_forward_aggregate_cannot_certify_specific_strategy():
    bets = pd.DataFrame([row(2023, .91, True), row(2024, .91, True),
                         row(2025, .91, True)])
    forward = {"clean_entries": 400, "graded_bets": 200,
               "by_subgroup": {"favorite|home": {
                   "graded_bets": 120, "roi_ci_95": [0.1, 0.2],
                   "avg_execution_clv": 0.04}}}
    report = evaluate(bets, forward, {"production_eligible": True}, pd.DataFrame())
    assert report["forward_subgroup_preliminary_gate"] is True
    assert report["candidate_specific_independent_evidence"] == "NOT_AVAILABLE"
    assert report["profitability_proven"] is False


def test_missing_quote_verification_column_fails_closed():
    data = pd.DataFrame([row(2025, 0.91)]).drop(columns=["entry_quote_verified"])
    with pytest.raises(ValueError, match="entry_quote_verified"):
        evaluate(data, {}, {}, pd.DataFrame())
