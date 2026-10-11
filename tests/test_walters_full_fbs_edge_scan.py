"""One game -> one clearly-labelled research-only edge audit record."""
from datetime import UTC, datetime, timedelta

import pytest

from scripts.walters_full_fbs_edge_scan import build

NOW = datetime(2026, 10, 10, 20, 0, tzinfo=UTC)


def forecast(gid="one", *, scope="FBS_VS_FBS", signal="BET",
             raw_ev=0.35, market_line=-7.5):
    return {
        "season": "2026", "week": "6", "game_id": gid,
        "date": (NOW + timedelta(hours=10)).isoformat(),
        "away_team": f"A-{gid}", "home_team": f"H-{gid}",
        "fbs_matchup_scope": scope,
        "fbs_model_validation": (
            "STANDARD_RESEARCH_GATES"
            if scope == "FBS_VS_FBS"
            else "NO_BET_UNVALIDATED_OPPONENT_CLASS"
        ),
        "away_score_exact": "24.2", "home_score_exact": "31.8",
        "model_margin_home": "7.6", "model_total": "56.0",
        "market_spread_home": market_line,
        "quant_market": "spread", "quant_side": f"H-{gid}",
        "quant_price": market_line, "quant_odds": "-110",
        "quant_book": "DraftKings",
        "quant_ev": raw_ev, "quant_signal": signal,
        "market_execution_verified": "false",
        "market_quote_timestamp_verified": "false",
    }


def shortlist(gid):
    return {
        "game_id": gid, "decision": "NO_BET",
        "conservative_ev": "-.03",
        "reasons": "PROVIDER_TIMESTAMP_UNVERIFIED;NOT_ROBUST_CORE"
    }


def coverage(n):
    return {
        "season": 2026, "week": 6,
        "scope": "EVERY_FBS_INVOLVED_REGULAR_SEASON_MATCHUP",
        "model_cutoff_utc": NOW.isoformat(),
        "total_scheduled_fbs_games": n+2,
        "counts": {"PREDICTED": n, "STARTED_EXCLUDED": 1,
                   "COMPLETED_EXCLUDED": 1},
        "remaining_pregame_games": n,
        "remaining_pregame_coverage_complete": True,
    }


GATE = {
    "production_eligible": False, "release_state": "RESEARCH",
    "blockers": ["entry quote not verified"]
}


def test_every_game_scanned_not_just_shortlisted_one():
    rows = [forecast("1"), forecast("2", signal="PASS"),
            forecast("3", raw_ev=-.1)]
    out, summary = build(
        rows, [shortlist("1")], coverage(3), GATE, evaluated_at=NOW
    )
    assert len(out) == 3
    assert summary["total_fbs_involved_weekly_games"] == 5
    assert summary["games_shortlisted_by_independent_edge_gate"] == 1
    assert summary["games_with_raw_positive_model_hypothesis"] == 1
    assert summary["games_without_verified_book_execution"] == 3
    assert all(x["recommended_bet"] is False for x in out)
    assert all(x["model_home_margin"] == 7.6 for x in out)
    assert all(x["blockers"] for x in out)
    assert summary["automated_wagers_placed"] == 0
    assert out[0]["conservative_edge_ev"] == -.03
    assert out[1]["final_edge_decision"] == "NOT_SHORTLISTED"


def test_unvalidated_fbs_vs_fcs_cannot_be_raw_edge():
    r = forecast("other", scope="FBS_VS_NON_FBS_UNVALIDATED",
                 raw_ev=15.0, signal="STRONG")
    out, summary = build([r], [shortlist("other")], coverage(1), GATE,
                         evaluated_at=NOW)
    assert out[0]["research_status"] == "UNVALIDATED_OPPONENT_CLASS_NO_BET"
    assert "UNVALIDATED_FBS_NON_FBS_MATCHUP" in out[0]["blockers"]
    assert out[0]["recommended_bet"] is False
    assert summary["games_with_raw_positive_model_hypothesis"] == 0


def test_empty_future_slate_valid_when_source_all_started():
    out, summary = build([], [], coverage(0), GATE, evaluated_at=NOW)
    assert out == []
    assert summary["remaining_pregame_game_count"] == 0
    assert summary["bet_recommendations"] == 0


def test_missing_model_game_vs_coverage_fails_loudly():
    with pytest.raises(ValueError, match="mismatches"):
        build([forecast("1")], [], coverage(3), GATE, evaluated_at=NOW)


def test_unapproved_final_shortlist_extra_game_fails():
    with pytest.raises(ValueError, match="not in pregame"):
        build([forecast("1")], [shortlist("other")], coverage(1),
              GATE, evaluated_at=NOW)


def test_duplicate_game_rows_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        build([forecast(), forecast()], [], coverage(2), GATE,
              evaluated_at=NOW)
    with pytest.raises(ValueError, match="Duplicate"):
        build([forecast()], [shortlist("one"), shortlist("one")],
              coverage(1), GATE, evaluated_at=NOW)


def test_inconsistent_forecast_is_never_published():
    row = forecast()
    row["model_margin_home"] = "999"
    with pytest.raises(ValueError, match="Score/margin"):
        build([row], [], coverage(1), GATE, evaluated_at=NOW)


def test_missing_scope_is_no_bet_and_blocks_output():
    row = forecast()
    row.pop("fbs_matchup_scope")
    with pytest.raises(ValueError, match="classification"):
        build([row], [], coverage(1), GATE, evaluated_at=NOW)


def test_missing_quote_remains_non_actionable():
    row = forecast(raw_ev="", market_line="")
    out, summary = build([row], [], coverage(1), GATE, evaluated_at=NOW)
    assert out[0]["published_market_home_spread"] is None
    assert out[0]["raw_model_ev_hypothesis"] is None
    assert out[0]["research_status"] == "NO_MODEL_EDGE_HYPOTHESIS"
    assert "NO_PUBLISHED_MARKET_SPREAD" in out[0]["blockers"]
    assert "NO_PRICED_MODEL_CANDIDATE" in out[0]["blockers"]
    assert summary["games_with_raw_positive_model_hypothesis"] == 0


def test_fail_on_missing_release_authority():
    with pytest.raises(ValueError, match="release gate"):
        build([forecast()], [], coverage(1), {}, evaluated_at=NOW)


def test_started_game_must_never_reenter_current_edge_scan():
    row = forecast()
    row["date"] = NOW.isoformat()
    with pytest.raises(ValueError, match="already-started"):
        build([row], [], coverage(1), GATE, evaluated_at=NOW)


def test_no_future_time_machine():
    with pytest.raises(ValueError, match="timezone-aware"):
        build([], [], coverage(0), GATE,
              evaluated_at=datetime(2026, 10, 10, 20, 0))


def test_raw_positive_is_research_hypothesis_not_bet():
    row = forecast(raw_ev=2.7)
    row["market_execution_verified"] = "true"
    row["market_quote_timestamp_verified"] = "true"
    gate = {"production_eligible": True, "release_state": "PRODUCTION"}
    out, summary = build([row], [shortlist("one")], coverage(1), gate,
                         evaluated_at=NOW)
    assert out[0]["recommended_bet"] is False
    assert summary["bet_recommendations"] == 0
    assert summary["production_eligible"] is True
    assert out[0]["research_status"] == "RAW_EDGE_HYPOTHESIS_UNVERIFIED"


def test_nonempty_research_ledger_requires_model_period_coherence():
    row = forecast()
    row["game_id"] = ""
    with pytest.raises(ValueError, match="missing game_id"):
        build([row], [], coverage(1), GATE, evaluated_at=NOW)
