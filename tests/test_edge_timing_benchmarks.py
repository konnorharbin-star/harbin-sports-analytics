"""No-tuning, time-safe benchmark tests: frozen rows, paired counterfactuals and audit."""
import json

import pandas as pd
import pytest

from harbin.edge_timing_forward import (
    GRADED_COLUMNS, _movement, append_timing_decisions, grade_timing_forward,
)
from harbin.edge_timing_benchmarks import (
    _clean_forward, _bootstrap_lift, benchmark_timing_forward,
)


def frozen(game, action, kickoff="2026-10-10T20:00:00Z", decision="2026-10-07T20:00:00Z"):
    return {
        "game_id": game, "date": kickoff, "away_team": "Away", "home_team": "Home",
        "market": "spread", "side": "Home", "book": "Book A",
        "line": -9.5, "odds": -106, "quote_at": "2026-10-07T19:58:00Z",
        "edge_priority": "ROBUST_CORE", "edge_reliability_status": "SUPPORTED_SUBGROUP",
        "price_evidence_status": "CONFIRMED", "probability": 0.66,
        "ev": .24, "edge": 6.8, "bet_to_line": -10,
        "conservative_price_margin": .10, "timing_action": action,
        "timing_status": "RESEARCH_ONLY", "timing_reason": "TEST",
    }


def market(game, stamp, future_line, book="Book A"):
    return {
        "captured_at": stamp, "game_id": game,
        "kickoff": "2026-10-10T20:00:00Z",
        "home_team": "Home", "away_team": "Away",
        "market_quotes_json": json.dumps([
            {"provider": book, "home_spread": future_line,
             "home_spread_price": -106, "away_spread_price": -110}
        ]),
    }


def grade_fixture(tmp_path, include_g2=True):
    ledger = tmp_path / "ledger.csv"
    candidates = [frozen("g1", "BET_NOW_RESEARCH")]
    if include_g2:
        candidates.append(frozen("g2", "WAIT_MONITOR"))
    saved = append_timing_decisions(pd.DataFrame(candidates), ledger,
                                    "2026-10-07T20:00:00Z")
    assert saved["appended"] == len(candidates)
    entries = [market("g1", "2026-10-08T02:15:00Z", -10)]
    if include_g2:
        entries.append(market("g2", "2026-10-08T02:20:00Z", -9))
    pd.DataFrame(entries).to_csv(tmp_path / "market.csv", index=False)
    grade_timing_forward(
        ledger, tmp_path / "market.csv", tmp_path / "no_candidate.csv",
        tmp_path / "reports", as_of="2026-10-11T00:00:00Z",
    )
    return tmp_path / "reports" / "edge_timing_forward_graded.csv"


def test_pairwise_comparison_on_identical_candidates_and_quotes(tmp_path):
    graded = grade_fixture(tmp_path)
    report = benchmark_timing_forward(graded, tmp_path / "reports")
    cases = pd.read_csv(tmp_path / "reports" / "edge_timing_baseline_graded.csv")
    assert len(cases) == 2
    assert report["status"] == "EARLY_SAMPLE"
    assert report["overall"]["conclusive_paired"] == 2
    assert report["overall"]["model_accuracy"] == 1.0
    assert report["overall"]["always_now_accuracy"] == .5
    assert report["overall"]["always_wait_accuracy"] == .5
    assert report["overall"]["lift_vs_always_now"] == .5
    assert report["overall"]["lift_vs_always_wait"] == .5
    assert report["overall"]["primary_coverage_on_matured"] == 1.0
    assert report["overall"]["chronological_baseline_evaluated"] == 0
    assert report["confidence"]["lift_vs_always_now_95"] is None
    assert not report["betting_authorized"]
    assert not report["economic_value_established"]
    assert cases.loc[cases.game_id.eq("g1"), "model_minus_always_wait"].iloc[0] == 1
    assert cases.loc[cases.game_id.eq("g2"), "model_minus_always_now"].iloc[0] == 1


def test_empty_ledger_baseline_report_not_a_performance_claim(tmp_path):
    report = benchmark_timing_forward(tmp_path / "missing.csv", tmp_path)
    assert report["status"] == "PENDING_FORWARD"
    assert report["overall"]["decisions"] == 0
    assert report["overall"]["model_accuracy"] is None
    assert (tmp_path / "edge_timing_baseline_graded.csv").exists()


def valid_grade(game, kickoff, decision, action="BET_NOW_RESEARCH",
                line=-10, quote="2026-10-01T10:58:00Z"):
    observed = (pd.Timestamp(decision) + pd.Timedelta(hours=6, minutes=15)).isoformat()
    movement, ld, pp = _movement(-9.5, -106, line, -106)
    row = dict.fromkeys(GRADED_COLUMNS)
    row.update({
        "schema_version": 1, "decision_id": game + "|spread|Home",
        "game_id": game, "decision_at": decision, "quote_at": quote,
        "kickoff": kickoff, "market": "spread", "side": "Home",
        "book": "Book A", "entry_line": -9.5, "entry_odds": -106,
        "decision_mode": "RESEARCH_ONLY_NO_STAKING", "timing_action": action,
        "primary_status": "OBSERVED", "primary_observed_at": observed,
        "primary_source": "captured_market_quote", "primary_book": "Book A",
        "primary_line": line, "primary_odds": -106,
        "primary_line_change": ld, "primary_implied_change_pp": pp,
        "primary_movement": movement,
        "near_kickoff_status": "PENDING_KICKOFF",
    })
    return row


def test_chronological_baseline_cannot_train_on_not_yet_observed_quotes():
    earlier = [
        valid_grade(
            f"past{i}", ("2026-10-10T20:00:00Z" if i < 8 else
                         "2026-10-17T20:00:00Z" if i < 16 else
                         "2026-10-24T20:00:00Z"),
            "2026-10-01T12:00:00Z", quote="2026-10-01T11:58:00Z",
        ) for i in range(24)
    ]
    too_early = valid_grade(
        "early", "2026-11-14T20:00:00Z", "2026-10-01T09:00:00Z",
        action="WAIT_MONITOR", line=-9,
        quote="2026-10-01T08:58:00Z",
    )
    late = valid_grade(
        "late", "2026-11-14T20:00:00Z", "2026-11-01T12:00:00Z",
        action="WAIT_MONITOR", line=-9,
        quote="2026-11-01T11:58:00Z",
    )
    out = _clean_forward(pd.DataFrame([too_early] + earlier + [late]))
    early_result = out[out["game_id"].eq("early")].iloc[0]
    late_result = out[out["game_id"].eq("late")].iloc[0]
    assert pd.isna(early_result["chrono_action"])
    assert early_result["chrono_train_rows"] == 0
    assert late_result["chrono_train_rows"] == 24
    assert late_result["chrono_train_weeks"] == 3
    assert late_result["chrono_action"] == "BET_NOW_RESEARCH"
    assert late_result["model_minus_chrono"] == 1


def test_book_mismatch_corrupted_price_and_future_timestamps_fail_closed():
    clean = valid_grade("a", "2026-10-10T20:00:00Z",
                        "2026-10-01T12:00:00Z")
    wrong_book = dict(clean, decision_id="b|spread|Home", game_id="b",
                      primary_book="Book B")
    wrong_price = dict(clean, decision_id="c|spread|Home", game_id="c",
                       primary_implied_change_pp=9)
    after_kickoff = dict(clean, decision_id="d|spread|Home", game_id="d",
                         primary_observed_at="2026-10-11T20:00:00Z")
    frame = _clean_forward(pd.DataFrame(
        [clean, wrong_book, wrong_price, after_kickoff]
    ))
    assert frame["primary_direction_scored"].sum() == 1
    assert frame["primary_integrity"].value_counts().to_dict() == {
        "INVALID_OBSERVATION": 3, "VALID": 1
    }


def test_absent_wait_opportunity_not_counted_as_success(tmp_path):
    graded = grade_fixture(tmp_path, include_g2=False)
    out = pd.read_csv(graded)
    out.loc[0, "primary_status"] = "NO_HORIZON_QUOTE"
    out.to_csv(graded, index=False)
    report = benchmark_timing_forward(graded, tmp_path / "reports")
    assert report["overall"]["primary_observed"] == 0
    assert report["overall"]["matured_primary_windows"] == 1
    assert report["overall"]["primary_matured_missing"] == 1
    assert report["overall"]["primary_coverage_on_matured"] == 0.0
    assert report["overall"]["model_accuracy"] is None


def test_flat_and_mixed_are_not_false_directional_wins():
    a = valid_grade("a", "2026-10-10T20:00:00Z", "2026-10-01T12:00:00Z", line=-9.5)
    b = valid_grade("b", "2026-10-10T20:00:00Z", "2026-10-01T12:00:00Z", line=-9)
    b["primary_odds"] = -145
    label, ld, pp = _movement(-9.5, -106, -9, -145)
    b.update(primary_movement=label, primary_line_change=ld,
             primary_implied_change_pp=pp)
    out = _clean_forward(pd.DataFrame([a, b]))
    assert out["primary_quote_observed"].sum() == 2
    assert out["primary_direction_scored"].sum() == 0
    assert set(out["primary_movement"]) == {"FLAT", "MIXED_LINE_PRICE"}


def test_duplicate_or_malformed_frozen_decisions_raise():
    a = valid_grade("a", "2026-10-10T20:00:00Z", "2026-10-01T12:00:00Z")
    with pytest.raises(ValueError, match="duplicate"):
        _clean_forward(pd.DataFrame([a, a]))
    with pytest.raises(ValueError, match="schema"):
        _clean_forward(pd.DataFrame([{"decision_id": "g1"}]))


def test_week_block_bootstrap_only_after_threshold_and_is_reproducible():
    rows = []
    for i in range(104):
        rows.append({
            "kickoff_week": f"2026-W{1 + i // 13:02d}",
            "model_minus_always_now": 1 if i % 5 else -1,
        })
    full = pd.DataFrame(rows)
    assert _bootstrap_lift(full, "model_minus_always_now") == _bootstrap_lift(
        full, "model_minus_always_now"
    )
    assert _bootstrap_lift(full.iloc[:99], "model_minus_always_now") is None
    only_seven_weeks = full[~full["kickoff_week"].eq("2026-W08")]
    assert _bootstrap_lift(only_seven_weeks, "model_minus_always_now") is None
