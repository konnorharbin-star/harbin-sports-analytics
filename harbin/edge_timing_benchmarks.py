"""Prospective, paired execution-timing benchmarks for CFB research signals.

Compares the FIRST frozen BET_NOW/WAIT signal against ALWAYS_NOW,
ALWAYS_WAIT and a decision-time available majority-direction policy.
All policies face the SAME initial candidate and same-book 6-9 hour quote.
No certified close, actual fills, profitability, fair probability or staking
claims can be inferred from these observational prices.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .edge_timing_forward import (
    GRADED_COLUMNS, PRIMARY_HORIZON_HOURS, PRIMARY_TOLERANCE_HOURS,
    MIN_REVIEW_OBSERVATIONS, MIN_REVIEW_DISTINCT_GAMES,
    MIN_REVIEW_DISTINCT_WEEKS, TIMING_ACTIONS, _movement, _ts, _num,
)

SCHEMA_VERSION = 1
ACTIONS = ("BET_NOW_RESEARCH", "WAIT_MONITOR")
MOVEMENTS = ("BETTER", "WORSE")
MIN_BASELINE_TRAIN_OBS = 20
MIN_BASELINE_TRAIN_WEEKS = 3
MIN_COVERAGE = 0.80
MIN_ACTION_COVERAGE = 0.70
BLOCK_BOOTSTRAP_REPS = 2000
BLOCK_SEED = 20261007

OUTPUT_COLUMNS = [
    "decision_id", "game_id", "book", "side", "decision_at", "kickoff",
    "kickoff_week", "timing_action", "primary_status", "primary_observed_at",
    "primary_movement", "primary_integrity", "primary_eligible",
    "primary_quote_observed", "primary_direction_scored",
    "model_direction_correct", "always_now_correct", "always_wait_correct",
    "model_minus_always_now", "model_minus_always_wait",
    "chrono_action", "chrono_train_rows", "chrono_train_weeks",
    "chrono_direction_correct", "model_minus_chrono",
    "near_kickoff_status", "near_kickoff_movement", "near_kickoff_integrity",
    "near_kickoff_direction_scored", "near_kickoff_model_correct",
    "near_kickoff_always_now_correct", "near_kickoff_always_wait_correct",
]


def _load(path):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame(columns=GRADED_COLUMNS)
    return pd.read_csv(path, low_memory=False)


def _dt(row, key):
    return _ts(row.get(key))


def _strict_observation(row, prefix):
    """Re-audit observation timing and price direction; fail closed on drift."""
    status = str(row.get(prefix + "_status") or "")
    if status != "OBSERVED":
        return ("NOT_OBSERVED", None)
    decision, kickoff = _dt(row, "decision_at"), _dt(row, "kickoff")
    observed = _dt(row, prefix + "_observed_at")
    if pd.isna(decision) or pd.isna(kickoff) or pd.isna(observed):
        return ("INVALID_OBSERVATION", None)
    if not decision < observed < kickoff:
        return ("INVALID_OBSERVATION", None)
    if prefix == "primary":
        window_begin = decision + pd.Timedelta(hours=PRIMARY_HORIZON_HOURS)
        window_end = window_begin + pd.Timedelta(hours=PRIMARY_TOLERANCE_HOURS)
        if not window_begin <= observed <= window_end:
            return ("INVALID_OBSERVATION", None)
    elif prefix == "near_kickoff":
        if observed < kickoff - pd.Timedelta(minutes=60):
            return ("INVALID_OBSERVATION", None)
    else:
        raise ValueError("Only predeclared endpoints may be benchmarked")
    if not str(row.get(prefix + "_source") or "") in (
        "candidate_quote", "captured_market_quote"
    ):
        return ("INVALID_OBSERVATION", None)
    if str(row.get(prefix + "_book") or "") != str(row.get("book") or ""):
        return ("INVALID_OBSERVATION", None)
    entry_line, entry_odds = _num(row.get("entry_line")), _num(row.get("entry_odds"))
    future_line, future_odds = _num(row.get(prefix + "_line")), _num(row.get(prefix + "_odds"))
    if not all(np.isfinite(v) for v in (entry_line, entry_odds, future_line, future_odds)):
        return ("INVALID_OBSERVATION", None)
    if entry_odds == 0 or future_odds == 0:
        return ("INVALID_OBSERVATION", None)
    actual, line_delta, price_delta = _movement(
        entry_line, entry_odds, future_line, future_odds
    )
    if actual != str(row.get(prefix + "_movement") or ""):
        return ("INVALID_OBSERVATION", None)
    # Check the supposedly independent grader's changes as well. Published
    # numbers are rounded to three decimals; allow that rounding, not drift.
    recorded_line = _num(row.get(prefix + "_line_change"))
    recorded_price = _num(row.get(prefix + "_implied_change_pp"))
    if (not np.isfinite(recorded_line) or not np.isfinite(recorded_price)
            or abs(recorded_line - line_delta) > 0.0011
            or abs(recorded_price - price_delta) > 0.0011):
        return ("INVALID_OBSERVATION", None)
    return ("VALID", actual)


def _week(ts):
    d = ts.isocalendar()
    return f"{int(d.year):04d}-W{int(d.week):02d}"


def _baseline_score(action, movement):
    if movement not in MOVEMENTS or action not in ACTIONS:
        return None
    return int((action == "BET_NOW_RESEARCH") == (movement == "WORSE"))


def _clean_forward(frame):
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    missing = set(GRADED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Forward timing grade schema missing: " + ", ".join(sorted(missing)))
    data = frame.copy()
    data["_decision"] = pd.to_datetime(data["decision_at"], utc=True, errors="coerce")
    data["_kick"] = pd.to_datetime(data["kickoff"], utc=True, errors="coerce")
    data["_quoted"] = pd.to_datetime(data["quote_at"], utc=True, errors="coerce")
    data["_odds"] = pd.to_numeric(data["entry_odds"], errors="coerce")
    data["_line"] = pd.to_numeric(data["entry_line"], errors="coerce")
    age = (data["_decision"] - data["_quoted"]).dt.total_seconds() / 60
    valid = (
        data["decision_id"].notna() & data["game_id"].notna()
        & data["book"].notna() & data["side"].notna()
        & data["timing_action"].isin(ACTIONS)
        & data["decision_mode"].eq("RESEARCH_ONLY_NO_STAKING")
        & data["market"].eq("spread")
        & data["_decision"].notna() & data["_kick"].notna()
        & data["_quoted"].notna() & data["_line"].notna()
        & data["_odds"].notna() & data["_odds"].ne(0)
        & age.between(0, 120)
        & (data["_decision"] < data["_kick"])
    )
    if not bool(valid.all()):
        raise ValueError("Forward timing grade contains invalid frozen decisions")
    if data.duplicated(["decision_id"]).any():
        raise ValueError("Forward timing grade contains duplicate frozen decisions")
    data = data.sort_values(["_decision", "decision_id"]).reset_index(drop=True)
    new = []
    for _, row in data.iterrows():
        kickoff, decision = row["_kick"], row["_decision"]
        eligible = decision + pd.Timedelta(hours=PRIMARY_HORIZON_HOURS) < kickoff
        primary_integrity, primary_move = _strict_observation(row, "primary")
        near_integrity, near_move = _strict_observation(row, "near_kickoff")
        conclusive = primary_integrity == "VALID" and primary_move in MOVEMENTS
        near_conclusive = near_integrity == "VALID" and near_move in MOVEMENTS
        model = _baseline_score(row["timing_action"], primary_move) if conclusive else None
        now = _baseline_score("BET_NOW_RESEARCH", primary_move) if conclusive else None
        wait = _baseline_score("WAIT_MONITOR", primary_move) if conclusive else None
        near_model = _baseline_score(row["timing_action"], near_move) if near_conclusive else None
        near_now = _baseline_score("BET_NOW_RESEARCH", near_move) if near_conclusive else None
        near_wait = _baseline_score("WAIT_MONITOR", near_move) if near_conclusive else None
        new.append({
            "decision_id": str(row["decision_id"]),
            "game_id": str(row["game_id"]),
            "book": str(row["book"]),
            "side": str(row["side"]),
            "decision_at": decision.isoformat(),
            "kickoff": kickoff.isoformat(),
            "kickoff_week": _week(kickoff),
            "timing_action": str(row["timing_action"]),
            "primary_status": str(row["primary_status"]),
            "primary_observed_at": (
                _dt(row, "primary_observed_at").isoformat()
                if primary_integrity == "VALID" else None
            ),
            "primary_movement": primary_move if primary_integrity == "VALID" else None,
            "primary_integrity": primary_integrity,
            "primary_eligible": bool(eligible),
            "primary_quote_observed": bool(primary_integrity == "VALID"),
            "primary_direction_scored": bool(conclusive),
            "model_direction_correct": model,
            "always_now_correct": now,
            "always_wait_correct": wait,
            "model_minus_always_now": model - now if conclusive else None,
            "model_minus_always_wait": model - wait if conclusive else None,
            "chrono_action": None,
            "chrono_train_rows": 0,
            "chrono_train_weeks": 0,
            "chrono_direction_correct": None,
            "model_minus_chrono": None,
            "near_kickoff_status": str(row["near_kickoff_status"]),
            "near_kickoff_movement": near_move if near_integrity == "VALID" else None,
            "near_kickoff_integrity": near_integrity,
            "near_kickoff_direction_scored": bool(near_conclusive),
            "near_kickoff_model_correct": near_model,
            "near_kickoff_always_now_correct": near_now,
            "near_kickoff_always_wait_correct": near_wait,
        })
    out = pd.DataFrame(new, columns=OUTPUT_COLUMNS)
    # Strictly forward-only training: every training quote was observed before
    # THIS decision and belongs to an earlier kickoff week. Different current
    # decisions can have different, timestamp-correct training sets.
    conclusive_rows = out[out["primary_direction_scored"]].copy()
    for i, row in out.iterrows():
        historical = conclusive_rows[
            conclusive_rows["kickoff_week"].lt(row["kickoff_week"])
            & pd.to_datetime(conclusive_rows["primary_observed_at"], utc=True)
              .lt(_ts(row["decision_at"]))
        ]
        n, weeks = len(historical), historical["kickoff_week"].nunique()
        out.at[i, "chrono_train_rows"] = int(n)
        out.at[i, "chrono_train_weeks"] = int(weeks)
        if n < MIN_BASELINE_TRAIN_OBS or weeks < MIN_BASELINE_TRAIN_WEEKS:
            continue
        worse = int(historical["primary_movement"].eq("WORSE").sum())
        better = int(historical["primary_movement"].eq("BETTER").sum())
        if worse == better:
            continue
        action = "BET_NOW_RESEARCH" if worse > better else "WAIT_MONITOR"
        out.at[i, "chrono_action"] = action
        if bool(row["primary_direction_scored"]):
            reference = _baseline_score(action, row["primary_movement"])
            out.at[i, "chrono_direction_correct"] = reference
            out.at[i, "model_minus_chrono"] = (
                int(row["model_direction_correct"]) - reference
            )
    return out


def _bootstrap_lift(frame, diff_col):
    valid = frame.dropna(subset=[diff_col]).copy()
    if len(valid) < MIN_REVIEW_OBSERVATIONS:
        return None
    groups = [
        values[diff_col].astype(float).to_numpy()
        for _, values in valid.groupby("kickoff_week", sort=True)
    ]
    if len(groups) < MIN_REVIEW_DISTINCT_WEEKS:
        return None
    rng = np.random.default_rng(BLOCK_SEED)
    means = np.empty(BLOCK_BOOTSTRAP_REPS)
    for j in range(BLOCK_BOOTSTRAP_REPS):
        picks = rng.integers(0, len(groups), size=len(groups))
        observations = np.concatenate([groups[k] for k in picks])
        means[j] = float(observations.mean())
    return [round(float(v), 5) for v in np.percentile(means, [2.5, 97.5])]


def _rate(frame, column):
    data = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(data.mean()) if len(data) else None


def _group(frame):
    eligible = frame[frame["primary_eligible"]] if not frame.empty else frame
    observed = eligible[eligible["primary_quote_observed"]]
    scored = eligible[eligible["primary_direction_scored"]]
    chronological = scored[scored["chrono_direction_correct"].notna()]
    n_eligible = len(eligible)
    matured = eligible[eligible["primary_status"].ne("PENDING_HORIZON")]
    scores = {
        "decisions": int(len(frame)),
        "primary_eligible": int(n_eligible),
        "matured_primary_windows": int(len(matured)),
        "primary_observed": int(len(observed)),
        "primary_coverage_on_matured": (
            float(len(observed) / len(matured)) if len(matured) else None
        ),
        "primary_matured_missing": int((matured["primary_status"] == "NO_HORIZON_QUOTE").sum()),
        "invalid_observations": int((frame["primary_integrity"] == "INVALID_OBSERVATION").sum()),
        "conclusive_paired": int(len(scored)),
        "primary_flat_or_mixed": int(len(observed) - len(scored)),
        "model_accuracy": _rate(scored, "model_direction_correct"),
        "always_now_accuracy": _rate(scored, "always_now_correct"),
        "always_wait_accuracy": _rate(scored, "always_wait_correct"),
        "lift_vs_always_now": _rate(scored, "model_minus_always_now"),
        "lift_vs_always_wait": _rate(scored, "model_minus_always_wait"),
        "chronological_baseline_evaluated": int(len(chronological)),
        "chronological_baseline_accuracy": _rate(chronological, "chrono_direction_correct"),
        "signal_on_chronological_subset_accuracy": _rate(chronological, "model_direction_correct"),
        "lift_vs_chronological_baseline": _rate(chronological, "model_minus_chrono"),
        "near_kickoff_conclusive": int(frame["near_kickoff_direction_scored"].sum()) if len(frame) else 0,
        "near_kickoff_model_accuracy": _rate(frame, "near_kickoff_model_correct"),
        "near_kickoff_always_now_accuracy": _rate(frame, "near_kickoff_always_now_correct"),
        "near_kickoff_always_wait_accuracy": _rate(frame, "near_kickoff_always_wait_correct"),
    }
    return scores


def _summary(frame, source):
    overall = _group(frame)
    by_action = {
        action: _group(frame[frame["timing_action"].eq(action)])
        for action in ACTIONS
    }
    scored = frame[frame["primary_direction_scored"]].copy()
    weeks = int(scored["kickoff_week"].nunique()) if len(scored) else 0
    games = int(scored["game_id"].nunique()) if len(scored) else 0
    confidence = {
        "lift_vs_always_now_95": _bootstrap_lift(scored, "model_minus_always_now"),
        "lift_vs_always_wait_95": _bootstrap_lift(scored, "model_minus_always_wait"),
        "lift_vs_chronological_95": _bootstrap_lift(scored, "model_minus_chrono"),
        "method": "deterministic_kickoff_week_cluster_bootstrap",
        "resamples": BLOCK_BOOTSTRAP_REPS,
        "published_only_after": {
            "conclusive_paired": MIN_REVIEW_OBSERVATIONS,
            "distinct_weeks": MIN_REVIEW_DISTINCT_WEEKS,
        },
    }
    matured = overall["matured_primary_windows"]
    primary_cov = overall["primary_coverage_on_matured"]
    meets_design = (
        len(scored) >= MIN_REVIEW_OBSERVATIONS
        and games >= MIN_REVIEW_DISTINCT_GAMES
        and weeks >= MIN_REVIEW_DISTINCT_WEEKS
        and all(by_action[a]["conclusive_paired"] >= 20 for a in ACTIONS)
    )
    book_coverage_ok = (
        matured > 0 and primary_cov is not None and primary_cov >= MIN_COVERAGE
        and all(
            by_action[a]["primary_coverage_on_matured"] is not None
            and by_action[a]["primary_coverage_on_matured"] >= MIN_ACTION_COVERAGE
            for a in ACTIONS
        )
    )
    chronological_ok = overall["chronological_baseline_evaluated"] >= 30
    ci_ok = all(
        confidence[key] is not None and confidence[key][0] > 0
        for key in (
            "lift_vs_always_now_95", "lift_vs_always_wait_95",
            "lift_vs_chronological_95",
        )
    )
    # This never authorizes a bet or promotes the timing strategy.
    status = (
        "DATA_INTEGRITY_ERROR" if overall["invalid_observations"] else
        "PENDING_FORWARD" if overall["decisions"] == 0 else
        "EARLY_SAMPLE" if not meets_design else
        "INSUFFICIENT_PRICE_COVERAGE" if not book_coverage_ok else
        "NO_CHRONOLOGICAL_BENCHMARK" if not chronological_ok else
        "NO_CONFIRMED_BASELINE_LIFT" if not ci_ok else
        "INDEPENDENT_REVIEW_ELIGIBLE_NOT_APPROVED"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "research_only": True,
        "betting_authorized": False,
        "approved_units": 0,
        "no_official_close_claim": True,
        "no_executed_fill_claim": True,
        "economic_value_established": False,
        "source": str(source),
        "design": {
            "cohort": "same_frozen_BET_NOW_WAIT_decisions_in_primary_timing_ledger",
            "baseline_always_now": "take_frozen_initial_same_book_offer",
            "baseline_always_wait": "take_first_observed_same_book_offer_6_to_9h_later",
            "baseline_chronological": (
                "majority_direction_from_conclusive_prior_kickoff_weeks_with_"
                "observation_timestamp_before_current_decision;_otherwise_abstain"
            ),
            "metric": "correct_direction_only_on_unambiguous_paired_price_moves",
            "paired_missingness": "missing_wait_offer_unscored_and_reported",
            "noncausal": True,
            "signal_selection": "first_frozen_signal_only;_not_a_comparison_to_unselected_games",
            "primary_horizon_hours": PRIMARY_HORIZON_HOURS,
            "primary_tolerance_hours": PRIMARY_TOLERANCE_HOURS,
            "clustered_uncertainty": "kickoff_week_bootstrap_only_when_sample_gate_met",
            "review_gate": {
                "conclusive_observations": MIN_REVIEW_OBSERVATIONS,
                "games": MIN_REVIEW_DISTINCT_GAMES,
                "weeks": MIN_REVIEW_DISTINCT_WEEKS,
                "conclusive_per_action": 20,
                "primary_quote_coverage": MIN_COVERAGE,
                "per_action_primary_coverage": MIN_ACTION_COVERAGE,
                "chronological_baseline_evaluated": 30,
                "positive_week_bootstrap_CI_lower_bound_for_all_three_baselines": True,
            },
        },
        "distinct_scored_games": games,
        "distinct_scored_weeks": weeks,
        "overall": overall,
        "by_action": by_action,
        "confidence": confidence,
        "limitations": [
            "Always-WAIT depends on a subsequently observed quote; quote existence is not a guaranteed fill.",
            "Directional accuracy is not EV, CLV or ROI; spread and odds trade-offs are excluded.",
            "Same-game candidate selection can cause selection bias; findings are not causal.",
            "Missing future book prices may be nonrandom; report coverage before benchmarking.",
            "Independent audit and calibrated probability/price utility are prerequisites to any economic claim.",
            "No parameter tuning on prospective data and no automatic production promotion.",
        ],
    }


def benchmark_timing_forward(
    graded_path="reports/edge_timing_forward_graded.csv",
    reports_dir="reports",
):
    """Produce reproducible research benchmark report and case-level paired ledger."""
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    rows = _clean_forward(_load(graded_path))
    summary = _summary(rows, graded_path)
    rows.to_csv(reports / "edge_timing_baseline_graded.csv", index=False)
    (reports / "edge_timing_baseline_performance.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
