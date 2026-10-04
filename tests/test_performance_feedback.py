from __future__ import annotations

import csv
import json

import pandas as pd

from harbin.performance_feedback import build_performance_feedback, performance_feedback_for_row
from harbin.policy import DEFAULT_POLICY
from harbin.portfolio import apply_portfolio_controls


FIELDS = [
    "game_id",
    "quant_market",
    "quant_book",
    "quant_signal",
    "quant_edge",
    "profit",
    "execution_clv",
    "kickoff",
]


def _history(
    path,
    *,
    n=40,
    profit=-1.0,
    clv=-0.5,
    market="spread",
    book="Book A",
):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i in range(n):
            writer.writerow(
                {
                    "game_id": f"g{i}",
                    "quant_market": market,
                    "quant_book": book,
                    "quant_signal": "BET",
                    "quant_edge": 0.06,
                    "profit": profit,
                    "execution_clv": clv,
                    "kickoff": f"2026-09-{1 + (i % 28):02d}T17:00:00Z",
                }
            )
    return path


def _config():
    return {
        "enable_performance_feedback": True,
        "feedback_min_segment_bets": 20,
        "feedback_min_clv_coverage": 0.60,
        "feedback_weak_multiplier": 0.75,
        "feedback_severe_multiplier": 0.50,
        "feedback_severe_min_bets": 40,
        "feedback_severe_roi": -0.05,
        "feedback_severe_positive_clv_rate": 0.45,
    }


def _feedback_row():
    return {
        "quant_market": "spread",
        "quant_book": "Book A",
        "quant_signal": "BET",
        "quant_edge": 0.06,
    }


def test_small_sample_does_not_change_risk(tmp_path):
    path = _history(tmp_path / "graded.csv", n=10)
    report = build_performance_feedback(path, _config())
    feedback = performance_feedback_for_row(_feedback_row(), report)
    assert feedback["multiplier"] == 1.0


def test_good_clv_prevents_loss_chasing_penalty(tmp_path):
    path = _history(tmp_path / "graded.csv", n=40, profit=-1.0, clv=0.5)
    report = build_performance_feedback(path, _config())
    feedback = performance_feedback_for_row(_feedback_row(), report)
    assert feedback["multiplier"] == 1.0


def test_adverse_clv_and_roi_reduce_next_portfolio_stake(tmp_path):
    live = _history(tmp_path / "graded.csv", n=40, profit=-1.0, clv=-0.5)
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    policy["deployment_mode"] = "paper"
    policy["portfolio"].update(
        {
            "drawdown_soft_stop_units": 900.0,
            "drawdown_hard_stop_units": 1000.0,
            "min_trailing_bets_for_throttle": 999,
            "max_slate_units": 10.0,
            "max_game_units": 10.0,
            "max_market_units": 10.0,
            "max_book_units": 10.0,
            "max_team_units": 10.0,
            "max_kickoff_window_units": 10.0,
            **_config(),
        }
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    candidate = pd.DataFrame(
        [
            {
                "game_id": "next",
                "date": "2026-10-04T17:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "quant_signal": "BET",
                "quant_market": "spread",
                "quant_side": "Home",
                "quant_book": "Book A",
                "quant_price": -3.5,
                "quant_odds": -110,
                "market_book_count": 2,
                "quant_probability": 0.58,
                "quant_ev": 0.08,
                "quant_edge": 0.06,
                "risk_multiplier": 1.0,
                "stake_units": 1.0,
            }
        ]
    )
    out, summary = apply_portfolio_controls(
        candidate,
        policy_path=policy_path,
        release_gate_path=tmp_path / "missing_gate.json",
        live_bets_path=live,
    )
    row = out.iloc[0]
    assert row["performance_multiplier"] == 0.5
    assert row["performance_adjusted_units"] == 0.5
    assert row["portfolio_candidate_units"] == 0.5
    assert summary["performance_adjusted_proposed_units"] == 0.5
