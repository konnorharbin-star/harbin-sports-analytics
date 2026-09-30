import json

import pandas as pd

from harbin.policy import DEFAULT_POLICY, load_policy
from harbin.portfolio import apply_portfolio_controls, build_bankroll_risk_state


def _row(n, *, book="Book A", stake=1.0, ev=.10, market="spread", home=None, away=None, side=None):
    home = home or f"H{n}"
    away = away or f"A{n}"
    side = side or home
    price = -3.5 if market == "spread" else 50.5 if market == "total" else -110
    return {
        "game_id": str(n),
        "date": f"2026-10-0{1 + (n % 3)}T12:00:00Z",
        "home_team": home,
        "away_team": away,
        "quant_signal": "STRONG",
        "quant_market": market,
        "quant_side": side,
        "quant_book": book,
        "quant_price": price,
        "quant_odds": -110,
        "market_book_count": 2,
        "quant_ev": ev,
        "risk_multiplier": 1.0,
        "stake_units": stake,
    }


def _write_policy(path, **portfolio):
    d = json.loads(json.dumps(DEFAULT_POLICY))
    d["deployment_mode"] = "production"
    d["portfolio"].update(portfolio)
    path.write_text(json.dumps(d))
    return path


def _write_gate(path, eligible=True):
    path.write_text(json.dumps({
        "release_state": "PRODUCTION" if eligible else "PAPER",
        "production_eligible": eligible,
        "blockers": [] if eligible else ["test blocker"],
    }))
    return path


def _write_live(path, profits=None, clv=.01):
    profits = profits if profits is not None else [1, -1] * 20
    pd.DataFrame({
        "profit": profits,
        "execution_clv": [clv] * len(profits),
        "kickoff": pd.date_range("2026-01-01", periods=len(profits), freq="D"),
    }).to_csv(path, index=False)
    return path


def test_legacy_policy_deep_merges_stage5_portfolio_defaults(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"portfolio": {"max_slate_units": 2.0}}))
    d = load_policy(p)
    assert d["portfolio"]["max_slate_units"] == 2.0
    assert d["portfolio"]["max_book_units"] == DEFAULT_POLICY["portfolio"]["max_book_units"]
    assert d["portfolio"]["drawdown_hard_stop_units"] == DEFAULT_POLICY["portfolio"]["drawdown_hard_stop_units"]


def test_missing_executable_book_cannot_be_approved(tmp_path):
    policy = _write_policy(tmp_path / "policy.json")
    gate = _write_gate(tmp_path / "gate.json")
    live = _write_live(tmp_path / "live.csv")
    out, summary = apply_portfolio_controls(pd.DataFrame([_row(1, book="")]), policy, gate, live)
    assert summary["approved_units"] == 0
    assert summary["execution_blocked_bets"] == 1
    assert "sportsbook provenance" in out.iloc[0]["portfolio_limit_reason"]


def test_single_book_concentration_cap_is_enforced(tmp_path):
    policy = _write_policy(tmp_path / "policy.json", max_book_units=1.0, max_kickoff_window_units=5.0)
    gate = _write_gate(tmp_path / "gate.json")
    live = _write_live(tmp_path / "live.csv")
    df = pd.DataFrame([
        _row(1, stake=.75, ev=.12),
        _row(2, stake=.75, ev=.11),
        _row(3, stake=.75, ev=.10),
    ])
    out, summary = apply_portfolio_controls(df, policy, gate, live)
    assert summary["allocated_by_book"]["Book A"] <= 1.0001
    assert out["portfolio_candidate_units"].sum() <= 1.0001


def test_current_drawdown_hard_stop_blocks_production(tmp_path):
    policy = _write_policy(
        tmp_path / "policy.json",
        drawdown_soft_stop_units=2.0,
        drawdown_hard_stop_units=5.0,
    )
    gate = _write_gate(tmp_path / "gate.json")
    live = _write_live(tmp_path / "live.csv", profits=[1] * 6 + [-1] * 8, clv=-.01)
    state = build_bankroll_risk_state(live, load_policy(policy)["portfolio"])
    assert state["hard_stop"] is True
    out, summary = apply_portfolio_controls(pd.DataFrame([_row(1)]), policy, gate, live)
    assert summary["mode"] == "halted"
    assert summary["approved_units"] == 0
    assert out.iloc[0]["portfolio_action"] == "PASS"


def test_soft_drawdown_reduces_unit_budget(tmp_path):
    policy = _write_policy(
        tmp_path / "policy.json",
        drawdown_soft_stop_units=2.0,
        drawdown_hard_stop_units=10.0,
        drawdown_floor_multiplier=.25,
        max_slate_units=5.0,
    )
    gate = _write_gate(tmp_path / "gate.json")
    live = _write_live(tmp_path / "live.csv", profits=[1] * 5 + [-1] * 4, clv=.01)
    state = build_bankroll_risk_state(live, load_policy(policy)["portfolio"])
    assert 0 < state["risk_multiplier"] < 1
    _, summary = apply_portfolio_controls(pd.DataFrame([_row(1), _row(2)]), policy, gate, live)
    assert summary["effective_unit_caps"]["max_slate_units"] < 5.0


def test_directional_team_exposure_does_not_charge_opponent(tmp_path):
    policy = _write_policy(
        tmp_path / "policy.json",
        max_team_units=.75,
        max_slate_units=5.0,
        max_market_units=5.0,
        max_book_units=5.0,
        max_kickoff_window_units=5.0,
    )
    live = _write_live(tmp_path / "live.csv")
    df = pd.DataFrame([
        _row(1, stake=.75, home="A", away="B", side="A"),
        _row(2, stake=.75, home="B", away="C", side="B"),
    ])
    out, _ = apply_portfolio_controls(df, policy, tmp_path / "missing_gate.json", live)
    assert abs(float(out["portfolio_candidate_units"].sum()) - 1.5) < 1e-9


def test_max_bet_count_prioritizes_higher_ranked_candidates(tmp_path):
    policy = _write_policy(
        tmp_path / "policy.json",
        max_bets=1,
        max_slate_units=5.0,
        max_book_units=5.0,
        max_kickoff_window_units=5.0,
    )
    live = _write_live(tmp_path / "live.csv")
    df = pd.DataFrame([_row(1, ev=.08), _row(2, ev=.12)])
    out, summary = apply_portfolio_controls(df, policy, tmp_path / "missing_gate.json", live)
    chosen = out[out["portfolio_candidate_units"] > 0]
    assert len(chosen) == 1
    assert float(chosen.iloc[0]["quant_ev"]) == .12
    assert summary["bets"] == 1


def test_production_gate_cannot_use_stale_gate_without_live_ledger(tmp_path):
    policy = _write_policy(tmp_path / "policy.json")
    gate = _write_gate(tmp_path / "gate.json")
    out, summary = apply_portfolio_controls(
        pd.DataFrame([_row(1)]),
        policy,
        gate,
        tmp_path / "missing_live.csv",
    )
    assert summary["mode"] == "halted"
    assert summary["production_eligible"] is False
    assert summary["approved_units"] == 0
    assert "ledger is unavailable" in summary["production_block_reason"]
