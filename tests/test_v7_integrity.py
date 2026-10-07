import json
from types import SimpleNamespace
import pandas as pd

from harbin.advanced import identify_team_columns
from harbin.market_intel import MarketIntelligence, _apply_capture_fallback
from harbin.policy import signal_from_policy, market_allowed
from harbin.portfolio import apply_portfolio_controls
from harbin.pro_market import quant_signal


def test_advanced_identity_prefers_numeric_team_id_over_text_pos_team():
    df = pd.DataFrame({
        "team_id": [333, 145, 333, 145],
        "pos_team": ["Alabama", "Georgia", "Alabama", "Georgia"],
        "season": [2025] * 4,
        "week": [1, 1, 2, 2],
    })
    ic, nc = identify_team_columns(df)
    assert ic == "team_id"
    assert nc == "pos_team"


def test_advanced_identity_uses_numeric_pos_team_only_as_id():
    df = pd.DataFrame({"pos_team": [333, 145, 333, 145], "season": [2025] * 4, "week": [1, 1, 2, 2]})
    ic, nc = identify_team_columns(df)
    assert ic == "pos_team"
    assert nc is None


def test_unvalidated_market_fails_closed(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({
        "markets": {"total": {"enabled": False, "disabled_reason": "holdout failed", "excluded_weeks": []}}
    }))
    allowed, reason = market_allowed("total", path=str(p))
    assert not allowed and "holdout" in reason
    assert signal_from_policy(.20, 10, .70, "total", path=str(p)) == "PASS"


def test_excluded_week_fails_closed(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"markets": {"spread": {"enabled": True, "excluded_weeks": [1]}}}))
    assert signal_from_policy(.20, 10, .70, "spread", path=str(p), week=1) == "PASS"
    assert signal_from_policy(.20, 10, .70, "spread", path=str(p), week=2) == "STRONG"


def test_research_defaults_remain_reproducible():
    assert quant_signal(.20, 10, .70, "total", policy_path=None) == "STRONG"


def test_market_intel_selects_executable_best_lines_and_prices():
    g = SimpleNamespace(provider="BookA", home_ml=-150, away_ml=130, home_spread=-3.5, market_total=56)
    quotes = [
        {"provider":"BookA","home_ml":-150,"away_ml":130,"home_spread":-3.5,"market_total":56,
         "home_spread_price":-110,"away_spread_price":-110,"over_price":-108,"under_price":-112,
         "last_update":"2026-10-06T17:58:00Z"},
        {"provider":"BookB","home_ml":-145,"away_ml":135,"home_spread":-3.0,"market_total":55.5,
         "home_spread_price":-105,"away_spread_price":-115,"over_price":-110,"under_price":-110,
         "last_update":"2026-10-06T17:59:00Z"},
    ]
    s = MarketIntelligence._summary(g, quotes)
    assert s["best_home_spread"] == -3.0
    assert s["best_away_spread"] == 3.5
    assert s["best_over_total"] == 55.5
    assert s["best_under_total"] == 56.0
    assert s["best_home_spread_odds"] == -105
    assert s["best_home_ml"] == -145
    assert s["best_away_ml"] == 135
    assert s["best_home_ml_quote_at"] == "2026-10-06T17:59:00Z"
    assert s["best_away_ml_quote_at"] == "2026-10-06T17:59:00Z"
    assert s["best_home_spread_quote_at"] == "2026-10-06T17:59:00Z"
    assert s["best_away_spread_quote_at"] == "2026-10-06T17:58:00Z"
    assert s["best_over_quote_at"] == "2026-10-06T17:59:00Z"
    assert s["best_under_quote_at"] == "2026-10-06T17:58:00Z"


def test_portfolio_caps_same_kickoff_cluster(tmp_path):
    policy = tmp_path / "policy.json"
    gate = tmp_path / "gate.json"
    policy.write_text(json.dumps({
        "deployment_mode":"paper",
        "portfolio":{"max_slate_units":5,"max_game_units":1.5,"max_team_units":2,"max_market_units":5,"max_kickoff_window_units":2,"kickoff_window_hours":3}
    }))
    gate.write_text(json.dumps({"release_state":"PAPER","production_eligible":False,"blockers":[]}))
    df = pd.DataFrame([
        {"date":"2026-10-03T16:00:00Z","home_team":"A","away_team":"B","quant_signal":"BET","quant_market":"spread","quant_ev":.10,"risk_multiplier":1,"stake_units":1.5},
        {"date":"2026-10-03T17:00:00Z","home_team":"C","away_team":"D","quant_signal":"BET","quant_market":"spread","quant_ev":.09,"risk_multiplier":1,"stake_units":1.5},
    ])
    _, summary = apply_portfolio_controls(df, str(policy), str(gate))
    assert summary["paper_allocated_units"] == 2.0


def test_market_intel_uses_capture_time_when_source_timestamp_is_missing():
    g = SimpleNamespace(
        provider="BookA",
        home_ml=-150,
        away_ml=130,
        home_spread=-3.5,
        market_total=56,
    )
    quotes = [
        {
            "provider": "BookB",
            "source": "action_network",
            "home_ml": -145,
            "away_ml": 135,
            "home_spread": -3.0,
            "market_total": 55.5,
            "home_spread_price": -105,
            "away_spread_price": -115,
            "over_price": -110,
            "under_price": -110,
            "last_update": None,
        }
    ]

    summary = MarketIntelligence._summary(g, quotes)
    captured_at = "2026-10-07T02:30:00+00:00"
    s = _apply_capture_fallback(summary, captured_at)

    assert s["best_home_ml"] == -145
    assert s["best_home_ml_book"] == "BookB"
    assert s["best_home_ml_quote_at"] == captured_at
    assert s["best_home_ml_quote_time_source"] == "captured_at"
    assert s["best_home_spread_quote_at"] == captured_at
    assert s["best_over_quote_at"] == captured_at
    serialized = json.loads(s["market_quotes_json"])
    book_b = next(row for row in serialized if row["provider"] == "BookB")
    assert book_b["captured_at"] == captured_at
    assert book_b["last_update"] is None

