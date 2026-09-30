import json
import pandas as pd

from harbin.advanced import identify_team_columns
from harbin.policy import signal_from_policy, market_allowed
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
        "markets": {
            "total": {
                "enabled": False,
                "disabled_reason": "holdout failed",
                "excluded_weeks": [],
            }
        }
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
    # Historical backtests call policy_path=None and must not consume generated policy state.
    assert quant_signal(.20, 10, .70, "total", policy_path=None) == "STRONG"
