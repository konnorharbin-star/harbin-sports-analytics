import json
import pandas as pd
from harbin.policy import signal_from_policy, DEFAULT_POLICY
from harbin.portfolio import apply_portfolio_controls
from harbin.monitoring import build_live_monitoring


def test_policy_default_gates():
    # Missing policy is fail-closed in v7: no evidence, no bet signal.
    assert signal_from_policy(.08,5,.60,"spread",path="/tmp/definitely_missing_policy.json") == "PASS"
    assert signal_from_policy(.01,8,.70,"spread",path="/tmp/definitely_missing_policy.json") == "PASS"


def test_portfolio_caps_and_paper_mode(tmp_path):
    p=tmp_path/"policy.json"; d=json.loads(json.dumps(DEFAULT_POLICY)); d["deployment_mode"]="paper"; d["portfolio"]["max_slate_units"]=1.0; p.write_text(json.dumps(d))
    df=pd.DataFrame([{"home_team":"A","away_team":"B","quant_signal":"STRONG","quant_market":"spread","quant_ev":.1,"risk_multiplier":1,"stake_units":1.2},{"home_team":"C","away_team":"D","quant_signal":"BET","quant_market":"total","quant_ev":.08,"risk_multiplier":1,"stake_units":1.0}])
    out,s=apply_portfolio_controls(df,p)
    assert s["mode"]=="paper" and s["paper_allocated_units"]<=1.0001
    assert float(out.portfolio_stake_units.sum())==0
    assert set(out.portfolio_action)<= {"PAPER","PASS"}


def test_monitoring_returns_bounded_score():
    pred=pd.DataFrame({"market_available":[True]*10,"model_margin_home":[1]*10,"model_total":[55]*10,"calibrated_home_probability":[.55]*10})
    meta={"market_coverage":{"games":10,"moneyline":10,"spread":10,"total":10},"advanced_features":{"live_coverage":1},"market_intelligence":{"multi_book_coverage":1},"current_context":{"sources":["x"]},"metrics":{"win_brier":.22,"win_ece":.05},"generated_at":"2099-01-01T00:00:00+00:00"}
    r=build_live_monitoring(pred,meta,reports_dir="/tmp/no_reports")
    assert 0<=r["live_readiness_score"]<=100
