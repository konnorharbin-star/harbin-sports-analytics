import json
import pandas as pd
from harbin.models import release_weight_guard
from harbin.contracts import validate_run
from harbin.release_gate import build_release_gate


def test_release_weight_falls_back_when_holdout_worse():
    folds=[{"mae":12.0,"baseline_mae":13.0},{"mae":12.5,"baseline_mae":13.0}]
    w,passed,reason=release_weight_guard(0.8,12.4,12.6,folds)
    assert w==0.0 and not passed
    assert "holdout" in reason


def test_release_weight_requires_prior_fold_support():
    folds=[{"mae":14.0,"baseline_mae":13.0},{"mae":12.0,"baseline_mae":13.0},{"mae":14.0,"baseline_mae":13.0}]
    w,passed,_=release_weight_guard(0.8,13.0,12.5,folds,min_fold_win_rate=.5)
    assert w==0.0 and not passed


def _good_pred():
    return pd.DataFrame([{
        "game_id":"1","season":2026,"week":5,"date":"2026-10-01T00:00:00Z","away_team":"A","home_team":"H",
        "model_margin_home":3.0,"model_total":52.0,"winner":"H","calibrated_home_probability":.58,
        "away_ml":130,"home_ml":-150,"market_spread_home":-3.0,"market_total":51.5,"away_score_exact":24.5,"home_score_exact":27.5,
    }])

def test_data_contract_passes_sane_row():
    r=validate_run(_good_pred(),{"metrics":{"margin_release_weight":.5,"total_release_weight":0.}})
    assert r["status"]=="PASS"


def test_data_contract_catches_duplicate_game():
    p=pd.concat([_good_pred(),_good_pred()],ignore_index=True)
    r=validate_run(p,{})
    assert r["status"]=="FAIL"
    assert any(x["code"]=="identity.duplicate_game_id" for x in r["issues"])


def _engineering_ready_meta(multi_book):
    return {
        "market_coverage":{"games":10,"moneyline":10,"spread":10,"total":10},
        "advanced_features":{"live_coverage":1.0,"dynamic_coverage":1.0},
        "current_context":{"coverage":1.0,"sources":["x"]},
        "market_intelligence":{"multi_book_coverage":multi_book},
        "metrics":{"margin_release_guard_passed":True,"total_release_guard_passed":True,"total_release_weight":.4,"win_brier":.15,"win_ece":.03},
    }


def test_release_gate_cannot_go_production_without_evidence(tmp_path):
    meta=_engineering_ready_meta(1.0)
    gate=build_release_gate(meta,{"live_readiness_score":95},{"status":"PASS"},tmp_path/"missing.json",tmp_path/"missing-live.json")
    assert gate["release_state"]=="PAPER"
    assert not gate["production_eligible"]


def test_release_gate_production_requires_history_and_live(tmp_path):
    evidence={"overall":{"bets":1500,"roi_ci_95":[.01,.08],"avg_clv":.02},"by_market":{"spread":{"roi":.03,"avg_clv":.01},"moneyline":{"roi":.02,"avg_clv":.01}},"by_season":{"2024":{"roi":.02,"avg_clv":.01},"2025":{"roi":.03,"avg_clv":.01}}}
    live={"graded_bets":350,"roi":.02,"avg_clv_proxy":.01}
    ep=tmp_path/"e.json"; lp=tmp_path/"l.json"; ep.write_text(json.dumps(evidence)); lp.write_text(json.dumps(live))
    meta=_engineering_ready_meta(.9)
    gate=build_release_gate(meta,{"live_readiness_score":95},{"status":"PASS"},ep,lp)
    assert gate["release_state"]=="PRODUCTION"
    assert gate["production_eligible"]
