import json
from pathlib import Path

import numpy as np
import pandas as pd

from harbin.grading_audit import grade_prediction_history
from harbin.policy import DEFAULT_POLICY, derive_production_policy
from harbin.portfolio_audit import apply_portfolio_controls
from harbin.pro_market import select_best_market
from harbin.proof import build_evidence_report
from harbin.release_gate import build_release_gate


def _policy_bets(eval_profit=.5):
    rows=[]
    for season,profit in ((2023,.5),(2024,.5),(2025,eval_profit)):
        for i in range(70):
            rows.append({
                "game_id":f"{season}-{i}","season":season,"week":1+i%10,"market":"spread",
                "ev":.20,"edge":10.0,"probability":.70,"profit":profit,"clv":.02,
                "entry_quote_verified":True,
            })
    return pd.DataFrame(rows)


def _robust_evidence(path):
    data={"status":"ROBUST","promotion_sample":{"entry_quote_verified":True}}
    Path(path).write_text(json.dumps(data)); return path


def test_policy_selection_does_not_use_untouched_evaluation(tmp_path):
    bets=_policy_bets(.5); bp=tmp_path/"bets.csv"; ep=_robust_evidence(tmp_path/"evidence.json")
    bets.to_csv(bp,index=False)
    p1=derive_production_policy(bp,tmp_path/"missing-summary.json",tmp_path/"p1.json",ep)
    selected1=p1["diagnostics"]["spread"]["selected"][:3]
    enabled1=p1["markets"]["spread"]["enabled"]

    bets.loc[bets.season==2025,"profit"]=-1.0; bets.to_csv(bp,index=False)
    p2=derive_production_policy(bp,tmp_path/"missing-summary.json",tmp_path/"p2.json",ep)
    selected2=p2["diagnostics"]["spread"]["selected"][:3]
    assert selected1==selected2
    assert p2["diagnostics"]["selection_uses_evaluation"] is False
    assert enabled1 is True and p2["markets"]["spread"]["enabled"] is False


def test_promotion_evidence_excludes_unverified_archive_entries(tmp_path):
    rows=[]
    for i in range(1000):
        rows.append({
            "game_id":str(i),"season":2024 if i<500 else 2025,"week":1+i%12,
            "market":"spread" if i%2==0 else "moneyline","signal":"STRONG","result":1,
            "profit":.10,"clv":.02,"entry_quote_verified":True,
        })
    for i in range(500):
        rows.append({
            "game_id":f"u{i}","season":2025,"week":1+i%12,"market":"total","signal":"STRONG","result":1,
            "profit":5.0,"clv":.50,"entry_quote_verified":False,
        })
    bp=tmp_path/"bets.csv"; pd.DataFrame(rows).to_csv(bp,index=False)
    report=build_evidence_report(tmp_path/"missing-summary.json",bp,tmp_path/"evidence.json")
    assert report["promotion_sample"]["verified_bets"]==1000
    assert report["promotion_sample"]["all_archive_bets"]==1500
    assert report["promotion_sample"]["excluded_unverified_bets"]==500
    assert "total" not in report["by_market"]
    assert report["promotion_sample"]["entry_quote_verified"] is True


def _engineering_ready_meta():
    return {
        "market_coverage":{"games":10,"moneyline":10,"spread":10,"total":10},
        "advanced_features":{"live_coverage":1.0,"dynamic_coverage":1.0},
        "current_context":{"coverage":1.0,"sources":["x"]},
        "market_intelligence":{"multi_book_coverage":1.0},
        "metrics":{"margin_release_guard_passed":True,"total_release_guard_passed":True,"total_release_weight":.2,"win_brier":.15,"win_ece":.03},
    }


def test_release_gate_rejects_nonportfolio_forward_evidence(tmp_path):
    evidence={
        "status":"ROBUST","overall":{"bets":1500,"roi_ci_95":[.01,.05],"avg_clv":.02},
        "promotion_sample":{"entry_quote_verified":True,"positive_markets":2,"positive_seasons":2},
    }
    live={"graded_bets":500,"roi":.03,"avg_clv_proxy":.01,"portfolio_verified":False}
    ep=tmp_path/"e.json"; lp=tmp_path/"l.json"; ep.write_text(json.dumps(evidence)); lp.write_text(json.dumps(live))
    gate=build_release_gate(_engineering_ready_meta(),{"live_readiness_score":95},{"status":"PASS"},ep,lp)
    assert gate["release_state"]=="SHADOW"
    assert gate["production_eligible"] is False
    assert any(x["name"]=="portfolio_verified_forward_ledger" and not x["passed"] for x in gate["checks"])


class _Client:
    def season_frame(self, season):
        return pd.DataFrame([
            {"game_id":"1","completed":True,"home_points":28,"away_points":20},
            {"game_id":"2","completed":True,"home_points":17,"away_points":21},
        ])


def test_forward_grading_requires_strict_timing_and_portfolio_decision(tmp_path):
    hist=tmp_path/"history"; reports=tmp_path/"reports"; hist.mkdir(); reports.mkdir()
    pd.DataFrame([
        {"decision_at":"2026-10-01T10:00:00Z","game_id":"1","season":2026,"week":5,"date":"2026-10-01T12:00:00Z","away_team":"A","home_team":"H","model_margin_home":4,"model_total":50,"quant_signal":"BET","quant_market":"spread","quant_side":"H","quant_book":"Book","quant_price":-3,"quant_odds":-110,"portfolio_candidate_units":.5,"portfolio_action":"PAPER","execution_ready":True},
        {"decision_at":"2026-10-01T13:00:00Z","game_id":"2","season":2026,"week":5,"date":"2026-10-01T12:00:00Z","away_team":"B","home_team":"J","model_margin_home":-2,"model_total":45,"quant_signal":"BET","quant_market":"moneyline","quant_side":"B","quant_book":"Book","quant_price":120,"quant_odds":120,"portfolio_candidate_units":.5,"portfolio_action":"PAPER","execution_ready":True},
    ]).to_csv(hist/"portfolio_decisions_v1.csv",index=False)
    report=grade_prediction_history(_Client(),hist,reports)
    assert report["portfolio_verified"] is True
    assert report["evidence_source"]=="portfolio_decisions_v1"
    assert report["timing_excluded_rows"]==1
    assert report["graded_bets"]==1


def test_missing_spread_price_cannot_create_quant_wager(tmp_path):
    row={
        "week":5,"home_team":"Home","away_team":"Away","provider":"Primary",
        "calibrated_home_probability":.60,"quant_best_ml_roi":np.nan,
        "cover_probability":.65,"spread_edge_pts":6.0,"spread_team":"Home","spread_line":-3.0,
        "best_home_spread_odds":np.nan,"best_home_spread_book":"Book",
        "total_probability":np.nan,"total_edge_pts":np.nan,
    }
    pick=select_best_market(row,policy_path=tmp_path/"missing-policy.json")
    assert pick["quant_signal"]=="PASS"
    assert np.isnan(pick["quant_odds"])


def test_production_portfolio_rejects_stale_quote_timestamp(tmp_path):
    policy=json.loads(json.dumps(DEFAULT_POLICY)); policy["deployment_mode"]="production"; pp=tmp_path/"policy.json"; pp.write_text(json.dumps(policy))
    gp=tmp_path/"gate.json"; gp.write_text(json.dumps({"release_state":"PRODUCTION","production_eligible":True,"blockers":[]}))
    live=tmp_path/"live.csv"; pd.DataFrame({"profit":[1,-1]*20,"execution_clv":[.01]*40,"kickoff":pd.date_range("2026-01-01",periods=40,freq="D")}).to_csv(live,index=False)
    row={
        "game_id":"1","date":"2026-10-03T18:00:00Z","home_team":"H","away_team":"A",
        "quant_signal":"STRONG","quant_market":"spread","quant_side":"H","quant_book":"Book",
        "quant_price":-3.0,"quant_odds":-110,"quant_quote_at":"2020-01-01T00:00:00Z","market_book_count":2,
        "quant_ev":.15,"risk_multiplier":1.0,"stake_units":.5,
    }
    out,summary=apply_portfolio_controls(pd.DataFrame([row]),pp,gp,live)
    assert summary["approved_units"]==0
    assert out.iloc[0]["execution_ready"]==False
    assert "stale" in out.iloc[0]["portfolio_limit_reason"]
