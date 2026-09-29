from __future__ import annotations

import json
from pathlib import Path


def _clip(x,lo=0.0,hi=100.0): return max(lo,min(hi,float(x)))

def _read_json(path):
    try:return json.loads(Path(path).read_text())
    except Exception:return {}


def build_health_report(meta: dict, pred=None, reports_dir="reports") -> dict:
    c=meta.get("market_coverage") or {}; games=max(1,int(c.get("games",0) or 0)); market_score=100*min(c.get("moneyline",0),c.get("spread",0),c.get("total",0))/games
    adv=meta.get("advanced_features") or {}; overall_cov=float(adv.get("live_coverage",adv.get("coverage",0)) or 0); dyn_cov=adv.get("dynamic_coverage"); dyn_cov=overall_cov if dyn_cov is None else float(dyn_cov or 0); advanced_score=100*(.45*overall_cov+.55*dyn_cov)
    metrics=meta.get("metrics") or {}; base=metrics.get("margin_baseline_mae"); model=metrics.get("margin_mae"); improvement=max(0,(float(base)-float(model))/float(base)) if base and model is not None else 0
    folds=metrics.get("margin_walkforward_folds") or []; fold_improved=sum(1 for f in folds if f.get("mae",999)<=f.get("baseline_mae",-999))/len(folds) if folds else 0; validation_score=_clip(50+300*improvement+30*fold_improved)
    brier=metrics.get("win_brier"); ece=metrics.get("win_ece"); calibration_score=40 if brier is None else .65*_clip((.28-float(brier))/.10*100)+.35*_clip((.12-float(ece or .12))/.12*100)
    context=meta.get("current_context") or {}; context_score=_clip(45+55*float(bool(context.get("sources"))))
    intel=meta.get("market_intelligence") or {}; mb=float(intel.get("multi_book_coverage",0) or 0); intelligence_score=_clip(60+40*mb) if market_score>0 else 0
    reports=Path(reports_dir); evidence=_read_json(reports/"evidence_report.json"); pv=_read_json(reports/"policy_validation.json"); policy=_read_json(reports/"production_policy.json")
    proof_status=evidence.get("status","UNPROVEN"); o=evidence.get("overall") or _read_json(reports/"backtest_summary.json").get("overall",{}); n=int(o.get("bets",0) or 0); ci=o.get("roi_ci_95") or [None,None]; clv=o.get("avg_clv")
    enabled=[m for m,v in (policy.get("markets") or {}).items() if v.get("enabled")]; holdout_ready=pv.get("status")=="FORWARD_PAPER_READY"
    proof_score=20+min(20,20*n/1000)+(15 if clv is not None and float(clv)>0 else 0)+(20 if ci[0] is not None and float(ci[0])>0 else 0)+(15 if enabled else 0)+(10 if holdout_ready else 0); proof_score=_clip(proof_score)
    monitor_score=70.; mon=_read_json("outputs/live_monitoring.json")
    if mon: monitor_score=float(mon.get("live_readiness_score",70))
    scores={"pipeline_and_data":100.,"market_coverage":round(market_score,1),"advanced_features":round(advanced_score,1),"model_validation":round(validation_score,1),"probability_calibration":round(calibration_score,1),"market_intelligence":round(intelligence_score,1),"current_context":round(context_score,1),"live_monitoring":round(monitor_score,1),"historical_betting_proof":round(proof_score,1)}
    weighted=.08*scores["pipeline_and_data"]+.10*scores["market_coverage"]+.14*scores["advanced_features"]+.16*scores["model_validation"]+.12*scores["probability_calibration"]+.09*scores["market_intelligence"]+.05*scores["current_context"]+.10*scores["live_monitoring"]+.16*scores["historical_betting_proof"]
    blockers=[]
    if overall_cov<.8: blockers.append("advanced feature overall coverage below 80%")
    if dyn_cov<.8: blockers.append("dynamic pregame advanced-feature coverage below 80%")
    if market_score<90: blockers.append("complete verified market coverage below 90%")
    if not folds: blockers.append("season walk-forward diagnostic folds unavailable")
    if brier is None: blockers.append("out-of-sample probability calibration metrics unavailable")
    if not enabled: blockers.append("no market has passed strict evidence gating")
    if not holdout_ready: blockers.append("production policy has not passed final holdout validation")
    if proof_status not in {"VALIDATED","ROBUST"}: blockers.append("historical market edge is not yet validated")
    if mb<.5: blockers.append("independent multi-book live consensus coverage below 50%")
    if monitor_score<90: blockers.append("live operational monitoring score below 90")
    return {"system_health_score":round(weighted,1),"meaning":"engineering/model-readiness score only; not a profitability or win-rate guarantee","components":scores,"betting_proof_status":proof_status,"validated_markets":enabled,"policy_validation":pv.get("status","UNKNOWN"),"advanced_dynamic_coverage":round(dyn_cov,4),"blockers":blockers}
