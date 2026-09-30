from __future__ import annotations

import json
from pathlib import Path


def _clip(x,lo=0.0,hi=100.0): return max(lo,min(hi,float(x)))


def build_health_report(meta: dict, pred=None, reports_dir="reports") -> dict:
    c=meta.get("market_coverage") or {}; games=max(1,int(c.get("games",0) or 0)); market_score=100*min(c.get("moneyline",0),c.get("spread",0),c.get("total",0))/games
    adv=meta.get("advanced_features") or {}
    # Dynamic efficiency is the material live input. Static talent/returning-production
    # coverage alone must not earn a perfect advanced-feature score.
    dynamic_cov=float(adv.get("dynamic_coverage",0) or 0)
    broad_cov=float(adv.get("live_coverage",adv.get("coverage",0)) or 0)
    advanced_score=100*(.80*dynamic_cov+.20*broad_cov)
    metrics=meta.get("metrics") or {}; base=metrics.get("margin_baseline_mae"); model=metrics.get("margin_mae"); improvement=max(0,(float(base)-float(model))/float(base)) if base and model is not None else 0
    folds=metrics.get("margin_walkforward_folds") or []; fold_improved=sum(1 for f in folds if f.get("mae",999)<=f.get("baseline_mae",-999))/len(folds) if folds else 0; validation_score=_clip(55+300*improvement+25*fold_improved)
    brier=metrics.get("win_brier"); ece=metrics.get("win_ece"); calibration_score=45 if brier is None else .7*_clip((.28-float(brier))/.10*100)+.3*_clip((.15-float(ece or .15))/.15*100)
    context=meta.get("current_context") or {}; context_score=_clip(55+45*float(bool(context.get("sources"))))
    intel=meta.get("market_intelligence") or {}; intelligence_score=_clip(65+35*float(intel.get("multi_book_coverage",0) or 0)) if market_score>0 else 0
    reports=Path(reports_dir); proof_score=20.; proof_status="UNPROVEN"; evidence=reports/"evidence_report.json"
    if evidence.exists():
        try:
            e=json.loads(evidence.read_text()); proof_status=e.get("status","UNPROVEN"); o=e.get("overall") or {}; n=int(o.get("bets",0) or 0); ci=o.get("roi_ci_95") or [None,None]; clv=o.get("avg_clv"); proof_score=min(100,25+min(35,35*n/1000)+(20 if clv is not None and float(clv)>0 else 0)+(20 if ci[0] is not None and float(ci[0])>0 else 0))
        except Exception: pass
    elif (reports/"backtest_summary.json").exists():
        try:
            p=json.loads((reports/"backtest_summary.json").read_text()); n=int(p.get("overall",{}).get("bets",0) or 0); proof_score=_clip(min(75,75*n/1000)); proof_status="DEVELOPING" if n>=150 else "UNPROVEN"
        except Exception: pass
    monitor_score=70.
    mon=Path("outputs/live_monitoring.json")
    if mon.exists():
        try: monitor_score=float(json.loads(mon.read_text()).get("live_readiness_score",70))
        except Exception: pass
    scores={"pipeline_and_data":100.,"market_coverage":round(market_score,1),"advanced_features":round(advanced_score,1),"model_validation":round(validation_score,1),"probability_calibration":round(calibration_score,1),"market_intelligence":round(intelligence_score,1),"current_context":round(context_score,1),"live_monitoring":round(monitor_score,1),"historical_betting_proof":round(proof_score,1)}
    weighted=.08*scores["pipeline_and_data"]+.10*scores["market_coverage"]+.14*scores["advanced_features"]+.16*scores["model_validation"]+.12*scores["probability_calibration"]+.09*scores["market_intelligence"]+.05*scores["current_context"]+.10*scores["live_monitoring"]+.16*scores["historical_betting_proof"]
    blockers=[]
    if dynamic_cov<.80: blockers.append("dynamic advanced-efficiency coverage below 80%")
    if market_score<90: blockers.append("complete verified market coverage below 90%")
    if not folds: blockers.append("season walk-forward diagnostic folds unavailable")
    if brier is None: blockers.append("out-of-sample probability calibration metrics unavailable")
    if proof_status not in {"VALIDATED","ROBUST"}: blockers.append("historical market edge is not yet validated")
    if monitor_score<75: blockers.append("live operational monitoring score below 75")
    return {"system_health_score":round(weighted,1),"meaning":"engineering/model-readiness score only; not a profitability or win-rate guarantee","components":scores,"betting_proof_status":proof_status,"blockers":blockers}
