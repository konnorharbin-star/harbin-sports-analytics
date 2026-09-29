from __future__ import annotations

import json
from pathlib import Path


def _clip(x,lo=0.0,hi=100.0): return max(lo,min(hi,float(x)))


def build_health_report(meta: dict, pred=None, reports_dir="reports") -> dict:
    """System-readiness report, not a claim of predictive or betting superiority."""
    c=meta.get("market_coverage") or {}; games=max(1,int(c.get("games",0) or 0)); market_score=100*min(c.get("moneyline",0),c.get("spread",0),c.get("total",0))/games
    adv=meta.get("advanced_features") or {}; adv_cov=float(adv.get("live_coverage",adv.get("coverage",0)) or 0); advanced_score=100*adv_cov
    metrics=meta.get("metrics") or {}; base=metrics.get("margin_baseline_mae"); model=metrics.get("margin_mae"); improvement=max(0,(float(base)-float(model))/float(base)) if base and model is not None else 0
    folds=metrics.get("margin_walkforward_folds") or []; fold_improved=sum(1 for f in folds if f.get("mae",999)<=f.get("baseline_mae",-999))/len(folds) if folds else 0; validation_score=_clip(55+300*improvement+25*fold_improved)
    brier=metrics.get("win_brier"); ece=metrics.get("win_ece")
    if brier is None: calibration_score=45
    else:
        bc=_clip((.28-float(brier))/.10*100); ec=_clip((.15-float(ece or .15))/.15*100); calibration_score=.7*bc+.3*ec
    context=meta.get("current_context") or {}; context_score=_clip(55+45*float(bool(context.get("sources"))))
    intel=meta.get("market_intelligence") or {}; multi=float(intel.get("multi_book_coverage",0) or 0); intelligence_score=_clip(65+35*multi) if market_score>0 else 0
    reports=Path(reports_dir); bt=reports/"backtest_summary.json"; proof_score=20.; proof_status="UNPROVEN"
    if bt.exists():
        try:
            p=json.loads(bt.read_text()); bets=int(p.get("overall",{}).get("bets",0) or 0); clv=int(p.get("overall",{}).get("clv_samples",0) or 0); ci=p.get("overall",{}).get("roi_ci_95") or [None,None]
            proof_score=_clip(min(70,70*bets/1000)+min(20,20*clv/500)+(10 if ci[0] is not None else 0)); proof_status="ESTABLISHED SAMPLE" if bets>=500 and clv>=250 else "DEVELOPING SAMPLE" if bets>=100 else "UNPROVEN"
        except Exception: pass
    scores={"pipeline_and_data":100.,"market_coverage":round(market_score,1),"advanced_features":round(advanced_score,1),"model_validation":round(validation_score,1),"probability_calibration":round(calibration_score,1),"market_intelligence":round(intelligence_score,1),"current_context":round(context_score,1),"historical_betting_proof":round(proof_score,1)}
    weighted=.10*scores["pipeline_and_data"]+.12*scores["market_coverage"]+.16*scores["advanced_features"]+.18*scores["model_validation"]+.14*scores["probability_calibration"]+.10*scores["market_intelligence"]+.05*scores["current_context"]+.15*scores["historical_betting_proof"]
    blockers=[]
    if advanced_score<75: blockers.append("advanced feature coverage below 75%")
    if market_score<90: blockers.append("complete live market coverage below 90%")
    if not folds: blockers.append("season walk-forward validation unavailable")
    if brier is None: blockers.append("out-of-sample probability calibration metrics unavailable")
    if proof_status=="UNPROVEN": blockers.append("historical betting sample not yet large enough to establish market edge")
    return {"system_health_score":round(weighted,1),"meaning":"engineering/model-readiness score only; not a profitability or win-rate guarantee","components":scores,"betting_proof_status":proof_status,"blockers":blockers}
