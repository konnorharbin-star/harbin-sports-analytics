from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import math
import numpy as np
import pandas as pd


def _clip(x, lo=0.0, hi=100.0): return max(lo, min(hi, float(x)))

def _safe_mean(s):
    x = pd.to_numeric(s, errors="coerce").dropna(); return float(x.mean()) if len(x) else None

def _drift_score(live: pd.Series, hist: pd.Series):
    a = pd.to_numeric(live, errors="coerce").dropna(); b = pd.to_numeric(hist, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 50: return None
    sd = float(b.std(ddof=1)); z = abs(float(a.mean()) - float(b.mean())) / max(sd, 1e-6)
    ratio = float(a.std(ddof=1)) / max(sd, 1e-6) if len(a) > 1 else 1.0
    penalty = min(70, 25*z + 25*abs(math.log(max(.1, ratio))))
    return _clip(100-penalty)


def build_live_monitoring(pred: pd.DataFrame, meta: dict, reports_dir="reports") -> dict:
    alerts=[]; scores={}
    c=meta.get("market_coverage") or {}; games=max(1,int(c.get("games",0) or 0)); market=min(c.get("moneyline",0),c.get("spread",0),c.get("total",0))/games; scores["market_coverage"]=_clip(100*market)
    adv=meta.get("advanced_features") or {}; scores["advanced_coverage"]=_clip(100*float(adv.get("live_coverage",adv.get("coverage",0)) or 0))
    intel=meta.get("market_intelligence") or {}; scores["multi_book"]=_clip(100*float(intel.get("multi_book_coverage",0) or 0))
    ctx=meta.get("current_context") or {}; scores["context"]=_clip(70+30*float(bool(ctx.get("sources"))))
    metrics=meta.get("metrics") or {}; brier=metrics.get("win_brier"); ece=metrics.get("win_ece")
    scores["calibration"]=_clip(100 - 220*max(0,float(brier or .25)-.20) - 250*float(ece or .10))
    try:
        gen=datetime.fromisoformat(str(meta.get("generated_at")).replace("Z","+00:00")); age=(datetime.now(timezone.utc)-gen.astimezone(timezone.utc)).total_seconds()/3600
    except Exception: age=999
    scores["freshness"]=_clip(100-6*max(0,age-1));
    if age>8: alerts.append(f"model output is {age:.1f} hours old")
    reports=Path(reports_dir); ref=reports/"backtest_predictions.csv"; drift=[]
    if ref.exists() and not pred.empty:
        try:
            h=pd.read_csv(ref,low_memory=False)
            for lc,hc,name in [("model_margin_home","pred_margin_home","margin"),("model_total","pred_total","total"),("calibrated_home_probability","home_win_probability","probability")]:
                if lc in pred.columns and hc in h.columns:
                    s=_drift_score(pred[lc],h[hc]);
                    if s is not None: drift.append(s); scores[f"drift_{name}"]=round(s,1)
        except Exception as exc: alerts.append(f"drift reference unavailable: {type(exc).__name__}")
    scores["distribution_stability"] = round(float(np.mean(drift)),1) if drift else 70.0
    if scores["distribution_stability"] < 60: alerts.append("live prediction distribution differs materially from walk-forward history")
    missing = float(pred.isna().mean().mean()) if not pred.empty else 1.0; scores["output_completeness"]=_clip(100*(1-min(.5,missing)/.5))
    weights={"market_coverage":.14,"advanced_coverage":.16,"multi_book":.10,"context":.08,"calibration":.18,"freshness":.10,"distribution_stability":.14,"output_completeness":.10}; total=sum(weights[k]*scores.get(k,0) for k in weights)
    if scores["market_coverage"]<90: alerts.append("verified ML/spread/total coverage below 90%")
    if scores["advanced_coverage"]<80: alerts.append("advanced feature coverage below 80%")
    return {"live_readiness_score":round(total,1),"scores":scores,"alerts":alerts,"meaning":"operational/model-monitoring score; not a profitability guarantee"}


def write_live_monitoring(pred, meta, output="outputs/live_monitoring.json", reports_dir="reports"):
    r=build_live_monitoring(pred,meta,reports_dir); p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2)); return r
