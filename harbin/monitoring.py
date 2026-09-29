from __future__ import annotations
import json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

def _clip(x,lo=0.,hi=100.):return max(lo,min(hi,float(x)))
def _drift(live,hist):
    a=pd.to_numeric(live,errors="coerce").dropna();b=pd.to_numeric(hist,errors="coerce").dropna()
    if len(a)<5 or len(b)<50:return None
    sd=max(float(b.std(ddof=1)),1e-6);z=abs(float(a.mean())-float(b.mean()))/sd;ratio=float(a.std(ddof=1))/sd if len(a)>1 else 1.;return _clip(100-min(70,25*z+25*abs(math.log(max(.1,ratio)))))
def build_live_monitoring(pred,meta,reports_dir="reports"):
    alerts=[];s={};c=meta.get("market_coverage") or {};games=max(1,int(c.get("games",0) or 0));s["market_coverage"]=_clip(100*min(c.get("moneyline",0),c.get("spread",0),c.get("total",0))/games)
    adv=meta.get("advanced_features") or {};dyn=float(adv.get("dynamic_coverage",adv.get("live_coverage",0)) or 0);fc=int(adv.get("dynamic_feature_count",adv.get("feature_count",0)) or 0);s["advanced_coverage"]=_clip(100*dyn)*min(1.,fc/10 if fc else 0.)
    intel=meta.get("market_intelligence") or {};s["multi_book"]=_clip(100*float(intel.get("multi_book_coverage",0) or 0));ctx=meta.get("current_context") or {};context_cov=float(ctx.get("coverage",0) or 0);weather=float(ctx.get("weather_coverage",0) or 0);s["context"]=_clip(100*(.65*context_cov+.35*weather))
    m=meta.get("metrics") or {};brier=m.get("win_brier");ece=m.get("win_ece");s["calibration"]=_clip(100-220*max(0,float(brier or .25)-.20)-250*float(ece or .10))
    try:gen=datetime.fromisoformat(str(meta.get("generated_at")).replace("Z","+00:00"));age=(datetime.now(timezone.utc)-gen.astimezone(timezone.utc)).total_seconds()/3600
    except Exception:age=999
    s["freshness"]=_clip(100-6*max(0,age-1));
    if age>8:alerts.append(f"model output is {age:.1f} hours old")
    refs=Path(reports_dir)/"backtest_predictions.csv";ds=[]
    if refs.exists() and not pred.empty:
        try:
            h=pd.read_csv(refs,low_memory=False)
            for lc,hc,n in [("model_margin_home","pred_margin_home","margin"),("model_total","pred_total","total"),("calibrated_home_probability","home_win_probability","probability")]:
                if lc in pred and hc in h:
                    x=_drift(pred[lc],h[hc]);
                    if x is not None:ds.append(x);s["drift_"+n]=round(x,1)
        except Exception as e:alerts.append(f"drift reference unavailable: {type(e).__name__}")
    s["distribution_stability"]=round(float(np.mean(ds)),1) if ds else 65.;missing=float(pred.isna().mean().mean()) if not pred.empty else 1.;s["output_completeness"]=_clip(100*(1-min(.5,missing)/.5))
    w={"market_coverage":.13,"advanced_coverage":.17,"multi_book":.09,"context":.10,"calibration":.18,"freshness":.09,"distribution_stability":.14,"output_completeness":.10};total=sum(w[k]*s.get(k,0) for k in w)
    if s["market_coverage"]<90:alerts.append("verified ML/spread/total coverage below 90%")
    if s["advanced_coverage"]<70:alerts.append("real pregame advanced-efficiency coverage below target")
    if s["multi_book"]<50:alerts.append("multi-book consensus is limited; configure THE_ODDS_API_KEY for free multi-book supplementation")
    if s["context"]<60:alerts.append("injury/weather/travel context coverage is limited")
    return {"live_readiness_score":round(total,1),"scores":s,"alerts":alerts,"meaning":"operational/model-monitoring score; not a profitability guarantee"}
def write_live_monitoring(pred,meta,output="outputs/live_monitoring.json",reports_dir="reports"):
    r=build_live_monitoring(pred,meta,reports_dir);p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2));return r
