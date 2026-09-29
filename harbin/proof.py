from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def _segment(df, col):
    out={}
    if col not in df.columns: return out
    for k,g in df.groupby(col,dropna=False):
        p=pd.to_numeric(g.get("profit"),errors="coerce").dropna(); c=pd.to_numeric(g.get("clv"),errors="coerce").dropna()
        out[str(k)]={"bets":int(len(p)),"roi":float(p.mean()) if len(p) else None,"units":float(p.sum()) if len(p) else 0.0,"avg_clv":float(c.mean()) if len(c) else None}
    return out


def build_evidence_report(summary_path="reports/backtest_summary.json", bets_path="reports/backtest_bets.csv", out_path="reports/evidence_report.json"):
    summary={}
    try: summary=json.loads(Path(summary_path).read_text())
    except Exception: pass
    bets=pd.read_csv(bets_path,low_memory=False) if Path(bets_path).exists() else pd.DataFrame()
    overall=summary.get("overall") or {}; n=int(overall.get("bets",0) or 0); roi=overall.get("roi"); clv=overall.get("avg_clv"); ci=overall.get("roi_ci_95") or [None,None]
    by_market=_segment(bets,"market") if len(bets) else {}; by_signal=_segment(bets,"signal") if len(bets) else {}; by_season=_segment(bets,"season") if len(bets) else {}; by_week=_segment(bets,"week") if len(bets) else {}
    market_positive=sum(1 for x in by_market.values() if x["bets"]>=50 and (x["roi"] or -1)>0); season_positive=sum(1 for x in by_season.values() if x["bets"]>=50 and (x["roi"] or -1)>0)
    if n>=1000 and ci[0] is not None and float(ci[0])>0 and clv is not None and float(clv)>0 and market_positive>=2 and season_positive>=2: status="ROBUST"
    elif n>=500 and roi is not None and float(roi)>0 and clv is not None and float(clv)>0: status="VALIDATED"
    elif n>=150: status="DEVELOPING"
    else: status="UNPROVEN"
    report={"status":status,"overall":overall,"by_market":by_market,"by_signal":by_signal,"by_season":by_season,"by_week":by_week,"criteria":{"robust":"1000+ bets, ROI 95% CI lower bound > 0, positive CLV, positive evidence across >=2 markets and >=2 seasons","validated":"500+ bets with positive ROI and CLV","developing":"150+ bets","unproven":"below developing sample"},"note":"Historical evidence can fail to persist out of sample; this report does not guarantee future profit."}
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(report,indent=2)); return report
