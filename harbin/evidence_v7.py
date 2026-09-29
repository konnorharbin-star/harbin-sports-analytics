from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .policy import load_policy


def _pct(x):
    try:return float(x)
    except Exception:return None


def validate_policy_against_backtest(
    bets_path="reports/backtest_bets.csv",
    policy_path="reports/production_policy.json",
    out_path="reports/policy_validation.json",
):
    p=load_policy(policy_path); bp=Path(bets_path)
    out={"policy_version":p.get("version"),"deployment_mode":p.get("deployment_mode","paper"),"markets":{},"blocked_weeks":(p.get("regime_filters") or {}).get("blocked_weeks",[]),"status":"UNPROVEN"}
    if not bp.exists():
        Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(out,indent=2)); return out
    df=pd.read_csv(bp,low_memory=False)
    if df.empty:
        Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(out,indent=2)); return out
    last=int(pd.to_numeric(df.get("season"),errors="coerce").dropna().max()) if "season" in df else None
    hold=df[df.season==last].copy() if last is not None else df.iloc[int(len(df)*.7):].copy()
    blocked={int(x) for x in out["blocked_weeks"]}
    if "week" in hold:hold=hold[~pd.to_numeric(hold.week,errors="coerce").fillna(-1).astype(int).isin(blocked)]
    total_profit=[]
    for market,cfg in (p.get("markets") or {}).items():
        m=hold[hold.market.astype(str).str.lower()==market].copy() if "market" in hold else hold.iloc[0:0]
        enabled=bool(cfg.get("enabled",False))
        if not enabled:
            out["markets"][market]={"enabled":False,"bets":0,"roi":None,"units":0.0,"reason":"not statistically validated"}; continue
        q=cfg.get("lean") or {}; ev=pd.to_numeric(m.get("ev"),errors="coerce"); edge=pd.to_numeric(m.get("edge"),errors="coerce").abs(); prob=pd.to_numeric(m.get("probability"),errors="coerce")
        sel=m[(ev>=float(q.get("min_ev",1)))&(edge>=float(q.get("min_edge",999)))&(prob>=float(q.get("min_prob",1)))].copy()
        profits=pd.to_numeric(sel.get("profit"),errors="coerce").dropna(); roi=float(profits.mean()) if len(profits) else None; units=float(profits.sum()) if len(profits) else 0.0
        clv=pd.to_numeric(sel.get("clv"),errors="coerce").dropna(); avg_clv=float(clv.mean()) if len(clv) else None
        out["markets"][market]={"enabled":True,"bets":int(len(profits)),"roi":roi,"units":units,"avg_clv":avg_clv,"holdout_season":last,"evidence_tier":cfg.get("evidence_tier")}
        total_profit.extend(profits.tolist())
    arr=np.asarray(total_profit,float)
    overall_roi=float(arr.mean()) if len(arr) else None
    out["overall"]={"bets":int(len(arr)),"units":float(arr.sum()) if len(arr) else 0.0,"roi":overall_roi}
    validated=[m for m,v in out["markets"].items() if v.get("enabled") and v.get("bets",0)>=100 and v.get("roi") is not None and v.get("roi")>0]
    out["status"]="FORWARD_PAPER_READY" if validated else "RESEARCH_ONLY"
    out["validated_markets_in_holdout"]=validated
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(out,indent=2)); return out


def build_upgrade_audit(meta_path="outputs/system_health.json", backtest_path="reports/backtest_summary.json", policy_validation_path="reports/policy_validation.json", out_path="reports/v7_audit.json"):
    def read(path):
        try:return json.loads(Path(path).read_text())
        except Exception:return {}
    health,bt,pv=read(meta_path),read(backtest_path),read(policy_validation_path)
    comp=health.get("components") or {}
    findings=[]
    if comp.get("market_intelligence",0)<90:findings.append("Multi-book consensus remains an external-data limitation unless a second independent live book source is configured.")
    if comp.get("live_monitoring",0)<90:findings.append("Live monitoring cannot be considered mature until enough forward snapshots accumulate for drift and CLV stability.")
    if (bt.get("overall") or {}).get("roi",0)<=0:findings.append("Raw historical signal set is not profitable overall; production gates must remain selective and paper-only.")
    if (bt.get("advanced_features") or {}).get("dynamic_coverage",0)<.8:findings.append("Historical dynamic advanced-feature coverage is below target; identity matching/source coverage needs verification.")
    report={
        "engineering_readiness":health.get("system_health_score"),
        "betting_proof_status":health.get("betting_proof_status"),
        "policy_validation":pv,
        "raw_backtest_roi":(bt.get("overall") or {}).get("roi"),
        "raw_backtest_bets":(bt.get("overall") or {}).get("bets"),
        "findings":findings,
        "release_gate":{
            "real_money_allowed":False,
            "why":"A high-quality research stack can be complete before a market edge is statistically proven. Real-money promotion requires positive forward evidence, not a software score.",
            "requirements":["positive holdout/forward ROI with uncertainty bounds","positive CLV across a meaningful sample","stable probability calibration","multi-book or independently verified market prices","no unresolved data-quality alerts"]
        }
    }
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(report,indent=2)); return report
