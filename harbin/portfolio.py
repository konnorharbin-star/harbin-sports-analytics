from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd

from .policy import load_policy


def _finite(v):
    try: return math.isfinite(float(v))
    except Exception: return False

def _read_gate(path):
    p=Path(path)
    if not p.exists(): return {"release_state":"PAPER","production_eligible":False,"blockers":["release gate missing"]}
    try: return json.loads(p.read_text())
    except Exception: return {"release_state":"PAPER","production_eligible":False,"blockers":["release gate unreadable"]}

def apply_portfolio_controls(pred: pd.DataFrame, policy_path="reports/production_policy.json",
                             release_gate_path="outputs/release_gate.json") -> tuple[pd.DataFrame,dict]:
    policy=load_policy(policy_path); gate=_read_gate(release_gate_path); limits=policy.get("portfolio") or {}; out=pred.copy()
    gate_state=str(gate.get("release_state","PAPER")).upper(); policy_mode=str(policy.get("deployment_mode","paper")).lower()
    production_allowed=bool(gate.get("production_eligible")) and policy_mode=="production"
    effective_mode="production" if production_allowed else "shadow" if gate_state=="SHADOW" else "paper"
    if out.empty:
        return out,{"mode":effective_mode,"proposed_units":0.,"approved_units":0.,"bets":0,"release_state":gate_state}
    out["paper_stake_units"]=pd.to_numeric(out.get("stake_units",0),errors="coerce").fillna(0.)
    out["portfolio_stake_units"]=0.; out["portfolio_action"]="PASS"
    score=pd.to_numeric(out.get("quant_ev",0),errors="coerce").fillna(0)*pd.to_numeric(out.get("risk_multiplier",1),errors="coerce").fillna(0)
    order=list(score.sort_values(ascending=False).index)
    slate_cap=float(limits.get("max_slate_units",5.)); game_cap=float(limits.get("max_game_units",1.)); team_cap=float(limits.get("max_team_units",1.5)); market_cap=float(limits.get("max_market_units",2.5))
    slate=0.; by_team={}; by_market={}
    for i in order:
        r=out.loc[i]; signal=str(r.get("quant_signal") or "PASS").upper(); proposed=float(r.get("paper_stake_units") or 0)
        if signal=="PASS" or proposed<=0: continue
        teams=[str(r.get("home_team") or ""),str(r.get("away_team") or "")]; market=str(r.get("quant_market") or "unknown")
        remaining=min(game_cap,slate_cap-slate,market_cap-by_market.get(market,0.),*(team_cap-by_team.get(t,0.) for t in teams if t))
        units=max(0.,min(proposed,remaining))
        if units<.05: continue
        out.at[i,"portfolio_stake_units"]=round(units if production_allowed else 0.,2)
        out.at[i,"portfolio_action"]="BET" if production_allowed else "SHADOW" if effective_mode=="shadow" else "PAPER"
        slate+=units; by_market[market]=by_market.get(market,0.)+units
        for t in teams:
            if t: by_team[t]=by_team.get(t,0.)+units
    summary={
        "mode":effective_mode,"policy_mode":policy_mode,"release_state":gate_state,"production_eligible":production_allowed,
        "proposed_units":round(float(out.paper_stake_units.sum()),2),"approved_units":round(float(out.portfolio_stake_units.sum()),2),
        "paper_or_shadow_allocated_units":round(float(slate),2),"bets":int((out.portfolio_action!="PASS").sum()),"limits":limits,
        "release_blockers":gate.get("blockers",[])[:20],
    }
    return out,summary

def write_portfolio_outputs(pred: pd.DataFrame, output_dir="outputs", policy_path="reports/production_policy.json",
                            release_gate_path="outputs/release_gate.json"):
    out,summary=apply_portfolio_controls(pred,policy_path,release_gate_path); p=Path(output_dir); p.mkdir(parents=True,exist_ok=True)
    cols=[c for c in ["date","away_team","home_team","quant_signal","quant_market","quant_side","quant_price","quant_probability","quant_ev","quant_edge","risk_multiplier","data_quality_score","paper_stake_units","portfolio_stake_units","portfolio_action"] if c in out.columns]
    out[cols].to_csv(p/"portfolio_card.csv",index=False); (p/"portfolio_summary.json").write_text(json.dumps(summary,indent=2))
    return out,summary
