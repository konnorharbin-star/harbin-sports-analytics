from __future__ import annotations

import math
import numpy as np

from .market import roi
from .policy import signal_from_policy, load_policy, DEFAULT_POLICY

PRODUCTION_POLICY_PATH="reports/production_policy.json"


def decimal_odds(american):
    o=float(american); return 1+((100/abs(o)) if o<0 else (o/100))


def kelly_fraction(p, american):
    d=decimal_odds(american); b=d-1; q=1-float(p)
    return max(0.0,(b*float(p)-q)/b) if b>0 else 0.0


def fractional_kelly_units(p, american, risk_multiplier=1.0, cap=1.25, policy_path=PRODUCTION_POLICY_PATH):
    policy=load_policy(policy_path); frac=float((policy.get("portfolio") or {}).get("kelly_fraction",.20)); raw=max(0.0,kelly_fraction(p,american))*frac*4.0
    return min(float(cap), raw*max(.0,min(1.,float(risk_multiplier))))


def standard_price_ev(p, price=-110): return roi(float(p),float(price))


def _signal_from_thresholds(ev, edge_value, probability, market, thresholds):
    try:
        ev=float(ev); edge=abs(float(edge_value)); probability=float(probability)
    except Exception:
        return "PASS"
    if ev<=0: return "PASS"
    t=(thresholds or {}).get(market,{})
    for label in ("strong","bet","lean"):
        q=t.get(label) or {}
        if ev>=float(q.get("min_ev",999)) and edge>=float(q.get("min_edge",999)) and probability>=float(q.get("min_prob",1.1)):
            return label.upper()
    return "PASS"


def quant_signal(ev, edge_value, probability, market: str, policy_path=None):
    """Classify an edge without contaminating research with generated policy state.

    `policy_path=None` intentionally means the fixed conservative defaults. Historical
    backtests and unit tests therefore remain reproducible and cannot recursively use a
    policy derived from their own output. Live selection explicitly passes the validated
    production-policy path.
    """
    if policy_path is None:
        return _signal_from_thresholds(ev,edge_value,probability,market,DEFAULT_POLICY.get("markets",{}))
    return signal_from_policy(ev,edge_value,probability,market,path=policy_path)


def select_best_market(row, risk_multiplier=1.0, policy_path=PRODUCTION_POLICY_PATH):
    candidates=[]
    if not np.isnan(row.get("quant_best_ml_roi",np.nan)):
        side=row.get("quant_best_ml_side"); p=None; odds=None
        if side==row.get("home_team"): p=row.get("calibrated_home_probability"); odds=row.get("home_ml")
        elif side==row.get("away_team"): p=1-float(row.get("calibrated_home_probability")); odds=row.get("away_ml")
        if p is not None and odds is not None:
            edge=row.get("quant_best_ml_edge_pp",0); ev=row.get("quant_best_ml_roi",-1); sig=quant_signal(ev,edge,p,"moneyline",policy_path); candidates.append((float(ev),"moneyline",side,odds,float(p),float(edge),sig))
    if not np.isnan(row.get("cover_probability",np.nan)) and not np.isnan(row.get("spread_edge_pts",np.nan)):
        p=float(row["cover_probability"]); ev=standard_price_ev(p); edge=float(row["spread_edge_pts"]); sig=quant_signal(ev,edge,p,"spread",policy_path); candidates.append((ev,"spread",row.get("spread_team"),row.get("spread_line"),p,edge,sig))
    if not np.isnan(row.get("total_probability",np.nan)) and not np.isnan(row.get("total_edge_pts",np.nan)):
        p=float(row["total_probability"]); ev=standard_price_ev(p); edge=float(row["total_edge_pts"]); sig=quant_signal(ev,edge,p,"total",policy_path); candidates.append((ev,"total",row.get("total_dir"),row.get("market_total"),p,edge,sig))
    candidates=[x for x in candidates if x[6]!="PASS"]
    if not candidates: return {"quant_signal":"PASS","quant_market":None,"quant_side":None,"quant_ev":0.0,"stake_units":0.0,"quant_probability":np.nan}
    ev,market,side,price,p,edge,sig=max(candidates,key=lambda x:x[0])
    if market=="moneyline": units=fractional_kelly_units(p,price,risk_multiplier,policy_path=policy_path)
    else: units=min(1.25,max(0.0,(ev/.05)*.40))*max(.15,min(1.0,float(risk_multiplier)))
    return {"quant_signal":sig,"quant_market":market,"quant_side":side,"quant_price":price,"quant_ev":ev,"quant_probability":p,"quant_edge":edge,"stake_units":round(units,2)}


def risk_multiplier(availability_risk=0.0, data_quality=1.0, volatility=0.0):
    a=max(0.0,min(1.0,float(availability_risk or 0))); q=max(0.0,min(1.0,float(data_quality or 0))); v=max(0.0,float(volatility or 0)); vol_penalty=max(.55,1-v/60)
    return max(.15,min(1.0,(1-.55*a)*(0.55+.45*q)*vol_penalty))
