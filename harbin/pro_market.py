from __future__ import annotations

import math
import numpy as np

from .market import roi
from .policy import signal_from_policy, load_policy, DEFAULT_POLICY, market_allowed

PRODUCTION_POLICY_PATH="reports/production_policy.json"


def decimal_odds(american):
    o=float(american); return 1+((100/abs(o)) if o<0 else (o/100))


def kelly_fraction(p, american):
    d=decimal_odds(american); b=d-1; q=1-float(p)
    return max(0.0,(b*float(p)-q)/b) if b>0 else 0.0


def fractional_kelly_units(p, american, risk_multiplier=1.0, cap=1.25, policy_path=PRODUCTION_POLICY_PATH):
    policy=load_policy(policy_path); frac=float((policy.get("portfolio") or {}).get("kelly_fraction",.20)); raw=max(0.0,kelly_fraction(p,american))*frac*4.0
    return min(float(cap), raw*max(.0,min(1.,float(risk_multiplier))))


def standard_price_ev(p, price=-110):
    return roi(float(p),float(price))


def _finite(v):
    try: return math.isfinite(float(v))
    except Exception: return False


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


def quant_signal(ev, edge_value, probability, market: str, policy_path=None, week=None):
    """Classify an edge without contaminating research with generated policy state."""
    if policy_path is None:
        return _signal_from_thresholds(ev,edge_value,probability,market,DEFAULT_POLICY.get("markets",{}))
    return signal_from_policy(ev,edge_value,probability,market,path=policy_path,week=week)


def select_best_market(row, risk_multiplier=1.0, policy_path=PRODUCTION_POLICY_PATH):
    """Select the best executable market and carry its sportsbook provenance."""
    candidates=[]; week=row.get("week")

    if _finite(row.get("quant_best_ml_roi")):
        side=row.get("quant_best_ml_side"); p=None; odds=None; book=None
        if side==row.get("home_team"):
            p=row.get("calibrated_home_probability")
            use_best=_finite(row.get("best_home_ml")); odds=row.get("best_home_ml") if use_best else row.get("home_ml")
            book=row.get("best_home_ml_book") if use_best else row.get("provider")
        elif side==row.get("away_team"):
            p=1-float(row.get("calibrated_home_probability")); use_best=_finite(row.get("best_away_ml")); odds=row.get("best_away_ml") if use_best else row.get("away_ml")
            book=row.get("best_away_ml_book") if use_best else row.get("provider")
        if p is not None and _finite(odds):
            edge=float(row.get("quant_best_ml_edge_pp",0)); ev=roi(float(p),float(odds)); sig=quant_signal(ev,edge,p,"moneyline",policy_path,week=week)
            candidates.append({"ev":float(ev),"market":"moneyline","side":side,"line":float(odds),"odds":float(odds),"p":float(p),"edge":edge,"signal":sig,"book":book})

    if _finite(row.get("cover_probability")) and _finite(row.get("spread_edge_pts")):
        p=float(row["cover_probability"]); edge=float(row["spread_edge_pts"])
        odds=float(row.get("spread_odds")) if _finite(row.get("spread_odds")) else -110.0
        side=row.get("spread_team")
        if side==row.get("home_team"): book=row.get("best_home_spread_book") or row.get("provider")
        elif side==row.get("away_team"): book=row.get("best_away_spread_book") or row.get("provider")
        else: book=row.get("provider")
        ev=roi(p,odds); sig=quant_signal(ev,edge,p,"spread",policy_path,week=week)
        candidates.append({"ev":ev,"market":"spread","side":side,"line":row.get("spread_line"),"odds":odds,"p":p,"edge":edge,"signal":sig,"book":book})

    if _finite(row.get("total_probability")) and _finite(row.get("total_edge_pts")):
        p=float(row["total_probability"]); edge=float(row["total_edge_pts"])
        odds=float(row.get("total_odds")) if _finite(row.get("total_odds")) else -110.0
        side=row.get("total_dir"); book=(row.get("best_over_book") if side=="O" else row.get("best_under_book") if side=="U" else None) or row.get("provider")
        ev=roi(p,odds); sig=quant_signal(ev,edge,p,"total",policy_path,week=week)
        candidates.append({"ev":ev,"market":"total","side":side,"line":row.get("market_total"),"odds":odds,"p":p,"edge":edge,"signal":sig,"book":book})

    candidates=[x for x in candidates if x["signal"]!="PASS"]
    if not candidates:
        blocked=[]
        for m in ("moneyline","spread","total"):
            allowed,reason=market_allowed(m,week=week,path=policy_path)
            if not allowed: blocked.append(f"{m}: {reason}")
        return {"quant_signal":"PASS","quant_market":None,"quant_side":None,"quant_book":None,"quant_ev":0.0,"stake_units":0.0,"quant_probability":np.nan,"quant_odds":np.nan,"policy_block_reason":" | ".join(blocked)}

    best=max(candidates,key=lambda x:x["ev"])
    units=fractional_kelly_units(best["p"],best["odds"],risk_multiplier,policy_path=policy_path)
    return {
        "quant_signal":best["signal"],"quant_market":best["market"],"quant_side":best["side"],"quant_book":best.get("book"),
        "quant_price":best["line"],"quant_odds":best["odds"],"quant_ev":best["ev"],
        "quant_probability":best["p"],"quant_edge":best["edge"],"stake_units":round(units,2),
        "policy_block_reason":"",
    }


def risk_multiplier(availability_risk=0.0, data_quality=1.0, volatility=0.0):
    a=max(0.0,min(1.0,float(availability_risk or 0))); q=max(0.0,min(1.0,float(data_quality or 0))); v=max(0.0,float(volatility or 0)); vol_penalty=max(.55,1-v/60)
    return max(.15,min(1.0,(1-.55*a)*(0.55+.45*q)*vol_penalty))
