from __future__ import annotations

import math
import numpy as np

from .market import american_implied, no_vig, roi


def decimal_odds(american):
    o=float(american); return 1+((100/abs(o)) if o<0 else (o/100))


def kelly_fraction(p, american):
    d=decimal_odds(american); b=d-1; q=1-float(p)
    return max(0.0,(b*float(p)-q)/b) if b>0 else 0.0


def quarter_kelly_units(p, american, cap=1.5):
    return min(float(cap), 4.0*0.25*kelly_fraction(p,american))


def standard_price_ev(p, price=-110):
    return roi(float(p),float(price))


def quant_signal(ev, edge_value, probability, market: str):
    """Independent quant gate; intentionally separate from Cooper replica badges."""
    ev=float(ev); edge=abs(float(edge_value)); p=float(probability)
    if ev<=0: return "PASS"
    if market=="moneyline":
        if ev>=.07 and edge>=4 and p>=.56: return "STRONG"
        if ev>=.04 and edge>=2.5: return "BET"
        if ev>=.02 and edge>=1.5: return "LEAN"
    elif market=="spread":
        if ev>=.07 and edge>=5.0 and p>=.57: return "STRONG"
        if ev>=.04 and edge>=3.0: return "BET"
        if ev>=.02 and edge>=2.0: return "LEAN"
    elif market=="total":
        if ev>=.07 and edge>=6.0 and p>=.57: return "STRONG"
        if ev>=.04 and edge>=4.0: return "BET"
        if ev>=.02 and edge>=2.5: return "LEAN"
    return "PASS"


def select_best_market(row, risk_multiplier=1.0):
    candidates=[]
    if not np.isnan(row.get("quant_best_ml_roi",np.nan)):
        side=row.get("quant_best_ml_side"); p=None; odds=None
        if side==row.get("home_team"): p=row.get("calibrated_home_probability"); odds=row.get("home_ml")
        elif side==row.get("away_team"): p=1-float(row.get("calibrated_home_probability")); odds=row.get("away_ml")
        if p is not None and odds is not None:
            edge=row.get("quant_best_ml_edge_pp",0); ev=row.get("quant_best_ml_roi",-1); sig=quant_signal(ev,edge,p,"moneyline")
            candidates.append((float(ev),"moneyline",side,odds,float(p),float(edge),sig))
    if not np.isnan(row.get("cover_probability",np.nan)) and not np.isnan(row.get("spread_edge_pts",np.nan)):
        p=float(row["cover_probability"]); ev=standard_price_ev(p); edge=float(row["spread_edge_pts"]); sig=quant_signal(ev,edge,p,"spread")
        candidates.append((ev,"spread",row.get("spread_team"),row.get("spread_line"),p,edge,sig))
    if not np.isnan(row.get("total_probability",np.nan)) and not np.isnan(row.get("total_edge_pts",np.nan)):
        p=float(row["total_probability"]); ev=standard_price_ev(p); edge=float(row["total_edge_pts"]); sig=quant_signal(ev,edge,p,"total")
        candidates.append((ev,"total",row.get("total_dir"),row.get("market_total"),p,edge,sig))
    candidates=[x for x in candidates if x[6]!="PASS"]
    if not candidates: return {"quant_signal":"PASS","quant_market":None,"quant_side":None,"quant_ev":0.0,"stake_units":0.0,"quant_probability":np.nan}
    best=max(candidates,key=lambda x:x[0]); ev,market,side,price,p,edge,sig=best
    if market=="moneyline": units=quarter_kelly_units(p,price)
    else:
        # standard -110 Kelly translated into a small unit scale
        units=min(1.5,max(0.0,(ev/.05)*.5))
    units*=max(.15,min(1.0,float(risk_multiplier)))
    return {"quant_signal":sig,"quant_market":market,"quant_side":side,"quant_price":price,"quant_ev":ev,"quant_probability":p,"quant_edge":edge,"stake_units":round(units,2)}


def risk_multiplier(availability_risk=0.0, data_quality=1.0, volatility=0.0):
    a=max(0.0,min(1.0,float(availability_risk or 0))); q=max(0.0,min(1.0,float(data_quality or 0)))
    v=max(0.0,float(volatility or 0)); vol_penalty=max(.55,1-v/60)
    return max(.15,min(1.0,(1-.55*a)*(0.55+.45*q)*vol_penalty))
