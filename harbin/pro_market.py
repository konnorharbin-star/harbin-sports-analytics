from __future__ import annotations

import math
import numpy as np

from .execution_market import install_execution_timestamp_patch
from .edge_regimes import effective_edge_status, match_edge_regime, match_edge_subgroup
from .market import roi
from .policy import signal_from_policy, load_policy, DEFAULT_POLICY, market_allowed

# Pipeline imports the base market class before this module. Patch that class object in
# place so every subsequent market attach includes selected-quote timestamps.
install_execution_timestamp_patch()

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


def _market_reliability_multiplier(market, side):
    """Conservative prior from the large historical diagnostic sample.

    This is deliberately one-sided: weak historical segments can reduce sizing but
    never create an edge or increase a stake.  The live performance-feedback layer
    remains the authority once enough timestamp-safe forward bets accumulate.
    """
    if market == "total" and str(side).upper() == "O":
        return 0.65
    return 1.0


def select_best_market(row, risk_multiplier=1.0, policy_path=PRODUCTION_POLICY_PATH, edge_report=None):
    """Select an executable market, prioritizing proven persistent edge regimes.

    Historical regime evidence can change which already-qualified positive-EV market
    is selected, but never changes the fair projection, probability, raw EV, tier,
    or increases sizing.
    """
    candidates=[]; week=row.get("week")

    if _finite(row.get("quant_best_ml_roi")):
        side=row.get("quant_best_ml_side"); p=None; odds=None; book=None; quote_at=None
        if side==row.get("home_team"):
            p=row.get("calibrated_home_probability"); use_best=_finite(row.get("best_home_ml")); odds=row.get("best_home_ml") if use_best else row.get("home_ml"); book=row.get("best_home_ml_book") if use_best else row.get("provider"); quote_at=row.get("best_home_ml_quote_at") if use_best else None
        elif side==row.get("away_team"):
            p=1-float(row.get("calibrated_home_probability")); use_best=_finite(row.get("best_away_ml")); odds=row.get("best_away_ml") if use_best else row.get("away_ml"); book=row.get("best_away_ml_book") if use_best else row.get("provider"); quote_at=row.get("best_away_ml_quote_at") if use_best else None
        if p is not None and _finite(odds):
            edge=float(row.get("quant_best_ml_edge_pp",0)); ev=roi(float(p),float(odds)); sig=quant_signal(ev,edge,p,"moneyline",policy_path,week=week)
            candidates.append({"ev":float(ev),"market":"moneyline","side":side,"line":float(odds),"odds":float(odds),"p":float(p),"edge":edge,"signal":sig,"book":book,"quote_at":quote_at})

    if _finite(row.get("cover_probability")) and _finite(row.get("spread_edge_pts")):
        p=float(row["cover_probability"]); edge=float(row["spread_edge_pts"]); side=row.get("spread_team")
        if side==row.get("home_team"):
            odds=row.get("best_home_spread_odds"); book=row.get("best_home_spread_book"); quote_at=row.get("best_home_spread_quote_at")
        elif side==row.get("away_team"):
            odds=row.get("best_away_spread_odds"); book=row.get("best_away_spread_book"); quote_at=row.get("best_away_spread_quote_at")
        else:
            odds=book=quote_at=None
        # No synthetic -110. If an executable spread price is absent, this market is PASS.
        if _finite(odds) and _finite(row.get("spread_line")):
            odds=float(odds); ev=roi(p,odds); sig=quant_signal(ev,edge,p,"spread",policy_path,week=week)
            candidates.append({"ev":ev,"market":"spread","side":side,"line":row.get("spread_line"),"odds":odds,"p":p,"edge":edge,"signal":sig,"book":book,"quote_at":quote_at})

    if _finite(row.get("total_probability")) and _finite(row.get("total_edge_pts")):
        p=float(row["total_probability"]); edge=float(row["total_edge_pts"]); side=row.get("total_dir")
        if side=="O": odds=row.get("best_over_odds"); book=row.get("best_over_book"); quote_at=row.get("best_over_quote_at")
        elif side=="U": odds=row.get("best_under_odds"); book=row.get("best_under_book"); quote_at=row.get("best_under_quote_at")
        else: odds=book=quote_at=None
        # No synthetic -110. If an executable total price is absent, this market is PASS.
        if _finite(odds) and _finite(row.get("market_total")):
            odds=float(odds); ev=roi(p,odds); sig=quant_signal(ev,edge,p,"total",policy_path,week=week)
            candidates.append({"ev":ev,"market":"total","side":side,"line":row.get("market_total"),"odds":odds,"p":p,"edge":edge,"signal":sig,"book":book,"quote_at":quote_at})

    candidates=[x for x in candidates if x["signal"]!="PASS"]
    if not candidates:
        blocked=[]
        for m in ("moneyline","spread","total"):
            allowed,reason=market_allowed(m,week=week,path=policy_path)
            if not allowed: blocked.append(f"{m}: {reason}")
        return {"quant_signal":"PASS","quant_market":None,"quant_side":None,"quant_book":None,"quant_quote_at":None,"quant_ev":0.0,"stake_units":0.0,"quant_probability":np.nan,"quant_odds":np.nan,"policy_block_reason":" | ".join(blocked)}

    raw_best=max(candidates,key=lambda x:x["ev"])
    for candidate in candidates:
        regime=match_edge_regime(edge_report,candidate["market"],candidate["edge"]) if edge_report else None
        subgroup=match_edge_subgroup(
            edge_report,
            candidate["market"],
            candidate["edge"],
            candidate["side"],
            candidate["line"],
            row.get("home_team"),
            row.get("away_team"),
        ) if edge_report else None
        effective=effective_edge_status(regime,subgroup)
        evidence_stats=subgroup if effective=="SUPPORTED_SUBGROUP" else regime
        candidate.update(price_evidence(candidate["odds"],evidence_stats))
        candidate["edge_regime_parent_status"]=(regime or {}).get("status","UNSUPPORTED")
        candidate["edge_regime_status"]=effective
        candidate["edge_regime_band"]=(regime or {}).get("edge_band","")
        candidate["edge_regime_bets"]=int((regime or {}).get("bets",0) or 0)
        candidate["edge_regime_roi"]=(regime or {}).get("roi")
        candidate["edge_regime_profitable_seasons"]=int((regime or {}).get("profitable_seasons",0) or 0)
        candidate["edge_regime_season_count"]=int((regime or {}).get("season_count",0) or 0)
        candidate["edge_subgroup_key"]=(subgroup or {}).get("key","")
        candidate["edge_subgroup_status"]=(subgroup or {}).get("status","INCONCLUSIVE_SUBGROUP")
        candidate["edge_subgroup_bets"]=int((subgroup or {}).get("bets",0) or 0)
        candidate["edge_subgroup_roi"]=(subgroup or {}).get("roi")
        candidate["edge_subgroup_profitable_seasons"]=int((subgroup or {}).get("profitable_seasons",0) or 0)
        candidate["edge_subgroup_season_count"]=int((subgroup or {}).get("season_count",0) or 0)
        candidate["edge_subgroup_discovery_roi"]=(subgroup or {}).get("discovery_roi")
        candidate["edge_subgroup_holdout_season"]=(subgroup or {}).get("holdout_season","")
        candidate["edge_subgroup_holdout_bets"]=int((subgroup or {}).get("holdout_bets",0) or 0)
        candidate["edge_subgroup_holdout_roi"]=(subgroup or {}).get("holdout_roi")
        candidate["edge_subgroup_holdout_win_rate"]=(subgroup or {}).get("holdout_win_rate")
        candidate["edge_subgroup_holdout_confirmed"]=bool((subgroup or {}).get("holdout_confirmed",False))
        price_status=candidate.get("price_evidence_status","UNKNOWN")
        candidate["edge_promotion_rank"]=(
            3 if effective=="SUPPORTED_SUBGROUP" and price_status=="CONFIRMED"
            else 2 if effective=="SUPPORTED_SUBGROUP" and price_status=="PLAUSIBLE"
            else 1 if effective=="PERSISTENT_PARENT_ONLY" and price_status=="CONFIRMED"
            else 0
        )

    contraindicated=[
        x for x in candidates
        if x["edge_regime_status"]=="CONTRAINDICATED_SUBGROUP"
    ]
    eligible=[
        x for x in candidates
        if x["edge_regime_status"]!="CONTRAINDICATED_SUBGROUP"
    ]
    raw_best_contraindicated=raw_best["edge_regime_status"]=="CONTRAINDICATED_SUBGROUP"

    if not eligible:
        return {
            "quant_signal":"PASS",
            "quant_market":None,
            "quant_side":None,
            "quant_book":None,
            "quant_quote_at":None,
            "quant_price":np.nan,
            "quant_odds":np.nan,
            "quant_ev":0.0,
            "quant_probability":np.nan,
            "quant_edge":0.0,
            "stake_units":0.0,
            "selection_basis":"contraindicated_edge_veto",
            "edge_selection_override":True,
            "edge_contraindicated_veto":True,
            "edge_contraindicated_candidates":len(contraindicated),
            "raw_ev_best_market":raw_best["market"],
            "raw_ev_best_side":raw_best["side"],
            "raw_ev_best_ev":raw_best["ev"],
            "raw_ev_best_edge_status":raw_best["edge_regime_status"],
            "policy_block_reason":"all qualified markets are chronologically contraindicated edge subgroups",
        }

    promotable=[x for x in eligible if x["edge_promotion_rank"]>0]
    eligible_raw_best=max(eligible,key=lambda x:x["ev"])
    best=max(promotable,key=lambda x:(x["edge_promotion_rank"],x["ev"])) if promotable else eligible_raw_best
    override=(
        best["market"]!=raw_best["market"] or str(best["side"])!=str(raw_best["side"])
    )
    segment_multiplier=_market_reliability_multiplier(best["market"],best["side"])
    units=fractional_kelly_units(best["p"],best["odds"],risk_multiplier*segment_multiplier,policy_path=policy_path)
    if best["edge_regime_status"]=="SUPPORTED_SUBGROUP":
        selection_basis="supported_edge_subgroup"
    elif best["edge_regime_status"]=="PERSISTENT_PARENT_ONLY":
        selection_basis="persistent_parent_regime"
    elif raw_best_contraindicated:
        selection_basis="highest_raw_ev_after_contraindicated_veto"
    else:
        selection_basis="highest_raw_ev"
    return {
        "quant_signal":best["signal"],"quant_market":best["market"],"quant_side":best["side"],"quant_book":best.get("book"),"quant_quote_at":best.get("quote_at"),
        "quant_price":best["line"],"quant_odds":best["odds"],"quant_ev":best["ev"],
        "quant_probability":best["p"],"quant_edge":best["edge"],"stake_units":round(units,2),
        "historical_segment_multiplier":segment_multiplier,
        "selection_basis":selection_basis,
        "edge_selection_override":override,
        "edge_contraindicated_veto":raw_best_contraindicated,
        "edge_contraindicated_candidates":len(contraindicated),
        "raw_ev_best_market":raw_best["market"],
        "raw_ev_best_side":raw_best["side"],
        "raw_ev_best_ev":raw_best["ev"],
        "raw_ev_best_edge_status":raw_best["edge_regime_status"],
        "edge_regime_parent_status":best["edge_regime_parent_status"],
        "edge_regime_status":best["edge_regime_status"],
        "edge_regime_band":best["edge_regime_band"],
        "edge_regime_bets":best["edge_regime_bets"],
        "edge_regime_roi":best["edge_regime_roi"],
        "edge_regime_profitable_seasons":best["edge_regime_profitable_seasons"],
        "edge_regime_season_count":best["edge_regime_season_count"],
        "edge_regime_candidate":best["edge_regime_status"] in {"SUPPORTED_SUBGROUP","PERSISTENT_PARENT_ONLY"},
        "price_evidence_status":best.get("price_evidence_status","UNKNOWN"),
        "price_evidence_sample":best.get("price_evidence_sample",0),
        "historical_price_win_rate":best.get("historical_price_win_rate"),
        "historical_price_wilson_lower":best.get("historical_price_wilson_lower"),
        "current_break_even_probability":best.get("current_break_even_probability"),
        "historical_price_margin":best.get("historical_price_margin"),
        "conservative_price_margin":best.get("conservative_price_margin"),
        "historical_fair_odds_lower_bound":best.get("historical_fair_odds_lower_bound"),
        "edge_subgroup_key":best["edge_subgroup_key"],
        "edge_subgroup_status":best["edge_subgroup_status"],
        "edge_subgroup_bets":best["edge_subgroup_bets"],
        "edge_subgroup_roi":best["edge_subgroup_roi"],
        "edge_subgroup_profitable_seasons":best["edge_subgroup_profitable_seasons"],
        "edge_subgroup_season_count":best["edge_subgroup_season_count"],
        "edge_subgroup_discovery_roi":best["edge_subgroup_discovery_roi"],
        "edge_subgroup_holdout_season":best["edge_subgroup_holdout_season"],
        "edge_subgroup_holdout_bets":best["edge_subgroup_holdout_bets"],
        "edge_subgroup_holdout_roi":best["edge_subgroup_holdout_roi"],
        "edge_subgroup_holdout_win_rate":best["edge_subgroup_holdout_win_rate"],
        "edge_subgroup_holdout_confirmed":best["edge_subgroup_holdout_confirmed"],
        "policy_block_reason":"",
    }


def risk_multiplier(availability_risk=0.0, data_quality=1.0, volatility=0.0):
    a=max(0.0,min(1.0,float(availability_risk or 0))); q=max(0.0,min(1.0,float(data_quality or 0))); v=max(0.0,float(volatility or 0)); vol_penalty=max(.55,1-v/60)
    return max(.15,min(1.0,(1-.55*a)*(0.55+.45*q)*vol_penalty))
