from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from .market import american_implied, no_vig, roi
from .tier_validation import MARKETS, TIERS, build_tier_performance, expected_display_tier


def _safe(v):
    try:
        x=float(v); return x if math.isfinite(x) else np.nan
    except Exception: return np.nan


def _edge_bucket(v):
    edge=_safe(v)
    if not math.isfinite(edge): return "missing"
    edge=abs(edge)
    if edge>1.0: edge/=100.0
    if edge<.02: return "<2%"
    if edge<.04: return "2-4%"
    if edge<.06: return "4-6%"
    return ">=6%"


def _grade_row(r,actual_margin,actual_total):
    market=str(r.get("quant_market") or ""); side=r.get("quant_side")
    if market=="moneyline":
        home=str(r.get("home_team")); away=str(r.get("away_team")); winner=home if actual_margin>0 else away if actual_margin<0 else None
        return 1 if winner and str(side)==winner else 0 if winner is None else -1
    if market=="spread":
        home=str(r.get("home_team")); line=float(r.get("quant_price")); val=actual_margin+line if str(side)==home else -actual_margin+line
        return 1 if val>1e-9 else -1 if val<-1e-9 else 0
    if market=="total":
        line=float(r.get("quant_price")); val=actual_total-line; val=-val if str(side)=="U" else val
        return 1 if val>1e-9 else -1 if val<-1e-9 else 0
    return np.nan


def _profit(result,price,market=None):
    """Profit in units using the actual executable American price when available."""
    if pd.isna(result): return np.nan
    if result==0: return 0.
    if result<0: return -1.
    o=_safe(price)
    if not math.isfinite(o):
        o=-110.0 if market in {"spread","total"} else np.nan
    if not math.isfinite(o) or o==0: return np.nan
    return 100/abs(o) if o<0 else o/100


def _latest_pre_kickoff_market(snapshot_df,gid,kickoff):
    """Return only a timestamp-valid snapshot strictly before kickoff."""
    if snapshot_df.empty or "game_id" not in snapshot_df.columns or "captured_at" not in snapshot_df.columns: return None
    z=snapshot_df[snapshot_df.game_id.astype(str)==str(gid)].copy()
    if z.empty: return None
    z["_ts"]=pd.to_datetime(z.captured_at,utc=True,errors="coerce"); z=z[z._ts.notna()]
    if pd.notna(kickoff): z=z[z._ts<kickoff]
    z=z.sort_values("_ts")
    return z.iloc[-1] if len(z) else None


def _close_side_probability(close,side,home):
    ch=_safe(close.get("consensus_home_novig_probability"))
    if math.isfinite(ch): return (ch if side==home else 1-ch),"consensus_latest_pre_kickoff_snapshot"
    hm,am=_safe(close.get("home_ml")),_safe(close.get("away_ml"))
    if math.isfinite(hm) and math.isfinite(am):
        pa,ph=no_vig(am,hm); return (ph if side==home else pa),"latest_verified_pre_kickoff_snapshot"
    col="home_ml" if side==home else "away_ml"; cp=_safe(close.get(col))
    if math.isfinite(cp): return american_implied(cp),"latest_verified_pre_kickoff_snapshot"
    return np.nan,"unavailable"


def _entry_side_probability(entry,side,home):
    eh=_safe(entry.get("consensus_home_novig_probability"))
    if math.isfinite(eh): return eh if side==home else 1-eh
    odds=_safe(entry.get("quant_odds"));
    if not math.isfinite(odds): odds=_safe(entry.get("quant_price"))
    return american_implied(odds) if math.isfinite(odds) else np.nan


def _clv_from_market_snapshot(entry,close):
    """Consensus-first CLV benchmark; positive values mean the entry beat the close."""
    if close is None: return np.nan,"none"
    market=str(entry.get("quant_market") or ""); side=str(entry.get("quant_side") or ""); home=str(entry.get("home_team") or "")
    try:
        if market=="spread":
            op=float(entry.quant_price); raw=_safe(close.get("consensus_home_spread")); source="consensus_latest_pre_kickoff_snapshot"
            if not math.isfinite(raw): raw=_safe(close.get("home_spread")); source="latest_verified_pre_kickoff_snapshot"
            if not math.isfinite(raw): return np.nan,"unavailable"
            cp=raw if side==home else -raw
            return float(op-cp),source
        if market=="total":
            op=float(entry.quant_price); cp=_safe(close.get("consensus_total")); source="consensus_latest_pre_kickoff_snapshot"
            if not math.isfinite(cp): cp=_safe(close.get("market_total")); source="latest_verified_pre_kickoff_snapshot"
            if not math.isfinite(cp): return np.nan,"unavailable"
            return (float(cp-op) if side=="O" else float(op-cp)),source
        if market=="moneyline":
            close_p,source=_close_side_probability(close,side,home); entry_p=_entry_side_probability(entry,side,home)
            if math.isfinite(close_p) and math.isfinite(entry_p): return float(close_p-entry_p),source
    except Exception: pass
    return np.nan,"unavailable"


def _execution_clv_from_market_snapshot(entry,close):
    """Price-aware CLV. For moneylines this compares executed odds to fair close."""
    if close is None: return np.nan
    market=str(entry.get("quant_market") or ""); side=str(entry.get("quant_side") or ""); home=str(entry.get("home_team") or "")
    if market in {"spread","total"}:
        v,_=_clv_from_market_snapshot(entry,close); return v
    if market=="moneyline":
        close_p,_=_close_side_probability(close,side,home); odds=_safe(entry.get("quant_odds"));
        if not math.isfinite(odds): odds=_safe(entry.get("quant_price"))
        if math.isfinite(close_p) and math.isfinite(odds): return float(close_p-american_implied(odds))
    return np.nan


def _bootstrap_roi(profits,seed=2026,n=4000):
    x=pd.to_numeric(pd.Series(profits),errors="coerce").dropna().to_numpy(float)
    if len(x)<30: return [None,None]
    rng=np.random.default_rng(seed); sims=np.empty(n)
    for i in range(n): sims[i]=rng.choice(x,size=len(x),replace=True).mean()
    lo,hi=np.quantile(sims,[.025,.975]); return [float(lo),float(hi)]


def _max_drawdown(profits):
    x=pd.to_numeric(pd.Series(profits),errors="coerce").fillna(0).to_numpy(float)
    if not len(x): return 0.
    curve=np.cumsum(x); peak=np.maximum.accumulate(np.r_[0.,curve])[:-1]
    return float(np.max(peak-curve))


def _summary(bets):
    if bets.empty:
        return {"graded_bets":0,"wins":0,"losses":0,"pushes":0,"units":0.0,"roi":None,"win_rate":None,"avg_clv_proxy":None,"positive_clv_rate":None,"clv_samples":0,"avg_execution_clv":None,"positive_execution_clv_rate":None,"execution_clv_samples":0,"roi_ci_95":[None,None],"max_drawdown":0.0}
    p=pd.to_numeric(bets.profit,errors="coerce"); r=pd.to_numeric(bets.result,errors="coerce"); c=pd.to_numeric(bets.clv_proxy,errors="coerce").dropna(); ec=pd.to_numeric(bets.get("execution_clv",pd.Series(index=bets.index,dtype=float)),errors="coerce").dropna(); wins=int((r>0).sum()); losses=int((r<0).sum()); pushes=int((r==0).sum())
    return {"graded_bets":int(len(bets)),"wins":wins,"losses":losses,"pushes":pushes,"units":float(p.sum()),"roi":float(p.mean()),"win_rate":float(wins/max(1,wins+losses)),"avg_clv_proxy":float(c.mean()) if len(c) else None,"positive_clv_rate":float((c>0).mean()) if len(c) else None,"clv_samples":int(len(c)),"avg_execution_clv":float(ec.mean()) if len(ec) else None,"positive_execution_clv_rate":float((ec>0).mean()) if len(ec) else None,"execution_clv_samples":int(len(ec)),"roi_ci_95":_bootstrap_roi(p),"max_drawdown":_max_drawdown(p)}



def _present_text(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _display_market_entry(entry, market):
    """Normalize one displayed market tier into the quant grading contract."""

    tier_col = {
        "moneyline": "ml_badge",
        "spread": "spread_badge",
        "total": "total_badge",
    }[market]
    tier = str(entry.get(tier_col) or "").upper()
    if tier not in TIERS:
        return None

    home = str(entry.get("home_team") or "")
    away = str(entry.get("away_team") or "")
    home_probability = _safe(entry.get("calibrated_home_probability"))
    side = ""
    line = np.nan
    odds = np.nan
    probability = np.nan
    edge = np.nan
    model_ev = np.nan
    book = None
    quote_at = None
    price_verified = False
    quote_time_source = None

    if market == "moneyline":
        side = str(entry.get("ml_team") or "")
        odds = _safe(entry.get("ml_odds"))
        if side not in {home, away} or not math.isfinite(odds):
            return None
        line = odds
        probability = (
            home_probability
            if side == home
            else 1.0 - home_probability
            if math.isfinite(home_probability)
            else np.nan
        )
        edge = _safe(entry.get("ml_edge_pp"))
        model_ev = _safe(entry.get("ml_est_roi"))
        if side == home:
            book = entry.get("ml_book") or entry.get("best_home_ml_book")
            quote_at = entry.get("ml_quote_at") or entry.get("best_home_ml_quote_at")
            quote_time_source = entry.get("ml_quote_time_source") or entry.get("best_home_ml_quote_time_source")
        else:
            book = entry.get("ml_book") or entry.get("best_away_ml_book")
            quote_at = entry.get("ml_quote_at") or entry.get("best_away_ml_quote_at")
            quote_time_source = entry.get("ml_quote_time_source") or entry.get("best_away_ml_quote_time_source")
    elif market == "spread":
        side = str(entry.get("spread_team") or "")
        line = _safe(entry.get("spread_line"))
        if side not in {home, away} or not math.isfinite(line):
            return None
        stored_odds = _safe(entry.get("spread_odds"))
        odds = stored_odds if math.isfinite(stored_odds) else -110.0
        probability = _safe(entry.get("cover_probability"))
        edge = _safe(entry.get("spread_edge_pts"))
        if math.isfinite(probability):
            model_ev = roi(probability, odds)
        if side == home:
            book = entry.get("spread_book") or entry.get("best_home_spread_book")
            quote_at = entry.get("spread_quote_at") or entry.get("best_home_spread_quote_at")
            quote_time_source = entry.get("spread_quote_time_source") or entry.get("best_home_spread_quote_time_source")
        else:
            book = entry.get("spread_book") or entry.get("best_away_spread_book")
            quote_at = entry.get("spread_quote_at") or entry.get("best_away_spread_quote_at")
            quote_time_source = entry.get("spread_quote_time_source") or entry.get("best_away_spread_quote_time_source")
    else:
        side = str(entry.get("total_dir") or "").upper()
        line = _safe(entry.get("market_total"))
        if side not in {"O", "U"} or not math.isfinite(line):
            return None
        stored_odds = _safe(entry.get("total_odds"))
        odds = stored_odds if math.isfinite(stored_odds) else -110.0
        price_verified = math.isfinite(stored_odds)
        probability = _safe(entry.get("total_probability"))
        edge = _safe(entry.get("total_edge_pts"))
        if math.isfinite(probability):
            model_ev = roi(probability, odds)
        if side == "O":
            book = entry.get("total_book") or entry.get("best_over_book")
            quote_at = entry.get("total_quote_at") or entry.get("best_over_quote_at")
            quote_time_source = entry.get("total_quote_time_source") or entry.get("best_over_quote_time_source")
        else:
            book = entry.get("total_book") or entry.get("best_under_book")
            quote_at = entry.get("total_quote_at") or entry.get("best_under_quote_at")
            quote_time_source = entry.get("total_quote_time_source") or entry.get("best_under_quote_time_source")

    price_verified = (
        math.isfinite(_safe(odds))
        and _present_text(book)
        and _present_text(quote_at)
    )

    return pd.Series(
        {
            **entry.to_dict(),
            "tier": tier,
            "quant_market": market,
            "quant_side": side,
            "quant_price": line,
            "quant_odds": odds,
            "quant_book": book,
            "quant_quote_at": quote_at or entry.get("snapshot_at"),
            "quote_time_source": quote_time_source,
            "model_probability": probability,
            "model_edge": edge,
            "model_ev": model_ev,
            "execution_odds": odds,
            "price_verified": bool(price_verified),
        }
    )


def _grade_display_tiers(hist, finals, markets):
    """Grade the exact displayed STRONG/BET/LEAN label for each game/market."""

    rows = []
    badge_columns = {
        "moneyline": "ml_badge",
        "spread": "spread_badge",
        "total": "total_badge",
    }
    for gid, group in hist.groupby(hist.game_id.astype(str), sort=False):
        if gid not in finals:
            continue
        pre = group.sort_values("_ts")
        first = pre.iloc[0]
        hpnt, apnt = finals[gid]
        actual_margin = hpnt - apnt
        actual_total = hpnt + apnt
        kickoff = pd.to_datetime(first.get("date"), utc=True, errors="coerce")
        close = _latest_pre_kickoff_market(markets, gid, kickoff)

        for market in MARKETS:
            column = badge_columns[market]
            if column not in pre.columns:
                continue
            tier_values = pre[column].fillna("").astype(str).str.upper()
            actionable = pre[tier_values.isin(TIERS)]
            if actionable.empty:
                continue
            raw_entry = actionable.iloc[0]
            entry = _display_market_entry(raw_entry, market)
            if entry is None:
                continue

            result = _grade_row(entry, actual_margin, actual_total)
            profit = _profit(result, entry.get("execution_odds"), market)
            clv, clv_source = _clv_from_market_snapshot(entry, close)
            execution_clv = _execution_clv_from_market_snapshot(entry, close)
            expected_tier = expected_display_tier(market, entry)
            tier_consistent = expected_tier == str(entry.get("tier") or "").upper()
            validation_eligible = tier_consistent and bool(entry.get("price_verified"))
            rows.append(
                {
                    "game_id": gid,
                    "season": first.get("season"),
                    "week": first.get("week"),
                    "away_team": first.get("away_team"),
                    "home_team": first.get("home_team"),
                    "market": market,
                    "tier": entry.get("tier"),
                    "side": entry.get("quant_side"),
                    "line": entry.get("quant_price"),
                    "execution_odds": entry.get("execution_odds"),
                    "book": entry.get("quant_book"),
                    "price_verified": entry.get("price_verified"),
                    "expected_tier": expected_tier,
                    "tier_consistent": tier_consistent,
                    "validation_eligible": validation_eligible,
                    "model_probability": entry.get("model_probability"),
                    "model_edge": entry.get("model_edge"),
                    "model_ev": entry.get("model_ev"),
                    "projected_margin_home": first.get("model_margin_home"),
                    "actual_margin_home": actual_margin,
                    "projected_total": first.get("model_total"),
                    "actual_total": actual_total,
                    "result": result,
                    "flat_stake_units": 1.0,
                    "flat_profit": profit,
                    "clv_proxy": clv,
                    "execution_clv": execution_clv,
                    "clv_source": clv_source,
                    "entry_snapshot": raw_entry.get("snapshot_at"),
                    "entry_quote_at": entry.get("quant_quote_at"),
                    "quote_time_source": entry.get("quote_time_source"),
                    "kickoff": first.get("date"),
                }
            )
    return pd.DataFrame(rows)

def grade_prediction_history(client,history_dir="history",reports_dir="reports"):
    """Grade independent live/paper predictions using only stored pre-kickoff state."""
    hdir=Path(history_dir); hp=hdir/"prediction_snapshots_v4.csv"; mp=hdir/"market_snapshots.csv"; reports=Path(reports_dir); reports.mkdir(parents=True,exist_ok=True)
    method="latest timestamp-valid snapshot strictly before kickoff; spread/total use consensus close when available; moneyline uses consensus no-vig probability; execution CLV retains the actual bet price"
    if not hp.exists():
        report={"graded_games":0,**_summary(pd.DataFrame()),"status":"no v4 prediction history yet","clv_method":method}; (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
    hist=pd.read_csv(hp,low_memory=False)
    if hist.empty:
        report={"graded_games":0,**_summary(pd.DataFrame()),"status":"no v4 prediction history yet","clv_method":method}; (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
    hist["_ts"]=pd.to_datetime(hist.get("snapshot_at"),utc=True,errors="coerce"); hist["_kick"]=pd.to_datetime(hist.get("date"),utc=True,errors="coerce"); hist=hist[(hist._ts.isna())|(hist._kick.isna())|(hist._ts<hist._kick)].copy()
    markets=pd.read_csv(mp,low_memory=False) if mp.exists() else pd.DataFrame()

    finals={}
    for season in sorted({int(x) for x in pd.to_numeric(hist.season,errors="coerce").dropna().unique()}):
        try: sf=client.season_frame(season)
        except Exception: continue
        done=sf[sf["completed"].astype(str).str.lower().isin({"true","1","t","yes"})]
        for _,r in done.iterrows():
            try: finals[str(r.game_id)]=(float(r.home_points),float(r.away_points))
            except Exception: pass

    rows=[]
    for gid,g in hist.groupby(hist.game_id.astype(str),sort=False):
        if gid not in finals: continue
        pre=g.sort_values("_ts"); first=pre.iloc[0]; actionable=pre[(pre.get("quant_signal",pd.Series(index=pre.index,dtype=object)).fillna("PASS").astype(str).str.upper()!="PASS") & pre.get("quant_market",pd.Series(index=pre.index,dtype=object)).notna()]
        entry=actionable.iloc[0] if len(actionable) else None; hpnt,apnt=finals[gid]; am=hpnt-apnt; at=hpnt+apnt; kick=pd.to_datetime(first.get("date"),utc=True,errors="coerce")
        result=profit=clv=execution_clv=np.nan; clv_source="none"; signal="PASS"; market=""; side=""; qprice=qodds=np.nan; qbook=None; entry_ts=None; close_books=np.nan; close_ts=None
        if entry is not None:
            signal=str(entry.get("quant_signal") or "PASS").upper(); market=str(entry.get("quant_market") or ""); side=str(entry.get("quant_side") or ""); qprice=entry.get("quant_price"); qodds=entry.get("quant_odds",np.nan); qbook=entry.get("quant_book"); entry_ts=entry.get("snapshot_at"); result=_grade_row(entry,am,at)
            execution_odds=qodds if math.isfinite(_safe(qodds)) else qprice if market=="moneyline" else -110.0; profit=_profit(result,execution_odds,market)
            close=_latest_pre_kickoff_market(markets,gid,kick); clv,clv_source=_clv_from_market_snapshot(entry,close); execution_clv=_execution_clv_from_market_snapshot(entry,close)
            if close is not None: close_books=close.get("market_book_count",np.nan); close_ts=close.get("captured_at")
        rows.append({"game_id":gid,"season":first.get("season"),"week":first.get("week"),"away_team":first.get("away_team"),"home_team":first.get("home_team"),"projected_margin_home":first.get("model_margin_home"),"actual_margin_home":am,"projected_total":first.get("model_total"),"actual_total":at,"quant_signal":signal,"quant_market":market,"quant_side":side,"quant_price":qprice,"quant_odds":qodds,"quant_book":qbook,"quant_edge":entry.get("quant_edge",np.nan) if entry is not None else np.nan,"data_quality_score":entry.get("data_quality_score",np.nan) if entry is not None else np.nan,"context_risk":entry.get("context_risk",np.nan) if entry is not None else np.nan,"risk_multiplier":entry.get("risk_multiplier",np.nan) if entry is not None else np.nan,"market_disagreement":entry.get("market_disagreement",np.nan) if entry is not None else np.nan,"performance_multiplier":entry.get("performance_multiplier",np.nan) if entry is not None else np.nan,"performance_feedback_reason":entry.get("performance_feedback_reason") if entry is not None else None,"result":result,"profit":profit,"clv_proxy":clv,"execution_clv":execution_clv,"clv_source":clv_source,"close_market_book_count":close_books,"close_snapshot_at":close_ts,"first_snapshot":first.get("snapshot_at"),"bet_entry_snapshot":entry_ts,"kickoff":first.get("date")})

    df=pd.DataFrame(rows); df.to_csv(reports/"live_graded_predictions.csv",index=False); bets=df[(df.quant_signal!="PASS") & df.result.notna()].copy() if len(df) else pd.DataFrame()
    if len(bets): bets["edge_bucket"]=bets.get("quant_edge",pd.Series(index=bets.index,dtype=float)).apply(_edge_bucket)
    bets.to_csv(reports/"live_graded_bets.csv",index=False)

    tier_rows=_grade_display_tiers(hist,finals,markets)
    tier_rows.to_csv(reports/"live_graded_tiers.csv",index=False)
    tier_validation=build_tier_performance(tier_rows)
    (reports/"tier_performance.json").write_text(json.dumps(tier_validation,indent=2))
    pd.DataFrame(tier_validation.get("matrix") or []).to_csv(reports/"tier_performance.csv",index=False)
    pd.DataFrame(tier_validation.get("clean_matrix") or []).to_csv(reports/"tier_validation_clean.csv",index=False)

    overall=_summary(bets); by_market={str(k):_summary(v) for k,v in bets.groupby("quant_market")} if len(bets) else {}; by_signal={str(k):_summary(v) for k,v in bets.groupby("quant_signal")} if len(bets) else {}; by_season={str(k):_summary(v) for k,v in bets.groupby("season")} if len(bets) else {}; by_book={str(k):_summary(v) for k,v in bets.dropna(subset=["quant_book"]).groupby("quant_book")} if len(bets) and "quant_book" in bets.columns else {}; by_edge_bucket={str(k):_summary(v) for k,v in bets.groupby("edge_bucket")} if len(bets) and "edge_bucket" in bets.columns else {}
    verified=int((pd.to_numeric(bets.clv_proxy,errors="coerce").notna()).sum()) if len(bets) else 0
    report={"graded_games":int(len(df)),**overall,"by_market":by_market,"by_signal":by_signal,"by_book":by_book,"by_edge_bucket":by_edge_bucket,"by_season":by_season,"tier_validation":tier_validation,"clv_method":method,"verified_close_clv_samples":verified,"status":"live/shadow evidence only; tier validation is display-model quality evidence and does not authorize portfolio stake"}
    (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
