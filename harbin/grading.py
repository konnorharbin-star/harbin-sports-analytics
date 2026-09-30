from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from .market import american_implied, no_vig


def _safe(v):
    try:
        x=float(v); return x if math.isfinite(x) else np.nan
    except Exception: return np.nan


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
        rows.append({"game_id":gid,"season":first.get("season"),"week":first.get("week"),"away_team":first.get("away_team"),"home_team":first.get("home_team"),"projected_margin_home":first.get("model_margin_home"),"actual_margin_home":am,"projected_total":first.get("model_total"),"actual_total":at,"quant_signal":signal,"quant_market":market,"quant_side":side,"quant_price":qprice,"quant_odds":qodds,"quant_book":qbook,"result":result,"profit":profit,"clv_proxy":clv,"execution_clv":execution_clv,"clv_source":clv_source,"close_market_book_count":close_books,"close_snapshot_at":close_ts,"first_snapshot":first.get("snapshot_at"),"bet_entry_snapshot":entry_ts,"kickoff":first.get("date")})

    df=pd.DataFrame(rows); df.to_csv(reports/"live_graded_predictions.csv",index=False); bets=df[(df.quant_signal!="PASS") & df.result.notna()].copy() if len(df) else pd.DataFrame(); bets.to_csv(reports/"live_graded_bets.csv",index=False)
    overall=_summary(bets); by_market={str(k):_summary(v) for k,v in bets.groupby("quant_market")} if len(bets) else {}; by_signal={str(k):_summary(v) for k,v in bets.groupby("quant_signal")} if len(bets) else {}; by_season={str(k):_summary(v) for k,v in bets.groupby("season")} if len(bets) else {}
    verified=int((pd.to_numeric(bets.clv_proxy,errors="coerce").notna()).sum()) if len(bets) else 0
    report={"graded_games":int(len(df)),**overall,"by_market":by_market,"by_signal":by_signal,"by_season":by_season,"clv_method":method,"verified_close_clv_samples":verified,"status":"live/shadow evidence only; not historical backtest evidence"}
    (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
