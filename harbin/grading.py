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


def _profit(result,price,market):
    if pd.isna(result): return np.nan
    if result==0: return 0.
    if result<0: return -1.
    if market!="moneyline": return 100/110
    o=float(price); return 100/abs(o) if o<0 else o/100


def _latest_pre_kickoff_market(snapshot_df,gid,kickoff):
    if snapshot_df.empty or "game_id" not in snapshot_df.columns: return None
    z=snapshot_df[snapshot_df.game_id.astype(str)==str(gid)].copy()
    if z.empty: return None
    if "captured_at" in z:
        z["_ts"]=pd.to_datetime(z.captured_at,utc=True,errors="coerce")
        if pd.notna(kickoff): z=z[z._ts<kickoff]
        z=z.sort_values("_ts")
    return z.iloc[-1] if len(z) else None


def _clv_from_market_snapshot(entry,close):
    if close is None: return np.nan,"none"
    market=str(entry.get("quant_market") or ""); side=str(entry.get("quant_side") or ""); home=str(entry.get("home_team") or "")
    try:
        if market=="spread":
            op=float(entry.quant_price); ch=float(close.home_spread); cp=ch if side==home else -ch
            return float(op-cp),"latest_verified_pre_kickoff_snapshot"
        if market=="total":
            op=float(entry.quant_price); cp=float(close.market_total)
            return (float(cp-op) if side=="O" else float(op-cp)),"latest_verified_pre_kickoff_snapshot"
        if market=="moneyline":
            # Prefer no-vig probability movement when both sides are available; this
            # avoids rewarding a bet merely because bookmaker margin changed.
            ah,aa=_safe(close.get("home_ml")),_safe(close.get("away_ml"))
            if math.isfinite(ah) and math.isfinite(aa):
                pa,ph=no_vig(aa,ah)
                close_p=ph if side==home else pa
                entry_home=_safe(entry.get("home_ml")); entry_away=_safe(entry.get("away_ml"))
                if math.isfinite(entry_home) and math.isfinite(entry_away):
                    epa,eph=no_vig(entry_away,entry_home); entry_p=eph if side==home else epa
                else: entry_p=american_implied(float(entry.quant_price))
                return float(close_p-entry_p),"latest_verified_pre_kickoff_snapshot"
            col="home_ml" if side==home else "away_ml"; cp=float(close[col]); return float(american_implied(cp)-american_implied(float(entry.quant_price))),"latest_verified_pre_kickoff_snapshot"
    except Exception: pass
    return np.nan,"unavailable"


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
        return {"graded_bets":0,"wins":0,"losses":0,"pushes":0,"units":0.0,"roi":None,"win_rate":None,"avg_clv_proxy":None,"positive_clv_rate":None,"clv_samples":0,"roi_ci_95":[None,None],"max_drawdown":0.0}
    p=pd.to_numeric(bets.profit,errors="coerce"); r=pd.to_numeric(bets.result,errors="coerce"); c=pd.to_numeric(bets.clv_proxy,errors="coerce").dropna(); wins=int((r>0).sum()); losses=int((r<0).sum()); pushes=int((r==0).sum())
    return {"graded_bets":int(len(bets)),"wins":wins,"losses":losses,"pushes":pushes,"units":float(p.sum()),"roi":float(p.mean()),"win_rate":float(wins/max(1,wins+losses)),"avg_clv_proxy":float(c.mean()) if len(c) else None,"positive_clv_rate":float((c>0).mean()) if len(c) else None,"clv_samples":int(len(c)),"roi_ci_95":_bootstrap_roi(p),"max_drawdown":_max_drawdown(p)}


def grade_prediction_history(client,history_dir="history",reports_dir="reports"):
    """Grade independent live/paper predictions using only information stored pre-kickoff.

    Entry is the earliest actionable (`quant_signal != PASS`) prediction snapshot for a
    game. CLV proxy comes from the latest hourly market snapshot strictly before kickoff.
    Prediction snapshots are never used as the preferred close. This report is rebuilt
    deterministically on each run, so corrections to final scores or line history flow
    through without duplicating bets.
    """
    hdir=Path(history_dir); hp=hdir/"prediction_snapshots_v4.csv"; mp=hdir/"market_snapshots.csv"; reports=Path(reports_dir); reports.mkdir(parents=True,exist_ok=True)
    if not hp.exists():
        report={"graded_games":0,**_summary(pd.DataFrame()),"status":"no v4 prediction history yet","clv_method":"latest hourly market snapshot strictly before kickoff"}; (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
    hist=pd.read_csv(hp,low_memory=False)
    if hist.empty:
        report={"graded_games":0,**_summary(pd.DataFrame()),"status":"no v4 prediction history yet","clv_method":"latest hourly market snapshot strictly before kickoff"}; (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
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
        result=profit=clv=np.nan; clv_source="none"; signal="PASS"; market=""; side=""; qprice=np.nan; entry_ts=None
        if entry is not None:
            signal=str(entry.get("quant_signal") or "PASS").upper(); market=str(entry.get("quant_market") or ""); side=str(entry.get("quant_side") or ""); qprice=entry.get("quant_price"); entry_ts=entry.get("snapshot_at"); result=_grade_row(entry,am,at); profit=_profit(result,qprice,market); close=_latest_pre_kickoff_market(markets,gid,kick); clv,clv_source=_clv_from_market_snapshot(entry,close)
        rows.append({"game_id":gid,"season":first.get("season"),"week":first.get("week"),"away_team":first.get("away_team"),"home_team":first.get("home_team"),"projected_margin_home":first.get("model_margin_home"),"actual_margin_home":am,"projected_total":first.get("model_total"),"actual_total":at,"quant_signal":signal,"quant_market":market,"quant_side":side,"quant_price":qprice,"result":result,"profit":profit,"clv_proxy":clv,"clv_source":clv_source,"first_snapshot":first.get("snapshot_at"),"bet_entry_snapshot":entry_ts,"kickoff":first.get("date")})

    df=pd.DataFrame(rows); df.to_csv(reports/"live_graded_predictions.csv",index=False); bets=df[(df.quant_signal!="PASS") & df.result.notna()].copy() if len(df) else pd.DataFrame(); bets.to_csv(reports/"live_graded_bets.csv",index=False)
    overall=_summary(bets); by_market={str(k):_summary(v) for k,v in bets.groupby("quant_market")} if len(bets) else {}; by_signal={str(k):_summary(v) for k,v in bets.groupby("quant_signal")} if len(bets) else {}; by_season={str(k):_summary(v) for k,v in bets.groupby("season")} if len(bets) else {}
    report={"graded_games":int(len(df)),**overall,"by_market":by_market,"by_signal":by_signal,"by_season":by_season,"clv_method":"latest hourly market snapshot strictly before kickoff; moneyline uses no-vig probability movement when both sides are stored","verified_close_clv_samples":int((bets.clv_source=="latest_verified_pre_kickoff_snapshot").sum()) if len(bets) else 0,"status":"live/shadow evidence only; not historical backtest evidence"}
    (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
