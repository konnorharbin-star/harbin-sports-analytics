from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def _grade_row(r,actual_margin,actual_total):
    market=str(r.get("quant_market") or ""); side=r.get("quant_side")
    if market=="moneyline":
        home=str(r.get("home_team")); away=str(r.get("away_team")); winner=home if actual_margin>0 else away if actual_margin<0 else None
        return 1 if winner and str(side)==winner else 0 if winner is None else -1
    if market=="spread":
        home=str(r.get("home_team")); line=float(r.get("quant_price")); val=actual_margin+line if str(side)==home else -actual_margin+line; return 1 if val>1e-9 else -1 if val<-1e-9 else 0
    if market=="total":
        line=float(r.get("quant_price")); val=actual_total-line; val=-val if str(side)=="U" else val; return 1 if val>1e-9 else -1 if val<-1e-9 else 0
    return np.nan


def _profit(result,price,market):
    if pd.isna(result): return np.nan
    if result==0: return 0.;
    if result<0: return -1.
    if market!="moneyline": return 100/110
    o=float(price); return 100/abs(o) if o<0 else o/100


def grade_prediction_history(client,history_dir="history",reports_dir="reports"):
    hp=Path(history_dir)/"prediction_snapshots_v4.csv"; reports=Path(reports_dir); reports.mkdir(parents=True,exist_ok=True)
    if not hp.exists(): return {"graded_games":0,"graded_bets":0,"status":"no v4 prediction history yet"}
    hist=pd.read_csv(hp,low_memory=False)
    if hist.empty: return {"graded_games":0,"graded_bets":0,"status":"no v4 prediction history yet"}
    hist["_ts"]=pd.to_datetime(hist["snapshot_at"],utc=True,errors="coerce"); hist["_kick"]=pd.to_datetime(hist["date"],utc=True,errors="coerce"); hist=hist[(hist["_ts"].isna())|(hist["_kick"].isna())|(hist["_ts"]<hist["_kick"])]
    finals={}
    for season in sorted({int(x) for x in hist.season.dropna().unique()}):
        try: sf=client.season_frame(season)
        except Exception: continue
        done=sf[sf["completed"].astype(str).str.lower().isin({"true","1","t","yes"})]
        for _,r in done.iterrows():
            try: finals[str(r["game_id"])]=(float(r["home_points"]),float(r["away_points"]))
            except Exception: pass
    rows=[]
    for gid,g in hist.groupby(hist.game_id.astype(str),sort=False):
        if gid not in finals: continue
        pre=g.sort_values("_ts"); first=pre.iloc[0]; close=pre.iloc[-1]; hpnt,apnt=finals[gid]; am=hpnt-apnt; at=hpnt+apnt; signal=str(first.get("quant_signal") or "PASS"); result=profit=np.nan
        if signal!="PASS" and pd.notna(first.get("quant_market")):
            result=_grade_row(first,am,at); profit=_profit(result,first.get("quant_price"),str(first.get("quant_market")))
        clv=np.nan; market=str(first.get("quant_market") or ""); side=str(first.get("quant_side") or "")
        try:
            if market=="spread":
                home=str(first.home_team); open_side=float(first.quant_price); close_home=float(close.market_spread_home); close_side=close_home if side==home else -close_home; clv=open_side-close_side
            elif market=="total":
                op=float(first.quant_price); cp=float(close.market_total); clv=cp-op if side=="O" else op-cp
            elif market=="moneyline":
                from .market import american_implied
                side_col="home_ml" if side==str(first.home_team) else "away_ml"; clv=american_implied(close[side_col])-american_implied(first.quant_price)
        except Exception: pass
        rows.append({"game_id":gid,"season":first.get("season"),"week":first.get("week"),"away_team":first.get("away_team"),"home_team":first.get("home_team"),"projected_margin_home":first.get("model_margin_home"),"actual_margin_home":am,"projected_total":first.get("model_total"),"actual_total":at,"quant_signal":signal,"quant_market":market,"quant_side":side,"quant_price":first.get("quant_price"),"result":result,"profit":profit,"clv_proxy":clv,"first_snapshot":first.get("snapshot_at"),"close_snapshot":close.get("snapshot_at")})
    df=pd.DataFrame(rows); df.to_csv(reports/"live_graded_predictions.csv",index=False); bets=df[df.quant_signal!="PASS"] if len(df) else pd.DataFrame()
    report={"graded_games":int(len(df)),"graded_bets":int(len(bets)),"units":float(pd.to_numeric(bets.profit,errors="coerce").sum()) if len(bets) else 0.,"roi":float(pd.to_numeric(bets.profit,errors="coerce").mean()) if len(bets) else None,"win_rate":float((bets.result>0).sum()/max(1,int((bets.result!=0).sum()))) if len(bets) else None,"avg_clv_proxy":float(pd.to_numeric(bets.clv_proxy,errors="coerce").dropna().mean()) if len(bets) and bets.clv_proxy.notna().any() else None,"note":"CLV proxy uses the latest stored pre-kickoff snapshot, not a guaranteed official closing line."}
    (reports/"live_performance.json").write_text(json.dumps(report,indent=2)); return report
