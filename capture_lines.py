#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from harbin.data import SportsDataVerseClient
from harbin.market_intel import MarketIntelligence


def _same(a,b):
    keys=("home_ml","away_ml","home_spread","market_total","consensus_home_spread","consensus_total","best_home_ml","best_away_ml","market_book_count")
    for k in keys:
        x,y=a.get(k),b.get(k)
        try:
            if pd.isna(x) and pd.isna(y): continue
            if abs(float(x)-float(y))>1e-9: return False
        except Exception:
            if str(x)!=str(y): return False
    return True

def build_snapshot_rows(games, intel_df, captured_at=None):
    captured_at=captured_at or datetime.now(timezone.utc)
    by={str(r.game_id):r for _,r in intel_df.iterrows()} if len(intel_df) else {}
    rows=[]
    for g in games:
        r=by.get(str(g.game_id)); kickoff=pd.to_datetime(g.date,utc=True,errors="coerce")
        mins=(kickoff-captured_at).total_seconds()/60 if pd.notna(kickoff) else None
        def val(name,default=None):
            if r is not None and name in r and pd.notna(r[name]): return r[name]
            return default
        rows.append({
            "captured_at":captured_at.isoformat(),"game_id":str(g.game_id),"season":int(g.season),"week":int(g.week),"kickoff":g.date,
            "away_team":g.away_team,"home_team":g.home_team,"provider":g.provider,
            "away_ml":g.away_ml,"home_ml":g.home_ml,"home_spread":g.home_spread,"market_total":g.market_total,
            "consensus_home_spread":val("consensus_home_spread"),"consensus_total":val("consensus_total"),
            "best_home_ml":val("best_home_ml"),"best_away_ml":val("best_away_ml"),"market_book_count":val("market_book_count",1),
            "market_books":val("market_books",g.provider or "primary"),"minutes_to_kickoff":mins,
        })
    return rows

def append_snapshots(rows,path="history/market_snapshots.csv"):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); new=pd.DataFrame(rows)
    if new.empty: return 0
    old=pd.read_csv(p,low_memory=False) if p.exists() else pd.DataFrame()
    keep=[]; now=pd.Timestamp.now(tz="UTC")
    if len(old) and "captured_at" in old:
        old["_ts"]=pd.to_datetime(old.captured_at,utc=True,errors="coerce")
    for _,row in new.iterrows():
        gid=str(row.game_id); prior=old[old.game_id.astype(str)==gid] if len(old) and "game_id" in old else pd.DataFrame()
        if prior.empty:
            keep.append(row); continue
        last=prior.sort_values("_ts").iloc[-1]
        same=_same(row,last)
        age_h=(now-last["_ts"]).total_seconds()/3600 if pd.notna(last.get("_ts")) else 99
        mins=row.get("minutes_to_kickoff")
        # Near kickoff, keep every hourly observation even when unchanged. Farther out,
        # keep a heartbeat every four hours plus every actual price/line change.
        near=mins is not None and pd.notna(mins) and -30<=float(mins)<=360
        if (not same) or near or age_h>=4:
            keep.append(row)
    if not keep: return 0
    add=pd.DataFrame(keep); combined=pd.concat([old.drop(columns=["_ts"],errors="ignore"),add],ignore_index=True,sort=False)
    combined.to_csv(p,index=False); return len(add)

def main():
    ap=argparse.ArgumentParser(description="Capture lightweight CFB market snapshots without retraining the model")
    ap.add_argument("--season",type=int); ap.add_argument("--week",type=int); a=ap.parse_args()
    client=SportsDataVerseClient()
    if a.season is None or a.week is None:
        s,w=client.detect(); season=a.season or s; week=a.week or w
    else: season,week=a.season,a.week
    games=[g for g in client.week(season,week) if not g.completed]
    base=pd.DataFrame([{"game_id":str(g.game_id),"away_team":g.away_team,"home_team":g.home_team} for g in games])
    intel=MarketIntelligence(); enriched,intel_meta=intel.attach(games,base) if len(base) else (base,{"coverage":0,"multi_book_coverage":0,"errors":[]})
    rows=build_snapshot_rows(games,enriched); added=append_snapshots(rows)
    status={"captured_at":datetime.now(timezone.utc).isoformat(),"season":int(season),"week":int(week),"games":len(games),"rows_appended":int(added),"odds_source":client.odds_source,"market_intelligence":intel_meta,"errors":client.odds_errors+intel_meta.get("errors",[])}
    out=Path("outputs"); out.mkdir(exist_ok=True); (out/"line_capture_status.json").write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
if __name__=="__main__": main()
