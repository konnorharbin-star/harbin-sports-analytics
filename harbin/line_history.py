from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .market import american_implied


def _read_histories(history_dir: Path) -> pd.DataFrame:
    frames=[]
    for name in ("prediction_snapshots_v4.csv","prediction_snapshots_v3.csv"):
        p=history_dir/name
        if p.exists():
            try:
                df=pd.read_csv(p,low_memory=False)
                if len(df): frames.append(df)
            except Exception: pass
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()


def attach_line_movement(pred: pd.DataFrame, history_dir="history"):
    """Attach first-seen line movement. This is not called CLV."""
    out=pred.copy(); hist=_read_histories(Path(history_dir))
    cols=("opening_home_spread","opening_total","opening_home_ml","opening_away_ml","spread_move_home","total_move","home_ml_implied_move","away_ml_implied_move")
    if out.empty: return out,{"tracked_games":0,"coverage":0.0}
    if hist.empty or "game_id" not in hist.columns:
        for c in cols: out[c]=np.nan
        return out,{"tracked_games":0,"coverage":0.0}
    if "snapshot_at" in hist.columns:
        hist["_ts"]=pd.to_datetime(hist["snapshot_at"],utc=True,errors="coerce"); hist=hist.sort_values("_ts")
    first=hist.groupby(hist["game_id"].astype(str),as_index=False).first(); first=first.set_index(first["game_id"].astype(str),drop=False); hits=0
    for i,row in out.iterrows():
        gid=str(row.game_id)
        if gid not in first.index:
            for c in cols: out.at[i,c]=np.nan
            continue
        hits+=1; f=first.loc[gid]
        vals={"opening_home_spread":f.get("market_spread_home",np.nan),"opening_total":f.get("market_total",np.nan),"opening_home_ml":f.get("home_ml",np.nan),"opening_away_ml":f.get("away_ml",np.nan)}
        for c,v in vals.items(): out.at[i,c]=v
        try: out.at[i,"spread_move_home"]=float(row.market_spread_home)-float(vals["opening_home_spread"])
        except Exception: out.at[i,"spread_move_home"]=np.nan
        try: out.at[i,"total_move"]=float(row.market_total)-float(vals["opening_total"])
        except Exception: out.at[i,"total_move"]=np.nan
        for side in ("home","away"):
            try: out.at[i,f"{side}_ml_implied_move"]=american_implied(row[f"{side}_ml"])-american_implied(vals[f"opening_{side}_ml"])
            except Exception: out.at[i,f"{side}_ml_implied_move"]=np.nan
    return out,{"tracked_games":hits,"coverage":hits/len(out)}
