from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .market import american_implied


def _read_prediction_histories(history_dir: Path) -> pd.DataFrame:
    frames=[]
    for name in ("prediction_snapshots_v4.csv","prediction_snapshots_v3.csv"):
        p=history_dir/name
        if p.exists():
            try:
                df=pd.read_csv(p,low_memory=False)
                if len(df): frames.append(df)
            except Exception: pass
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()


def _read_market_snapshots(history_dir: Path) -> pd.DataFrame:
    p=history_dir/"market_snapshots.csv"
    if not p.exists(): return pd.DataFrame()
    try:
        df=pd.read_csv(p,low_memory=False)
        if "captured_at" in df:
            df["_ts"]=pd.to_datetime(df["captured_at"],utc=True,errors="coerce")
        if "kickoff" in df:
            df["_kick"]=pd.to_datetime(df["kickoff"],utc=True,errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _implied_move(current, opening):
    try:
        return american_implied(float(current))-american_implied(float(opening))
    except Exception:
        return np.nan


def _float_move(current,opening):
    try: return float(current)-float(opening)
    except Exception: return np.nan


def attach_line_movement(pred: pd.DataFrame, history_dir="history"):
    """Attach opening and latest verified pre-kickoff market state.

    Stage 3 preserves both the primary-book history and the multi-book consensus history.
    Legacy prediction snapshots remain an opening-only fallback and are never labeled as
    a verified close.
    """
    out=pred.copy(); hdir=Path(history_dir); market=_read_market_snapshots(hdir); legacy=_read_prediction_histories(hdir)
    numeric_cols=(
        "opening_home_spread","opening_total","opening_home_ml","opening_away_ml",
        "latest_pre_kickoff_home_spread","latest_pre_kickoff_total","latest_pre_kickoff_home_ml","latest_pre_kickoff_away_ml",
        "opening_consensus_home_spread","opening_consensus_total","opening_consensus_home_novig_probability","opening_market_book_count",
        "latest_pre_kickoff_consensus_home_spread","latest_pre_kickoff_consensus_total","latest_pre_kickoff_consensus_home_novig_probability","latest_pre_kickoff_market_book_count",
        "spread_move_home","total_move","home_ml_implied_move","away_ml_implied_move","consensus_spread_move_home","consensus_total_move","consensus_home_novig_move","market_snapshot_count",
    )
    object_cols=("first_market_snapshot_at","last_pre_kickoff_snapshot_at","line_history_source","latest_pre_kickoff_market_books")
    if out.empty: return out,{"tracked_games":0,"coverage":0.0,"snapshot_rows":int(len(market)),"consensus_close_games":0,"source":"none"}
    for c in numeric_cols: out[c]=np.nan
    for c in object_cols: out[c]=pd.Series([None]*len(out),index=out.index,dtype="object")
    out["line_history_source"]="none"
    hits=0; market_hits=0; legacy_hits=0; consensus_close_games=0

    legacy_first=pd.DataFrame()
    if len(legacy) and "game_id" in legacy.columns:
        if "snapshot_at" in legacy.columns:
            legacy["_ts"]=pd.to_datetime(legacy["snapshot_at"],utc=True,errors="coerce"); legacy=legacy.sort_values("_ts")
        legacy_first=legacy.groupby(legacy["game_id"].astype(str),as_index=False).first(); legacy_first=legacy_first.set_index(legacy_first["game_id"].astype(str),drop=False)

    for i,row in out.iterrows():
        gid=str(row.game_id); used=False
        if len(market) and "game_id" in market.columns:
            z=market[market["game_id"].astype(str)==gid].copy()
            if len(z):
                kick=pd.to_datetime(row.get("date"),utc=True,errors="coerce")
                if "_ts" in z:
                    z=z[z["_ts"].notna()].sort_values("_ts")
                pre=z
                if pd.notna(kick) and "_ts" in z:
                    pre=z[z["_ts"]<kick]
                if len(pre):
                    first=pre.iloc[0]; last=pre.iloc[-1]; hits+=1; market_hits+=1; used=True
                    mapping={
                        "opening_home_spread":"home_spread","opening_total":"market_total","opening_home_ml":"home_ml","opening_away_ml":"away_ml",
                        "latest_pre_kickoff_home_spread":"home_spread","latest_pre_kickoff_total":"market_total","latest_pre_kickoff_home_ml":"home_ml","latest_pre_kickoff_away_ml":"away_ml",
                        "opening_consensus_home_spread":"consensus_home_spread","opening_consensus_total":"consensus_total","opening_consensus_home_novig_probability":"consensus_home_novig_probability","opening_market_book_count":"market_book_count",
                        "latest_pre_kickoff_consensus_home_spread":"consensus_home_spread","latest_pre_kickoff_consensus_total":"consensus_total","latest_pre_kickoff_consensus_home_novig_probability":"consensus_home_novig_probability","latest_pre_kickoff_market_book_count":"market_book_count",
                    }
                    for dst,src in mapping.items():
                        srcrow=first if dst.startswith("opening_") else last
                        out.at[i,dst]=srcrow.get(src,np.nan)
                    out.at[i,"market_snapshot_count"]=int(len(pre))
                    out.at[i,"first_market_snapshot_at"]=first.get("captured_at",None)
                    out.at[i,"last_pre_kickoff_snapshot_at"]=last.get("captured_at",None)
                    out.at[i,"latest_pre_kickoff_market_books"]=last.get("market_books",None)
                    out.at[i,"line_history_source"]="hourly_market_snapshots"
                    consensus_close_games+=int(pd.notna(last.get("consensus_home_spread",np.nan)) or pd.notna(last.get("consensus_total",np.nan)) or pd.notna(last.get("consensus_home_novig_probability",np.nan)))
        if not used and len(legacy_first) and gid in legacy_first.index:
            f=legacy_first.loc[gid]; hits+=1; legacy_hits+=1
            out.at[i,"opening_home_spread"]=f.get("market_spread_home",np.nan); out.at[i,"opening_total"]=f.get("market_total",np.nan)
            out.at[i,"opening_home_ml"]=f.get("home_ml",np.nan); out.at[i,"opening_away_ml"]=f.get("away_ml",np.nan)
            out.at[i,"opening_consensus_home_spread"]=f.get("consensus_home_spread",f.get("market_spread_home",np.nan)); out.at[i,"opening_consensus_total"]=f.get("consensus_total",f.get("market_total",np.nan)); out.at[i,"opening_consensus_home_novig_probability"]=f.get("consensus_home_novig_probability",np.nan); out.at[i,"opening_market_book_count"]=f.get("market_book_count",np.nan)
            out.at[i,"market_snapshot_count"]=1; out.at[i,"first_market_snapshot_at"]=f.get("snapshot_at",None); out.at[i,"line_history_source"]="prediction_snapshot_fallback"
        out.at[i,"spread_move_home"]=_float_move(row.get("market_spread_home"),out.at[i,"opening_home_spread"])
        out.at[i,"total_move"]=_float_move(row.get("market_total"),out.at[i,"opening_total"])
        out.at[i,"home_ml_implied_move"]=_implied_move(row.get("home_ml"),out.at[i,"opening_home_ml"])
        out.at[i,"away_ml_implied_move"]=_implied_move(row.get("away_ml"),out.at[i,"opening_away_ml"])
        out.at[i,"consensus_spread_move_home"]=_float_move(out.at[i,"latest_pre_kickoff_consensus_home_spread"],out.at[i,"opening_consensus_home_spread"])
        out.at[i,"consensus_total_move"]=_float_move(out.at[i,"latest_pre_kickoff_consensus_total"],out.at[i,"opening_consensus_total"])
        out.at[i,"consensus_home_novig_move"]=_float_move(out.at[i,"latest_pre_kickoff_consensus_home_novig_probability"],out.at[i,"opening_consensus_home_novig_probability"])
    return out,{"tracked_games":int(hits),"coverage":hits/len(out),"snapshot_rows":int(len(market)),"hourly_snapshot_games":int(market_hits),"legacy_fallback_games":int(legacy_hits),"consensus_close_games":int(consensus_close_games),"source":"hourly_market_snapshots" if market_hits else "prediction_snapshot_fallback" if legacy_hits else "none"}
