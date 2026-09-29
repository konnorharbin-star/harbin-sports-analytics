from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SportsDataVerseClient
from .ratings import OpponentAdjustedRatings
from .models import train_models, predict_models
from .market import (
    norm_cdf, replica_win_probability, replica_ml_label, replica_spread_label,
    replica_total_label, no_vig, fair_american, roi,
)
from .render import render_html, render_png


def build_predictions(games, frame, bundle):
    if frame.empty: return pd.DataFrame()
    margins, totals = predict_models(bundle, frame)
    by_id={g.game_id:g for g in games}; rows=[]
    for i,r in frame.reset_index(drop=True).iterrows():
        g=by_id[str(r.game_id)]; m,t=float(margins[i]),float(totals[i]); hp=(t+m)/2; ap=(t-m)/2
        winner=g.home_team if m>=0 else g.away_team; replica_p=replica_win_probability(m)
        calibrated_home_p=norm_cdf(m/max(6,bundle["margin_sigma"])); calibrated_winner_p=calibrated_home_p if m>=0 else 1-calibrated_home_p
        ml_odds=g.home_ml if m>=0 else g.away_ml; ml_badge=""; ml_edge=np.nan; ml_roi=np.nan; fair=np.nan
        if ml_odds is not None:
            ml_badge,ml_edge=replica_ml_label(replica_p,ml_odds); ml_roi=roi(calibrated_winner_p,ml_odds); fair=fair_american(calibrated_winner_p)
        spread_team=spread_line=spread_badge=np.nan; spread_edge=np.nan; cover_p=np.nan
        if g.home_spread is not None:
            spread_badge,edge_home=replica_spread_label(m,g.home_spread); spread_edge=abs(edge_home)
            if edge_home>=0: spread_team,spread_line=g.home_team,g.home_spread
            else: spread_team,spread_line=g.away_team,-g.home_spread
            cover_p=norm_cdf(abs(edge_home)/max(6,bundle["margin_sigma"]))
        total_dir=total_badge=np.nan; total_edge=np.nan; total_prob=np.nan
        if g.market_total is not None:
            total_badge,edge=replica_total_label(t,g.market_total); total_edge=abs(edge); total_dir="O" if edge>=0 else "U"; total_prob=norm_cdf(abs(edge)/max(6,bundle["total_sigma"]))
        nv_winner=np.nan; quant_ml_side=np.nan; quant_ml_edge=np.nan
        if g.away_ml is not None and g.home_ml is not None:
            pa,ph=no_vig(g.away_ml,g.home_ml); model_home=calibrated_home_p; edge_h=model_home-ph; edge_a=(1-model_home)-pa
            if edge_h>=edge_a: quant_ml_side=g.home_team; quant_ml_edge=100*edge_h
            else: quant_ml_side=g.away_team; quant_ml_edge=100*edge_a
            nv_winner=ph if m>=0 else pa
        rows.append({
            "game_id":g.game_id,"season":g.season,"week":g.week,"date":g.date,
            "away_team":g.away_team,"home_team":g.home_team,"away_score":int(round(ap)),"home_score":int(round(hp)),
            "away_score_exact":ap,"home_score_exact":hp,"model_margin_home":m,"model_total":t,"winner":winner,
            "win_probability":replica_p,"win_pct":int(round(100*replica_p)),"calibrated_winner_probability":calibrated_winner_p,
            "provider":g.provider,"away_ml":g.away_ml,"home_ml":g.home_ml,"ml_team":winner,"ml_odds":ml_odds,
            "ml_badge":ml_badge,"ml_edge_pp":ml_edge,"ml_est_roi":ml_roi,"ml_fair_odds":fair,"no_vig_market_prob_winner":nv_winner,
            "quant_best_ml_side":quant_ml_side,"quant_best_ml_edge_pp":quant_ml_edge,
            "market_spread_home":g.home_spread,"spread_team":spread_team,"spread_line":spread_line,"spread_badge":spread_badge,"spread_edge_pts":spread_edge,"cover_probability":cover_p,
            "market_total":g.market_total,"total_dir":total_dir,"proj_total":int(round(t)),"total_badge":total_badge,"total_edge_pts":total_edge,"total_probability":total_prob,
            "baseline_margin":float(r.baseline_margin),"baseline_total":float(r.baseline_total),"elo_diff_home":float(r.elo_diff_home),"net_eff_diff_home":float(r.net_eff_diff_home),
        })
    return pd.DataFrame(rows).sort_values(["date","game_id"]).reset_index(drop=True)


def run_week(season=None, week=None, history_start=None, root="."):
    root=Path(root); out=root/"outputs"; docs=root/"docs"; hist=root/"history"
    for p in (out,docs,hist): p.mkdir(parents=True,exist_ok=True)
    client=SportsDataVerseClient()
    if season is None or week is None:
        ds,dw=client.detect(); season=season or ds; week=week or dw
    history_start=history_start or max(2018,season-5)
    history=client.history(history_start,season,week)
    ratings=OpponentAdjustedRatings(); train_df=ratings.training_frame(history); bundle=train_models(train_df)
    games=[g for g in client.week(season,week) if not g.completed]
    frame=ratings.upcoming_frame(games); pred=build_predictions(games,frame,bundle)
    stamp=datetime.now(timezone.utc).astimezone().isoformat(); date=stamp[:10]; base=f"cfb_model_{season}_week{week}"
    pred.to_csv(out/f"{base}.csv",index=False); (out/f"{base}.json").write_text(pred.to_json(orient="records",indent=2))
    meta={"model_version":"2.0.0","season":season,"week":week,"history_start":history_start,"historical_games":len(history),"training_rows":len(train_df),"upcoming_games":len(games),"metrics":bundle["metrics"],"replica_margin_sigma":16.41,"calibrated_margin_sigma":bundle["margin_sigma"],"calibrated_total_sigma":bundle["total_sigma"],"generated_at":stamp,"source":"sportsdataverse/cfbfastR-data","odds_columns_detected":client.odds_columns}
    (out/f"{base}_metadata.json").write_text(json.dumps(meta,indent=2))
    render_html(pred,out/f"{base}.html",week,date); pages=max(1,math.ceil(len(pred)/14))
    for p in range(1,pages+1): render_png(pred,out/f"{base}_page{p}.png",p,week,date)
    render_html(pred,docs/"index.html",week,date); (docs/"latest.json").write_text(pred.to_json(orient="records",indent=2)); (docs/"metadata.json").write_text(json.dumps(meta,indent=2)); (docs/".nojekyll").write_text("")
    if len(pred):
        snap=pred.copy(); snap.insert(0,"snapshot_at",stamp); hp=hist/"prediction_snapshots_v2.csv"; old=pd.read_csv(hp) if hp.exists() else pd.DataFrame(); pd.concat([old,snap],ignore_index=True).to_csv(hp,index=False)
    mp=hist/"run_metadata_v2.jsonl"; mp.write_text((mp.read_text() if mp.exists() else "")+json.dumps(meta,separators=(",",":"))+"\n")
    return pred,meta
