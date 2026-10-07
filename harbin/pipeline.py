from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from .advanced import AdvancedFeatureStore
from .context import ContextStore
from .data import SportsDataVerseClient
from .grading import grade_prediction_history
from .health import build_health_report
from .line_history import attach_line_movement
from .market import norm_cdf, conditional_margin_sigma, conditional_total_sigma, replica_win_probability, replica_ml_label, replica_spread_label, replica_total_label, no_vig, fair_american, roi
from .market_intel import MarketIntelligence
from .models import train_models, predict_models, predict_home_probabilities
from .pro_market import select_best_market, risk_multiplier
from .ratings import OpponentAdjustedRatings
from .render import render_html, render_png
from .tier_validation import refresh_display_market_tiers

MODEL_VERSION="7.1.0"

def _finite(v):
    try: return math.isfinite(float(v))
    except Exception: return False


def build_predictions(games,frame,bundle):
    if frame.empty: return pd.DataFrame()
    margins,totals=predict_models(bundle,frame); home_probs=predict_home_probabilities(bundle,margins); by_id={str(g.game_id):g for g in games}; rows=[]
    for i,r in frame.reset_index(drop=True).iterrows():
        g=by_id[str(r.game_id)]; m,t=float(margins[i]),float(totals[i]); hp,ap=(t+m)/2,(t-m)/2; rp=replica_win_probability(m); sm=max(6,float(bundle["margin_sigma"])); st=max(6,float(bundle["total_sigma"])); ph=float(home_probs[i]); winner=g.home_team if ph>=.5 else g.away_team; pw=max(ph,1-ph); ml=g.home_ml if winner==g.home_team else g.away_ml; mtag=""; medge=mroi=np.nan; fair=fair_american(pw)
        if ml is not None:
            # Keep display confidence aligned with the calibrated probability/EV engine.
            implied=abs(float(ml))/(abs(float(ml))+100) if float(ml)<0 else 100/(float(ml)+100)
            medge=100*(pw-implied)
            mroi=roi(pw,ml)
            mtag=replica_ml_label(pw,ml)[0] if mroi>0 else ""
        spteam=spline=np.nan; stag=""; sedge=cp=np.nan
        if g.home_spread is not None:
            stag,eh=replica_spread_label(m,g.home_spread); sedge=abs(eh); spteam,spline=(g.home_team,float(g.home_spread)) if eh>=0 else (g.away_team,-float(g.home_spread)); cp=norm_cdf(abs(eh)/conditional_margin_sigma(sm,m))
        tdir=np.nan; ttag=""; tedge=tp=np.nan
        if g.market_total is not None:
            ttag,et=replica_total_label(t,g.market_total); tedge=abs(et); tdir="O" if et>=0 else "U"; tp=norm_cdf(abs(et)/conditional_total_sigma(st,t))
        nvw=qside=qedge=qroi=np.nan
        if g.away_ml is not None and g.home_ml is not None:
            pa,pmh=no_vig(g.away_ml,g.home_ml); eh=ph-pmh; ea=(1-ph)-pa
            if eh>=ea: qside=g.home_team; qedge=100*eh; qroi=roi(ph,g.home_ml)
            else: qside=g.away_team; qedge=100*ea; qroi=roi(1-ph,g.away_ml)
            nvw=pmh if m>=0 else pa
        rows.append({"game_id":g.game_id,"season":g.season,"week":g.week,"date":g.date,"away_team":g.away_team,"home_team":g.home_team,"away_score":int(round(ap)),"home_score":int(round(hp)),"away_score_exact":ap,"home_score_exact":hp,"model_margin_home":m,"model_total":t,"fair_spread_home":-m,"fair_total":t,"winner":winner,"win_probability":pw,"win_pct":int(round(100*pw)),"calibrated_home_probability":ph,"calibrated_winner_probability":pw,"provider":g.provider,"market_available":any(x is not None for x in (g.home_ml,g.away_ml,g.home_spread,g.market_total)),"away_ml":g.away_ml,"home_ml":g.home_ml,"ml_team":winner,"ml_odds":ml,"ml_badge":mtag,"ml_edge_pp":medge,"ml_est_roi":mroi,"ml_fair_odds":fair,"no_vig_market_prob_winner":nvw,"quant_best_ml_side":qside,"quant_best_ml_edge_pp":qedge,"quant_best_ml_roi":qroi,"market_spread_home":g.home_spread,"spread_team":spteam,"spread_line":spline,"spread_badge":stag,"spread_edge_pts":sedge,"cover_probability":cp,"market_total":g.market_total,"total_dir":tdir,"proj_total":int(round(t)),"total_badge":ttag,"total_edge_pts":tedge,"total_probability":tp,"baseline_margin":float(r.baseline_margin),"baseline_total":float(r.baseline_total),"elo_diff_home":float(r.elo_diff_home),"net_eff_diff_home":float(r.net_eff_diff_home),"volatility_avg":float(r.get("volatility_avg",np.nan))})
    return pd.DataFrame(rows).sort_values(["date","game_id"]).reset_index(drop=True)


def _coverage(pred):
    if pred.empty: return {"games":0,"any_market":0,"moneyline":0,"spread":0,"total":0}
    return {"games":int(len(pred)),"any_market":int(pred.market_available.fillna(False).astype(bool).sum()),"moneyline":int((pred.away_ml.notna()&pred.home_ml.notna()).sum()),"spread":int(pred.market_spread_home.notna().sum()),"total":int(pred.market_total.notna().sum())}

def _market_status(c,source):
    if c["games"]==0: return "No upcoming games were returned for this season/week."
    if c["any_market"]==0: return "PROJECTION-ONLY MODE — no verified sportsbook market data was available. No betting signals are shown."
    if c["any_market"]<c["games"]: return f"PARTIAL MARKET DATA — {c['any_market']}/{c['games']} games have at least one verified market from {source or 'available sources'}."
    return f"LIVE MARKET DATA — all {c['games']} games have verified market data from {source or 'available sources'}."
def _display_time(dt):
    x=dt.astimezone(ZoneInfo("America/Chicago"))
    try: return x.strftime("%b %-d, %Y · %-I:%M %p CT")
    except ValueError: return x.strftime("%b %d, %Y · %I:%M %p CT").replace(" 0"," ")
def _prediction_signature(pred):
    if pred.empty: return ""
    cols=["game_id","model_margin_home","model_total","away_ml","home_ml","market_spread_home","market_total","provider","ml_badge","ml_odds","ml_book","ml_quote_at","ml_quote_time_source","spread_badge","spread_team","spread_line","spread_odds","spread_book","spread_quote_at","spread_quote_time_source","total_badge","total_dir","total_odds","total_book","total_quote_at","total_quote_time_source","quant_signal","quant_market","quant_side","quant_price","quant_odds"]; temp=pred[[c for c in cols if c in pred.columns]].copy()
    for c in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[c]): temp[c]=temp[c].round(6)
    return temp.to_json(orient="records")
def _append_snapshot_if_changed(pred,path,stamp):
    if pred.empty: return False
    sig=_prediction_signature(pred); sp=path.with_suffix(".signature"); oldsig=sp.read_text() if sp.exists() else None
    if oldsig==sig: return False
    snap=pred.copy(); snap.insert(0,"snapshot_at",stamp); old=pd.read_csv(path,low_memory=False) if path.exists() else pd.DataFrame(); pd.concat([old,snap],ignore_index=True,sort=False).to_csv(path,index=False); sp.write_text(sig); return True


def _quantize(pred,bundle,adv_meta,ctx_meta):
    if pred.empty: return pred
    out=pred.copy(); ac=float(adv_meta.get("dynamic_coverage",adv_meta.get("coverage",0)) or 0); ca=1.0 if ctx_meta.get("sources") else .5; sm=max(6,float(bundle["margin_sigma"])); st=max(6,float(bundle["total_sigma"]))
    for i,r in out.iterrows():
        q=r.copy()
        if _finite(r.get("consensus_home_novig_probability")) and _finite(r.get("best_home_ml")) and _finite(r.get("best_away_ml")):
            ph=float(r.calibrated_home_probability); mh=float(r.consensus_home_novig_probability); eh,ea=ph-mh,(1-ph)-(1-mh)
            if eh>=ea: q["quant_best_ml_side"]=r.home_team; q["quant_best_ml_edge_pp"]=100*eh; q["quant_best_ml_roi"]=roi(ph,r.best_home_ml); q["home_ml"]=r.best_home_ml
            else: q["quant_best_ml_side"]=r.away_team; q["quant_best_ml_edge_pp"]=100*ea; q["quant_best_ml_roi"]=roi(1-ph,r.best_away_ml); q["away_ml"]=r.best_away_ml

        if _finite(r.get("consensus_home_spread")):
            consensus=float(r.consensus_home_spread); consensus_edge=float(r.model_margin_home)+consensus
            if consensus_edge>=0 and _finite(r.get("best_home_spread")) and _finite(r.get("best_home_spread_odds")):
                line=float(r.best_home_spread); edge=float(r.model_margin_home)+line
                q["spread_team"],q["spread_line"]=r.home_team,line
                q["spread_odds"]=float(r.best_home_spread_odds)
                q["spread_edge_pts"]=max(0.0,edge); q["cover_probability"]=norm_cdf(max(0.0,edge)/conditional_margin_sigma(sm,r.model_margin_home))
            elif consensus_edge<0 and _finite(r.get("best_away_spread")) and _finite(r.get("best_away_spread_odds")):
                line=float(r.best_away_spread); edge=line-float(r.model_margin_home)
                q["spread_team"],q["spread_line"]=r.away_team,line
                q["spread_odds"]=float(r.best_away_spread_odds)
                q["spread_edge_pts"]=max(0.0,edge); q["cover_probability"]=norm_cdf(max(0.0,edge)/conditional_margin_sigma(sm,r.model_margin_home))

        if _finite(r.get("consensus_total")):
            consensus=float(r.consensus_total); consensus_edge=float(r.model_total)-consensus
            if consensus_edge>=0 and _finite(r.get("best_over_total")) and _finite(r.get("best_over_odds")):
                line=float(r.best_over_total); edge=float(r.model_total)-line
                q["total_dir"]="O"; q["total_odds"]=float(r.best_over_odds)
                q["market_total"]=line; q["total_edge_pts"]=max(0.0,edge); q["total_probability"]=norm_cdf(max(0.0,edge)/conditional_total_sigma(st,r.model_total))
            elif consensus_edge<0 and _finite(r.get("best_under_total")) and _finite(r.get("best_under_odds")):
                line=float(r.best_under_total); edge=line-float(r.model_total)
                q["total_dir"]="U"; q["total_odds"]=float(r.best_under_odds)
                q["market_total"]=line; q["total_edge_pts"]=max(0.0,edge); q["total_probability"]=norm_cdf(max(0.0,edge)/conditional_total_sigma(st,r.model_total))

        # The Cooper-style badge must describe the line actually displayed after
        # line-shopping, not the stale primary-provider line used during base build.
        for key,value in refresh_display_market_tiers(q).items():
            q[key]=value

        mq=float(r.get("market_consensus_quality",.25) or .25); outlier_penalty=max(.65,1-.05*float(r.get("market_outlier_rejected_count",0) or 0)); dq=max(.15,min(1,(.58*ac+.27*mq+.15*ca)*outlier_penalty)); cr=float(r.get("context_risk",r.get("availability_risk",0)) or 0); vol=float(r.get("volatility_avg",14)) if _finite(r.get("volatility_avg")) else 14.; rm=risk_multiplier(cr,dq,vol); disagreement=0.
        if _finite(r.get("consensus_home_spread")): disagreement=max(disagreement,abs(float(r.model_margin_home)+float(r.consensus_home_spread)))
        if _finite(r.get("consensus_total")): disagreement=max(disagreement,abs(float(r.model_total)-float(r.consensus_total)))
        if disagreement>10: rm*=max(.5,1-.035*(disagreement-10))
        pick=select_best_market(q,rm); out.at[i,"data_quality_score"]=dq; out.at[i,"risk_multiplier"]=rm; out.at[i,"market_disagreement"]=disagreement
        for k,v in q.items():
            if k in {"ml_odds","ml_badge","ml_edge_pp","ml_est_roi","ml_book","ml_quote_at","ml_quote_time_source","spread_team","spread_line","spread_odds","spread_badge","spread_book","spread_quote_at","spread_quote_time_source","spread_edge_pts","cover_probability","total_dir","market_total","total_odds","total_badge","total_book","total_quote_at","total_quote_time_source","total_edge_pts","total_probability","home_ml","away_ml","quant_best_ml_side","quant_best_ml_edge_pp","quant_best_ml_roi"}: out.at[i,k]=v
        for k,v in pick.items(): out.at[i,k]=v
        if dq<.45 or float(pick.get("stake_units",0) or 0)<.08: out.at[i,"quant_signal"]="PASS"; out.at[i,"stake_units"]=0.
    return out


def _write_quant_html(pred,path,updated):
    rank={"STRONG":0,"BET":1,"LEAN":2,"PASS":3}; q=pred.copy(); q["_rank"]=q.get("quant_signal",pd.Series("PASS",index=q.index)).map(rank).fillna(3); q=q.sort_values(["_rank","quant_ev"],ascending=[True,False]); rows=[]
    for _,r in q.iterrows():
        ev=float(r.quant_ev) if _finite(r.get("quant_ev")) else 0.; prob=float(r.quant_probability) if _finite(r.get("quant_probability")) else 0.; price=r.get("quant_price"); odds=r.get("quant_odds"); rows.append(f"<tr><td>{r.away_team} @ {r.home_team}</td><td>{r.get('quant_signal','PASS')}</td><td>{r.get('quant_market') or '—'}</td><td>{r.get('quant_side') or '—'}</td><td>{price if pd.notna(price) else '—'}</td><td>{odds if pd.notna(odds) else '—'}</td><td>{prob:.1%}</td><td>{ev:.1%}</td><td>{float(r.get('stake_units',0) or 0):.2f}</td><td>{float(r.get('data_quality_score',0) or 0):.0%}</td></tr>")
    Path(path).write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Harbin Quant Card</title><style>body{{background:#0f1113;color:#f2f2f2;font-family:Arial;margin:0;padding:24px}}.wrap{{max-width:1200px;margin:auto}}p{{color:#9da1a6}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #292c30;text-align:left}}th{{color:#888;font-size:11px}}tr:nth-child(even){{background:#17191c}}</style></head><body><div class='wrap'><h1>HARBIN QUANT CARD</h1><p>Updated {updated}. Best verified executable line/price, positive-EV gates and portfolio risk controls; PASS is normal. No result guarantees future profitability.</p><table><thead><tr><th>GAME</th><th>SIGNAL</th><th>MARKET</th><th>SIDE</th><th>LINE</th><th>ODDS</th><th>MODEL P</th><th>EV</th><th>UNITS</th><th>DATA QUALITY</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></body></html>")


def _write_output_readme(out,base,meta,pages,run_tag):
    pngs="\n".join(
        f"- [Fresh page {i} — cache-safe]({base}_run_{run_tag}_page{i}.png)"
        for i in range(1,pages+1)
    )
    stable_pngs="\n".join(
        f"- [Stable page {i}]({base}_page{i}.png)"
        for i in range(1,pages+1)
    )
    h=meta.get("health",{}); ac=meta.get("advanced_features",{}).get("dynamic_coverage",meta.get("advanced_features",{}).get("live_coverage",0))
    (out/"README.md").write_text(f"# Latest CFB model output\n\n**Model:** v{meta['model_version']}  \n**Season / Week:** {meta['season']} / {meta['week']}  \n**Updated:** {meta['updated_at_ct']}  \n**Market:** {meta['market_status']}  \n**Dynamic advanced-feature live coverage:** {ac:.0%}  \n**System health:** {h.get('system_health_score','—')}/100 *(readiness, not predicted profitability)*\n\n## Use these\n- [Interactive Cooper-style table]({base}.html)\n- [Quant card](quant_card.html)\n- [Quant recommendations CSV](quant_recommendations.csv)\n- [Full model CSV]({base}.csv)\n- [Metadata / diagnostics]({base}_metadata.json)\n- [System health report](system_health.json)\n- [Market × tier forward validation](tier_performance.json)\n- [Posted market × tier matrix CSV](tier_performance.csv)\n- [Clean validation-eligible tier matrix](tier_validation_clean.csv)\n- [Posted flat-1u graded tier ledger](live_graded_tiers.csv)\n- [Clean validation-entry ledger](live_graded_tiers_clean.csv)\n\n## Fresh PNGs for mobile\n{pngs}\n\nThese filenames change on every run so GitHub mobile cannot reuse an old image preview.\n\n## Stable PNG names\n{stable_pngs}\n\nThe Cooper-style table is the reconstructed presentation layer. The Quant card is the independent EV/risk layer. Missing verified markets display **NO LINE**. Run the separate **CFB Backtest** workflow before treating signals as historically established.\n")


def _pregame_games(games, now=None):
    """Return only not-started games; live/in-game markets must never enter pregame picks."""

    now = now or datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    upcoming=[]; started=[]; invalid=[]
    for g in games or []:
        if getattr(g,"completed",False):
            continue
        kickoff=pd.to_datetime(getattr(g,"date",None),utc=True,errors="coerce")
        if pd.isna(kickoff):
            invalid.append(str(getattr(g,"game_id","")))
            continue
        if kickoff <= now_ts:
            started.append(str(getattr(g,"game_id","")))
            continue
        upcoming.append(g)
    return upcoming, {
        "returned_games": int(len(games or [])),
        "upcoming_games": int(len(upcoming)),
        "started_excluded": int(len(started)),
        "invalid_kickoff_excluded": int(len(invalid)),
        "started_game_ids": started,
        "invalid_kickoff_game_ids": invalid,
        "cutoff_utc": now_ts.isoformat(),
    }


def run_week(season=None,week=None,history_start=None,root="."):
    root=Path(root); out,docs,hist,reports=root/"outputs",root/"docs",root/"history",root/"reports"
    for p in (out,docs,hist,reports): p.mkdir(parents=True,exist_ok=True)
    client=SportsDataVerseClient()
    if season is None or week is None:
        ds,dw=client.detect(); season=season or ds; week=week or dw
    now=datetime.now(timezone.utc)
    history_start=history_start or max(2018,season-5); history=client.history(history_start,season,week); ratings=OpponentAdjustedRatings(); base_train=ratings.training_frame(history); advanced=AdvancedFeatureStore(history_start,season,cache_dir=root/"cache"/"advanced"); train_df,adv_train=advanced.enrich(base_train); bundle=train_models(train_df); raw_games=client.week(season,week); games,pregame_meta=_pregame_games(raw_games,now); base_frame=ratings.upcoming_frame(games); frame,adv_live=advanced.enrich(base_frame); pred=build_predictions(games,frame,bundle); intel=MarketIntelligence(); pred,intel_meta=intel.attach(games,pred); context=ContextStore(season,schedule_frame=client.season_frame(season),cache_dir=root/"cache"/"context"); pred,ctx=context.attach(pred); pred,line_meta=attach_line_movement(pred,hist); pred=_quantize(pred,bundle,adv_live,ctx); stamp=now.isoformat(); display=_display_time(now); base=f"cfb_model_{season}_week{week}"; run_tag=now.astimezone(ZoneInfo("America/Chicago")).strftime("%Y%m%d_%H%M%S_CT"); cov=_coverage(pred); status=_market_status(cov,client.odds_source)
    meta={"model_version":MODEL_VERSION,"season":int(season),"week":int(week),"history_start":int(history_start),"historical_games":int(len(history)),"training_rows":int(len(train_df)),"upcoming_games":int(len(games)),"validation":bundle.get("validation"),"metrics":dict(bundle["metrics"]),"model_selection":{"margin_blend_weight":float(bundle["margin_weight"]),"total_blend_weight":float(bundle["total_weight"])},"calibrated_margin_sigma":float(bundle["margin_sigma"]),"calibrated_total_sigma":float(bundle["total_sigma"]),"generated_at":stamp,"updated_at_ct":display,"schedule_source":"sportsdataverse/cfbfastR-data","odds_source":client.odds_source,"odds_errors":client.odds_errors,"market_coverage":cov,"market_status":status,"advanced_features":{"train_coverage":adv_train.get("coverage",0),"live_coverage":adv_live.get("coverage",0),"dynamic_coverage":adv_live.get("dynamic_coverage",0),"feature_count":adv_live.get("feature_count",adv_train.get("feature_count",0)),"dynamic_feature_count":adv_live.get("dynamic_feature_count",adv_train.get("dynamic_feature_count",0)),"identity":adv_live.get("identity",adv_train.get("identity",{})),"sources":adv_live.get("sources",[]),"errors":list(dict.fromkeys((adv_train.get("errors") or [])+(adv_live.get("errors") or [])))},"market_intelligence":intel_meta,"pregame_filter":pregame_meta,"current_context":{"coverage":ctx.get("coverage",0),"weather_coverage":ctx.get("weather_coverage",0),"sources":ctx.get("sources",[]),"errors":ctx.get("errors",[]),"source_available":bool(ctx.get("sources"))},"line_history":line_meta,"quant_picks":{"strong":int((pred.get("quant_signal",pd.Series(dtype=str))=="STRONG").sum()),"bet":int((pred.get("quant_signal",pd.Series(dtype=str))=="BET").sum()),"lean":int((pred.get("quant_signal",pd.Series(dtype=str))=="LEAN").sum())}}
    pred.to_csv(out/f"{base}.csv",index=False); (out/f"{base}.json").write_text(pred.to_json(orient="records",indent=2)); qcols=[c for c in ["game_id","date","away_team","home_team","model_margin_home","model_total","quant_signal","quant_market","quant_side","quant_price","quant_odds","quant_probability","quant_ev","quant_edge","stake_units","risk_multiplier","data_quality_score","context_risk","market_book_count","policy_block_reason"] if c in pred.columns]; q=pred[qcols].copy() if qcols else pd.DataFrame(); q=q[q.quant_signal!="PASS"].sort_values("quant_ev",ascending=False) if "quant_signal" in q.columns else q; q.to_csv(out/"quant_recommendations.csv",index=False); _write_quant_html(pred,out/"quant_card.html",display); snap=_append_snapshot_if_changed(pred,hist/"prediction_snapshots_v4.csv",stamp); meta["snapshot_appended"]=snap
    try:
        meta["live_performance"]=grade_prediction_history(client,hist,reports)
        tier_path=reports/"tier_performance.json"
        if tier_path.exists():
            tier=json.loads(tier_path.read_text())
            meta["tier_validation"]={
                "status":tier.get("status"),
                "graded_tier_bets":tier.get("graded_tier_bets",0),
                "validation_eligible_bets":tier.get("validation_eligible_bets",0),
                "excluded_from_validation":tier.get("excluded_from_validation",0),
                "validated_cells":tier.get("validated_cells",0),
                "underperforming_cells":tier.get("underperforming_cells",0),
            }
            for target in (out/"tier_performance.json",docs/"tier_performance.json"):
                target.write_text(json.dumps(tier,indent=2))
        for name in ("tier_performance.csv","tier_validation_clean.csv","live_graded_tiers.csv","live_graded_tiers_clean.csv"):
            src=reports/name
            if src.exists():
                (out/name).write_text(src.read_text())
                (docs/name).write_text(src.read_text())
    except Exception as exc:
        meta["live_performance"]={"error":f"{type(exc).__name__}: {exc}"}
        meta["tier_validation"]={"status":"ERROR","error":f"{type(exc).__name__}: {exc}"}
    health=build_health_report(meta,pred,reports); meta["health"]=health; (out/"system_health.json").write_text(json.dumps(health,indent=2)); (out/f"{base}_metadata.json").write_text(json.dumps(meta,indent=2)); pages=max(1,math.ceil(len(pred)/14)); render_html(pred,out/f"{base}.html",week,display,status)
    for old in out.glob(f"{base}_run_*_page*.png"):
        old.unlink()
    for p in range(1,pages+1):
        render_png(pred,out/f"{base}_page{p}.png",p,week,display,status)
        render_png(
            pred,
            out/f"{base}_run_{run_tag}_page{p}.png",
            p,
            week,
            display,
            status,
        )
    render_html(pred,docs/"index.html",week,display,status); _write_quant_html(pred,docs/"quant.html",display); (docs/"latest.csv").write_text(pred.to_csv(index=False)); (docs/"latest.json").write_text(pred.to_json(orient="records",indent=2)); (docs/"metadata.json").write_text(json.dumps(meta,indent=2)); (docs/"system_health.json").write_text(json.dumps(health,indent=2)); (docs/".nojekyll").write_text("")
    with (hist/"run_metadata_v4.jsonl").open("a") as f: f.write(json.dumps(meta,separators=(",",":"))+"\n")
    _write_output_readme(out,base,meta,pages,run_tag); return pred,meta
