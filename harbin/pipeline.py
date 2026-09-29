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
from .forward_proof import build_forward_evidence
from .grading import grade_prediction_history
from .health import build_health_report
from .line_history import attach_line_movement
from .market import norm_cdf, replica_win_probability, replica_ml_label, replica_spread_label, replica_total_label, no_vig, fair_american, roi
from .market_intel import MarketIntelligence
from .models import train_models, predict_models, predict_home_probabilities
from .monitoring import write_live_monitoring
from .portfolio import write_portfolio_outputs
from .policy import load_policy
from .pro_market import select_best_market, risk_multiplier
from .ratings import OpponentAdjustedRatings
from .render import render_html, render_png

MODEL_VERSION="7.0.0"

def _finite(v):
    try:return math.isfinite(float(v))
    except Exception:return False


def build_predictions(games,frame,bundle):
    if frame.empty:return pd.DataFrame()
    margins,totals=predict_models(bundle,frame);home_probs=predict_home_probabilities(bundle,margins);by_id={str(g.game_id):g for g in games};rows=[]
    for i,r in frame.reset_index(drop=True).iterrows():
        g=by_id[str(r.game_id)];m,t=float(margins[i]),float(totals[i]);hp,ap=(t+m)/2,(t-m)/2;winner=g.home_team if m>=0 else g.away_team;rp=replica_win_probability(m);sm=max(6,float(bundle["margin_sigma"]));st=max(6,float(bundle["total_sigma"]));ph=float(home_probs[i]);pw=ph if m>=0 else 1-ph;ml=g.home_ml if m>=0 else g.away_ml;mtag="";medge=mroi=np.nan;fair=fair_american(pw)
        if ml is not None:mtag,medge=replica_ml_label(rp,ml);mroi=roi(pw,ml)
        spteam=spline=np.nan;stag="";sedge=cp=np.nan
        if g.home_spread is not None:
            stag,eh=replica_spread_label(m,g.home_spread);sedge=abs(eh);spteam,spline=(g.home_team,float(g.home_spread)) if eh>=0 else (g.away_team,-float(g.home_spread));cp=norm_cdf(abs(eh)/sm)
        tdir=np.nan;ttag="";tedge=tp=np.nan
        if g.market_total is not None:
            ttag,et=replica_total_label(t,g.market_total);tedge=abs(et);tdir="O" if et>=0 else "U";tp=norm_cdf(abs(et)/st)
        nvw=qside=qedge=qroi=np.nan
        if g.away_ml is not None and g.home_ml is not None:
            pa,pmh=no_vig(g.away_ml,g.home_ml);eh=ph-pmh;ea=(1-ph)-pa
            if eh>=ea:qside=g.home_team;qedge=100*eh;qroi=roi(ph,g.home_ml)
            else:qside=g.away_team;qedge=100*ea;qroi=roi(1-ph,g.away_ml)
            nvw=pmh if m>=0 else pa
        rows.append({"game_id":g.game_id,"season":g.season,"week":g.week,"date":g.date,"away_team":g.away_team,"home_team":g.home_team,"away_score":int(round(ap)),"home_score":int(round(hp)),"away_score_exact":ap,"home_score_exact":hp,"model_margin_home":m,"model_total":t,"fair_spread_home":-m,"fair_total":t,"winner":winner,"win_probability":rp,"win_pct":int(round(100*rp)),"calibrated_home_probability":ph,"calibrated_winner_probability":pw,"provider":g.provider,"market_available":any(x is not None for x in (g.home_ml,g.away_ml,g.home_spread,g.market_total)),"away_ml":g.away_ml,"home_ml":g.home_ml,"ml_team":winner,"ml_odds":ml,"ml_badge":mtag,"ml_edge_pp":medge,"ml_est_roi":mroi,"ml_fair_odds":fair,"no_vig_market_prob_winner":nvw,"quant_best_ml_side":qside,"quant_best_ml_edge_pp":qedge,"quant_best_ml_roi":qroi,"market_spread_home":g.home_spread,"spread_team":spteam,"spread_line":spline,"spread_badge":stag,"spread_edge_pts":sedge,"cover_probability":cp,"market_total":g.market_total,"total_dir":tdir,"proj_total":int(round(t)),"total_badge":ttag,"total_edge_pts":tedge,"total_probability":tp,"baseline_margin":float(r.baseline_margin),"baseline_total":float(r.baseline_total),"elo_diff_home":float(r.elo_diff_home),"net_eff_diff_home":float(r.net_eff_diff_home),"volatility_avg":float(r.get("volatility_avg",np.nan))})
    return pd.DataFrame(rows).sort_values(["date","game_id"]).reset_index(drop=True)


def _coverage(pred):
    if pred.empty:return {"games":0,"any_market":0,"moneyline":0,"spread":0,"total":0}
    return {"games":int(len(pred)),"any_market":int(pred.market_available.fillna(False).astype(bool).sum()),"moneyline":int((pred.away_ml.notna()&pred.home_ml.notna()).sum()),"spread":int(pred.market_spread_home.notna().sum()),"total":int(pred.market_total.notna().sum())}

def _market_status(c,source):
    if c["games"]==0:return "No upcoming games were returned for this season/week."
    if c["any_market"]==0:return "PROJECTION-ONLY MODE — no verified sportsbook market data was available. No betting signals are shown."
    if c["any_market"]<c["games"]:return f"PARTIAL MARKET DATA — {c['any_market']}/{c['games']} games have at least one verified market from {source or 'available sources'}."
    return f"LIVE MARKET DATA — all {c['games']} games have verified market data from {source or 'available sources'}."
def _display_time(dt):
    x=dt.astimezone(ZoneInfo("America/Chicago"))
    try:return x.strftime("%b %-d, %Y · %-I:%M %p CT")
    except ValueError:return x.strftime("%b %d, %Y · %I:%M %p CT").replace(" 0"," ")

def _prediction_signature(pred):
    if pred.empty:return ""
    cols=["game_id","model_margin_home","model_total","away_ml","home_ml","market_spread_home","market_total","provider","quant_signal","quant_market","quant_side","quant_price","paper_stake_units"]
    temp=pred[[c for c in cols if c in pred.columns]].copy()
    for c in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[c]):temp[c]=temp[c].round(6)
    return temp.to_json(orient="records")
def _append_snapshot_if_changed(pred,path,stamp):
    if pred.empty:return False
    sig=_prediction_signature(pred);sp=path.with_suffix(".signature");oldsig=sp.read_text() if sp.exists() else None
    if oldsig==sig:return False
    snap=pred.copy();snap.insert(0,"snapshot_at",stamp);old=pd.read_csv(path,low_memory=False) if path.exists() else pd.DataFrame();pd.concat([old,snap],ignore_index=True,sort=False).to_csv(path,index=False);sp.write_text(sig);return True


def _quantize(pred,bundle,adv_meta,ctx_meta):
    if pred.empty:return pred
    out=pred.copy();overall=float(adv_meta.get("coverage",0) or 0);dynamic=float(adv_meta.get("dynamic_coverage",0) or 0);ac=.35*overall+.65*dynamic;sm=max(6,float(bundle["margin_sigma"]));st=max(6,float(bundle["total_sigma"]))
    for i,r in out.iterrows():
        q=r.copy()
        if _finite(r.get("consensus_home_novig_probability")) and _finite(r.get("best_home_ml")) and _finite(r.get("best_away_ml")):
            ph=float(r.calibrated_home_probability);mh=float(r.consensus_home_novig_probability);eh,ea=ph-mh,(1-ph)-(1-mh)
            if eh>=ea:q["quant_best_ml_side"]=r.home_team;q["quant_best_ml_edge_pp"]=100*eh;q["quant_best_ml_roi"]=roi(ph,r.best_home_ml);q["home_ml"]=r.best_home_ml
            else:q["quant_best_ml_side"]=r.away_team;q["quant_best_ml_edge_pp"]=100*ea;q["quant_best_ml_roi"]=roi(1-ph,r.best_away_ml);q["away_ml"]=r.best_away_ml
        if _finite(r.get("consensus_home_spread")):
            e=float(r.model_margin_home)+float(r.consensus_home_spread);q["spread_edge_pts"]=abs(e);q["cover_probability"]=norm_cdf(abs(e)/sm);q["spread_team"],q["spread_line"]=(r.home_team,float(r.consensus_home_spread)) if e>=0 else (r.away_team,-float(r.consensus_home_spread))
        if _finite(r.get("consensus_total")):
            e=float(r.model_total)-float(r.consensus_total);q["total_edge_pts"]=abs(e);q["total_probability"]=norm_cdf(abs(e)/st);q["total_dir"]="O" if e>=0 else "U";q["market_total"]=float(r.consensus_total)
        mq=float(r.get("market_consensus_quality",.25) or .25);context_cov=float(ctx_meta.get("coverage",0) or 0);dq=max(.15,min(1,.55*ac+.30*mq+.15*context_cov));cr=float(r.get("context_risk",r.get("availability_risk",0)) or 0);vol=float(r.get("volatility_avg",14)) if _finite(r.get("volatility_avg")) else 14.;rm=risk_multiplier(cr,dq,vol);disagreement=0.
        if _finite(r.get("consensus_home_spread")):disagreement=max(disagreement,abs(float(r.model_margin_home)+float(r.consensus_home_spread)))
        if _finite(r.get("consensus_total")):disagreement=max(disagreement,abs(float(r.model_total)-float(r.consensus_total)))
        if disagreement>10:rm*=max(.5,1-.035*(disagreement-10))
        pick=select_best_market(q,rm);out.at[i,"data_quality_score"]=dq;out.at[i,"risk_multiplier"]=rm;out.at[i,"market_disagreement"]=disagreement
        for k,v in pick.items():out.at[i,k]=v
        if dq<.45 or float(pick.get("stake_units",0) or 0)<.08:out.at[i,"quant_signal"]="PASS";out.at[i,"stake_units"]=0.
    return out


def _write_quant_html(pred,path,updated):
    rank={"STRONG":0,"BET":1,"LEAN":2,"PASS":3};q=pred.copy();q["_rank"]=q.get("quant_signal",pd.Series("PASS",index=q.index)).map(rank).fillna(3);q=q.sort_values(["_rank","quant_ev"],ascending=[True,False]);rows=[]
    for _,r in q.iterrows():
        ev=float(r.quant_ev) if _finite(r.get("quant_ev")) else 0.;prob=float(r.quant_probability) if _finite(r.get("quant_probability")) else 0.;price=r.get("quant_price");action=str(r.get("portfolio_action") or "PASS");paper=float(r.get("paper_stake_units",0) or 0);rows.append(f"<tr><td>{r.away_team} @ {r.home_team}</td><td>{r.get('quant_signal','PASS')}</td><td>{action}</td><td>{r.get('quant_market') or '—'}</td><td>{r.get('quant_side') or '—'}</td><td>{price if pd.notna(price) else '—'}</td><td>{prob:.1%}</td><td>{ev:.1%}</td><td>{paper:.2f}</td><td>{float(r.get('data_quality_score',0) or 0):.0%}</td></tr>")
    Path(path).write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Harbin Quant Card</title><style>body{{background:#0f1113;color:#f2f2f2;font-family:Arial;margin:0;padding:24px}}.wrap{{max-width:1200px;margin:auto}}p{{color:#9da1a6}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #292c30;text-align:left}}th{{color:#888;font-size:11px}}tr:nth-child(even){{background:#17191c}}</style></head><body><div class='wrap'><h1>HARBIN QUANT CARD</h1><p>Updated {updated}. Evidence-gated EV signals, capped risk controls, and PAPER portfolio allocation. PASS is normal.</p><table><thead><tr><th>GAME</th><th>SIGNAL</th><th>ACTION</th><th>MARKET</th><th>SIDE</th><th>PRICE/LINE</th><th>MODEL P</th><th>EV</th><th>PAPER UNITS</th><th>DATA QUALITY</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></body></html>")


def _write_output_readme(out,base,meta,pages):
    pngs="\n".join(f"- [Cooper-style page {i}]({base}_page{i}.png)" for i in range(1,pages+1));h=meta.get("health",{});a=meta.get("advanced_features",{});p=meta.get("production_policy",{});fp=meta.get("forward_evidence",{});port=meta.get("portfolio",{})
    (out/"README.md").write_text(f"# Latest CFB model output\n\n**Model:** v{meta['model_version']}  \n**Season / Week:** {meta['season']} / {meta['week']}  \n**Updated:** {meta['updated_at_ct']}  \n**Market:** {meta['market_status']}  \n**Advanced dynamic coverage:** {float(a.get('dynamic_coverage',0) or 0):.0%}  \n**System health:** {h.get('system_health_score','—')}/100 *(readiness, not predicted profitability)*  \n**Historically final-tested markets:** {p.get('validated_markets',[]) or 'none'}  \n**Forward evidence:** {fp.get('status','INSUFFICIENT_SAMPLE')}  \n**Portfolio mode:** {str(port.get('mode','paper')).upper()} — paper allocated {port.get('paper_allocated_units',0)}u\n\n## Use these\n- [Interactive Cooper-style table]({base}.html)\n- [Quant card](quant_card.html)\n- [Quant recommendations CSV](quant_recommendations.csv)\n- [Portfolio card](portfolio_card.csv)\n- [Portfolio summary](portfolio_summary.json)\n- [Full model CSV]({base}.csv)\n- [Metadata / diagnostics]({base}_metadata.json)\n- [System health report](system_health.json)\n- [Live monitoring](live_monitoring.json)\n{pngs}\n\nThe Cooper-style table is the reconstructed presentation layer. The Quant card is the independent evidence-gated EV/risk layer. Missing verified markets display **NO LINE**. Real-money deployment remains disabled pending a separate human release review after strict historical and forward-paper evidence gates.\n")


def run_week(season=None,week=None,history_start=None,root="."):
    root=Path(root);out,docs,hist,reports=root/"outputs",root/"docs",root/"history",root/"reports"
    for p in (out,docs,hist,reports):p.mkdir(parents=True,exist_ok=True)
    client=SportsDataVerseClient()
    if season is None or week is None:
        ds,dw=client.detect();season=season or ds;week=week or dw
    history_start=history_start or max(2018,season-5);history=client.history(history_start,season,week);ratings=OpponentAdjustedRatings();base_train=ratings.training_frame(history);advanced=AdvancedFeatureStore(history_start,season,cache_dir=root/"cache"/"advanced");train_df,adv_train=advanced.enrich(base_train);bundle=train_models(train_df);games=[g for g in client.week(season,week) if not g.completed];base_frame=ratings.upcoming_frame(games);frame,adv_live=advanced.enrich(base_frame);pred=build_predictions(games,frame,bundle);intel=MarketIntelligence();pred,intel_meta=intel.attach(games,pred);context=ContextStore(season,schedule_frame=client.season_frame(season),cache_dir=root/"cache"/"context");pred,ctx=context.attach(pred);pred,line_meta=attach_line_movement(pred,hist);pred=_quantize(pred,bundle,adv_live,ctx);pred,portfolio=write_portfolio_outputs(pred,output_dir=out,policy_path=reports/"production_policy.json");now=datetime.now(timezone.utc);stamp=now.isoformat();display=_display_time(now);base=f"cfb_model_{season}_week{week}";cov=_coverage(pred);status=_market_status(cov,client.odds_source);policy=load_policy(reports/"production_policy.json")
    meta={"model_version":MODEL_VERSION,"season":int(season),"week":int(week),"history_start":int(history_start),"historical_games":int(len(history)),"training_rows":int(len(train_df)),"upcoming_games":int(len(games)),"validation":bundle.get("validation"),"metrics":dict(bundle["metrics"]),"model_selection":{"margin_blend_weight":float(bundle["margin_weight"]),"total_blend_weight":float(bundle["total_weight"])},"calibrated_margin_sigma":float(bundle["margin_sigma"]),"calibrated_total_sigma":float(bundle["total_sigma"]),"generated_at":stamp,"updated_at_ct":display,"schedule_source":"sportsdataverse/cfbfastR-data","odds_source":client.odds_source,"odds_errors":client.odds_errors,"market_coverage":cov,"market_status":status,"advanced_features":{"train_coverage":adv_train.get("coverage",0),"train_dynamic_coverage":adv_train.get("dynamic_coverage",0),"live_coverage":adv_live.get("coverage",0),"dynamic_coverage":adv_live.get("dynamic_coverage",0),"feature_count":adv_live.get("feature_count",adv_train.get("feature_count",0)),"dynamic_feature_count":adv_live.get("dynamic_feature_count",adv_train.get("dynamic_feature_count",0)),"identity":adv_live.get("identity",adv_train.get("identity",{})),"sources":adv_live.get("sources",[]),"errors":list(dict.fromkeys((adv_train.get("errors") or [])+(adv_live.get("errors") or [])))},"market_intelligence":intel_meta,"current_context":{"coverage":ctx.get("coverage",0),"weather_coverage":ctx.get("weather_coverage",0),"injury_coverage":ctx.get("injury_coverage",0),"travel_coverage":ctx.get("travel_coverage",0),"sources":ctx.get("sources",[]),"errors":ctx.get("errors",[]),"source_available":bool(ctx.get("sources"))},"line_history":line_meta,"production_policy":{"version":policy.get("version"),"deployment_mode":policy.get("deployment_mode","paper"),"candidate_markets":policy.get("candidate_markets",[]),"validated_markets":policy.get("validated_markets",[]),"final_test_status":policy.get("final_test_status","UNPROVEN"),"blocked_weeks":(policy.get("regime_filters") or {}).get("blocked_weeks",[])},"portfolio":portfolio,"quant_picks":{"strong":int((pred.get("quant_signal",pd.Series(dtype=str))=="STRONG").sum()),"bet":int((pred.get("quant_signal",pd.Series(dtype=str))=="BET").sum()),"lean":int((pred.get("quant_signal",pd.Series(dtype=str))=="LEAN").sum())}}
    snap=_append_snapshot_if_changed(pred,hist/"prediction_snapshots_v4.csv",stamp);meta["snapshot_appended"]=snap
    try:meta["live_performance"]=grade_prediction_history(client,hist,reports)
    except Exception as exc:meta["live_performance"]={"error":f"{type(exc).__name__}: {exc}"}
    try:forward=build_forward_evidence(reports/"live_graded_predictions.csv",reports/"forward_evidence.json")
    except Exception as exc:forward={"status":"ERROR","real_money_allowed":False,"error":f"{type(exc).__name__}: {exc}"}
    meta["forward_evidence"]=forward;monitoring=write_live_monitoring(pred,meta,output=out/"live_monitoring.json",reports_dir=reports);meta["live_monitoring"]=monitoring;health=build_health_report(meta,pred,reports);meta["health"]=health
    pred.to_csv(out/f"{base}.csv",index=False);(out/f"{base}.json").write_text(pred.to_json(orient="records",indent=2));qcols=[c for c in ["game_id","date","away_team","home_team","model_margin_home","model_total","quant_signal","portfolio_action","quant_market","quant_side","quant_price","quant_probability","quant_ev","quant_edge","paper_stake_units","portfolio_stake_units","risk_multiplier","data_quality_score","context_risk","market_book_count"] if c in pred.columns];q=pred[qcols].copy() if qcols else pd.DataFrame();q=q[q.quant_signal!="PASS"].sort_values("quant_ev",ascending=False) if "quant_signal" in q.columns else q;q.to_csv(out/"quant_recommendations.csv",index=False);_write_quant_html(pred,out/"quant_card.html",display);(out/"system_health.json").write_text(json.dumps(health,indent=2));(out/f"{base}_metadata.json").write_text(json.dumps(meta,indent=2));pages=max(1,math.ceil(len(pred)/14));render_html(pred,out/f"{base}.html",week,display,status)
    for p in range(1,pages+1):render_png(pred,out/f"{base}_page{p}.png",p,week,display,status)
    render_html(pred,docs/"index.html",week,display,status);_write_quant_html(pred,docs/"quant.html",display);(docs/"latest.csv").write_text(pred.to_csv(index=False));(docs/"latest.json").write_text(pred.to_json(orient="records",indent=2));(docs/"metadata.json").write_text(json.dumps(meta,indent=2));(docs/"system_health.json").write_text(json.dumps(health,indent=2));(docs/"forward_evidence.json").write_text(json.dumps(forward,indent=2));(docs/".nojekyll").write_text("")
    with (hist/"run_metadata_v4.jsonl").open("a") as f:f.write(json.dumps(meta,separators=(",",":"))+"\n")
    _write_output_readme(out,base,meta,pages);return pred,meta
