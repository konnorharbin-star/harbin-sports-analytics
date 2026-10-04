from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from .advanced import AdvancedFeatureStore, canon_team
from .calibration import brier_score, log_loss_score, expected_calibration_error, calibration_table
from .data import SportsDataVerseClient
from .market import no_vig, norm_cdf, conditional_margin_sigma, conditional_total_sigma
from .models import train_models, predict_models, predict_home_probabilities
from .pro_market import quant_signal
from .ratings import OpponentAdjustedRatings

ARCHIVE_ODDS_URL="https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/csv/cfb_line_odds.csv.gz"
BOOK_PRIORITY=("draftkings","fanduel","espn","fanatics","circa","pinnacle","bovada")


def _safe(v):
    try:
        x=float(v); return x if math.isfinite(x) else np.nan
    except Exception: return np.nan

def _american_profit(o): o=float(o); return 100/abs(o) if o<0 else o/100
def _bet_profit(result,odds=-110): return 0. if result==0 else -1. if result<0 else _american_profit(odds)
def _max_drawdown(profits):
    x=np.asarray(profits,dtype=float)
    if not len(x): return 0.
    curve=np.cumsum(x); peak=np.maximum.accumulate(np.r_[0.,curve])[:-1]; return float(np.max(peak-curve))
def _bootstrap_roi_ci(profits,seed=26,n_boot=3000):
    x=np.asarray(profits,dtype=float)
    if len(x)<30: return [None,None]
    rng=np.random.default_rng(seed); vals=[float(rng.choice(x,size=len(x),replace=True).mean()) for _ in range(n_boot)]; return [float(x) for x in np.quantile(vals,[.025,.975])]


class ArchiveMarketStore:
    def __init__(self,cache_dir="cache/backtest"):
        self.cache=Path(cache_dir); self.cache.mkdir(parents=True,exist_ok=True); self.df=self._load(); self.by_id={}
        if len(self.df) and "game_id" in self.df.columns:
            self.df["_gid"]=self.df["game_id"].astype(str); self.by_id={k:v.copy() for k,v in self.df.groupby("_gid",sort=False)}
    def _load(self):
        p=self.cache/"cfb_line_odds.csv.gz"
        if not p.exists():
            r=requests.get(ARCHIVE_ODDS_URL,timeout=60,headers={"User-Agent":"HarbinSportsAnalytics/4.0"}); r.raise_for_status(); p.write_bytes(r.content)
        return pd.read_csv(p,compression="gzip",low_memory=False)
    @staticmethod
    def _rank(book):
        s=str(book).lower(); return next((i for i,b in enumerate(BOOK_PRIORITY) if b in s),len(BOOK_PRIORITY))
    def quote(self,game):
        rows=self.by_id.get(str(game.game_id))
        if rows is None or rows.empty or not {"market_type","abbr","lines","odds","book"}.issubset(rows.columns): return None
        work=rows.copy(); work["_market"]=work["market_type"].astype(str).str.lower().str.replace("-","_",regex=False).str.replace(" ","_",regex=False); work["_book"]=work["book"].astype(str); candidates=[]
        for book,br in work.groupby("_book",dropna=False):
            ms=set(br["_market"]); comp=sum(any(t in x for x in ms) for t in ("money","spread","total")); candidates.append((-comp,self._rank(book),str(book)))
        if not candidates: return None
        candidates.sort(); book=candidates[0][2]; br=work[work["_book"].astype(str)==book].copy(); hk,ak=canon_team(game.home_team),canon_team(game.away_team); br["_side"]=br["abbr"].map(canon_team)
        def side(market,key,col):
            z=br[br["_market"].str.contains(market,na=False)]; q=z[z["_side"]==key]
            if not len(q): q=z[z["_side"].map(lambda x:bool(x) and (x in key or key in x))]
            return _safe(q.iloc[-1][col]) if len(q) and col in q.columns else np.nan
        def total(col):
            z=br[br["_market"].str.contains("total",na=False)]
            if not len(z) or col not in z.columns: return np.nan
            over=z[z["_side"].str.contains("over",na=False)]; q=over if len(over) else z; return _safe(q.iloc[-1][col])
        out={"book":book,"final_home_ml":side("money",hk,"odds"),"final_away_ml":side("money",ak,"odds"),"final_home_spread":side("spread",hk,"lines"),"final_home_spread_odds":side("spread",hk,"odds"),"final_away_spread_odds":side("spread",ak,"odds"),"final_total":total("lines")}
        z=br[br["_market"].str.contains("total",na=False)]
        if len(z):
            over=z[z["_side"].str.contains("over",na=False)]; under=z[z["_side"].str.contains("under",na=False)]; out["final_over_odds"]=_safe((over if len(over) else z).iloc[-1]["odds"]); out["final_under_odds"]=_safe((under if len(under) else z).iloc[-1]["odds"])
        hol="opening_lines" in br.columns; hoo="opening_odds" in br.columns; out.update({"open_home_ml":side("money",hk,"opening_odds") if hoo else out["final_home_ml"],"open_away_ml":side("money",ak,"opening_odds") if hoo else out["final_away_ml"],"open_home_spread":side("spread",hk,"opening_lines") if hol else out["final_home_spread"],"open_home_spread_odds":side("spread",hk,"opening_odds") if hoo else out["final_home_spread_odds"],"open_away_spread_odds":side("spread",ak,"opening_odds") if hoo else out["final_away_spread_odds"],"open_total":total("opening_lines") if hol else out["final_total"]})
        if hoo and len(z):
            over=z[z["_side"].str.contains("over",na=False)]; under=z[z["_side"].str.contains("under",na=False)]; out["open_over_odds"]=_safe((over if len(over) else z).iloc[-1]["opening_odds"]); out["open_under_odds"]=_safe((under if len(under) else z).iloc[-1]["opening_odds"])
        else: out["open_over_odds"]=out.get("final_over_odds",-110); out["open_under_odds"]=out.get("final_under_odds",-110)
        for k in list(out):
            if k.startswith("open_") and pd.isna(out[k]):
                fk="final_"+k[5:]
                if fk in out: out[k]=out[fk]
        out["has_distinct_open"]=bool(hol or hoo); return out


def _grade_spread(actual_margin,side,home,line):
    val=actual_margin+line if side==home else -actual_margin+line; return 1 if val>1e-9 else -1 if val<-1e-9 else 0
def _grade_total(actual_total,side,line):
    val=actual_total-line; val=-val if side=="U" else val; return 1 if val>1e-9 else -1 if val<-1e-9 else 0
def _clv_spread(side,home,op,cl):
    if pd.isna(op) or pd.isna(cl): return np.nan
    a=op if side==home else -op; b=cl if side==home else -cl; return float(a-b)
def _clv_total(side,op,cl):
    if pd.isna(op) or pd.isna(cl): return np.nan
    return float(cl-op) if side=="O" else float(op-cl)
def _clv_ml(side,home,q):
    try:
        pao,pho=no_vig(q["open_away_ml"],q["open_home_ml"]); pac,phc=no_vig(q["final_away_ml"],q["final_home_ml"]); return float(phc-pho) if side==home else float(pac-pao)
    except Exception: return np.nan


def _market_bets(game,margin,total,p_home,sigma_m,sigma_t,q):
    bets=[]; am=float(game.home_score)-float(game.away_score); at=float(game.home_score)+float(game.away_score)
    if not any(pd.isna(q.get(k,np.nan)) for k in ("open_home_ml","open_away_ml")):
        pa,ph=no_vig(q["open_away_ml"],q["open_home_ml"]); choices=[(game.home_team,p_home,q["open_home_ml"],100*(p_home-ph)),(game.away_team,1-p_home,q["open_away_ml"],100*((1-p_home)-pa))]; side,p,odds,edge=max(choices,key=lambda x:x[3]); ev=p*_american_profit(odds)-(1-p); sig=quant_signal(ev,edge,p,"moneyline")
        if sig!="PASS":
            result=1 if ((am>0 and side==game.home_team) or (am<0 and side==game.away_team)) else 0 if am==0 else -1; bets.append({"market":"moneyline","side":side,"line":odds,"probability":p,"edge":edge,"ev":ev,"signal":sig,"result":result,"profit":_bet_profit(result,odds),"clv":_clv_ml(side,game.home_team,q),"book":q["book"]})
    hs=q.get("open_home_spread",np.nan)
    if not pd.isna(hs):
        eh=float(margin)+float(hs); side=game.home_team if eh>=0 else game.away_team; line=float(hs) if side==game.home_team else -float(hs); p=norm_cdf(abs(eh)/conditional_margin_sigma(sigma_m,margin)); odds=q.get("open_home_spread_odds") if side==game.home_team else q.get("open_away_spread_odds"); odds=-110 if pd.isna(odds) else float(odds); ev=p*_american_profit(odds)-(1-p); sig=quant_signal(ev,abs(eh),p,"spread")
        if sig!="PASS":
            res=_grade_spread(am,side,game.home_team,line); bets.append({"market":"spread","side":side,"line":line,"probability":p,"edge":abs(eh),"ev":ev,"signal":sig,"result":res,"profit":_bet_profit(res,odds),"odds":odds,"clv":_clv_spread(side,game.home_team,q.get("open_home_spread"),q.get("final_home_spread")),"book":q["book"]})
    ot=q.get("open_total",np.nan)
    if not pd.isna(ot):
        edge=float(total)-float(ot); side="O" if edge>=0 else "U"; p=norm_cdf(abs(edge)/conditional_total_sigma(sigma_t,total)); odds=q.get("open_over_odds") if side=="O" else q.get("open_under_odds"); odds=-110 if pd.isna(odds) else float(odds); ev=p*_american_profit(odds)-(1-p); sig=quant_signal(ev,abs(edge),p,"total")
        if sig!="PASS":
            res=_grade_total(at,side,float(ot)); bets.append({"market":"total","side":side,"line":float(ot),"probability":p,"edge":abs(edge),"ev":ev,"signal":sig,"result":res,"profit":_bet_profit(res,odds),"odds":odds,"clv":_clv_total(side,q.get("open_total"),q.get("final_total")),"book":q["book"]})
    return bets


def _group_summary(df):
    if df.empty: return {"bets":0,"wins":0,"losses":0,"pushes":0,"win_rate":None,"units":0.,"roi":None,"max_drawdown":0.,"avg_clv":None,"clv_samples":0,"roi_ci_95":[None,None]}
    wins=int((df.result>0).sum()); losses=int((df.result<0).sum()); pushes=int((df.result==0).sum()); profits=df.profit.astype(float).to_numpy(); clv=pd.to_numeric(df.clv,errors="coerce").dropna(); return {"bets":int(len(df)),"wins":wins,"losses":losses,"pushes":pushes,"win_rate":float(wins/max(1,wins+losses)),"units":float(profits.sum()),"roi":float(profits.mean()),"max_drawdown":_max_drawdown(profits),"avg_clv":float(clv.mean()) if len(clv) else None,"positive_clv_rate":float((clv>0).mean()) if len(clv) else None,"clv_samples":int(len(clv)),"roi_ci_95":_bootstrap_roi_ci(profits)}


def run_backtest(start_season=2023,end_season=2025,history_start=2018,reports_dir="reports"):
    reports=Path(reports_dir); reports.mkdir(parents=True,exist_ok=True); client=SportsDataVerseClient(); games=client.history(history_start,int(end_season),99); game_map={str(g.game_id):g for g in games}; ratings=OpponentAdjustedRatings(); base=ratings.training_frame(games); advanced=AdvancedFeatureStore(history_start,int(end_season)); full,adv_meta=advanced.enrich(base); market=ArchiveMarketStore(); quote_map={gid:market.quote(g) for gid,g in game_map.items()}; groups=full[(full.season>=int(start_season))&(full.season<=int(end_season))][["season","week"]].drop_duplicates().sort_values(["season","week"]); bets=[]; game_preds=[]
    for _,gw in groups.iterrows():
        s,w=int(gw.season),int(gw.week); train=full[(full.season<s)|((full.season==s)&(full.week<w))]; target=full[(full.season==s)&(full.week==w)]
        if len(train)<800 or target.empty: continue
        bundle=train_models(train); margins,totals=predict_models(bundle,target); probs=predict_home_probabilities(bundle,margins)
        for j,(_,row) in enumerate(target.reset_index(drop=True).iterrows()):
            g=game_map.get(str(row.game_id))
            if g is None or g.home_score is None or g.away_score is None: continue
            m,t,p=float(margins[j]),float(totals[j]),float(probs[j]); game_preds.append({"game_id":g.game_id,"season":s,"week":w,"home_team":g.home_team,"away_team":g.away_team,"pred_margin_home":m,"pred_total":t,"home_win_probability":p,"actual_margin_home":float(g.home_score)-float(g.away_score),"actual_total":float(g.home_score)+float(g.away_score),"home_win":int(float(g.home_score)>float(g.away_score))}); q=quote_map.get(str(g.game_id))
            if q:
                for b in _market_bets(g,m,t,p,bundle["margin_sigma"],bundle["total_sigma"],q): b.update({"game_id":g.game_id,"season":s,"week":w,"home_team":g.home_team,"away_team":g.away_team}); bets.append(b)
    bdf=pd.DataFrame(bets); gdf=pd.DataFrame(game_preds); bdf.to_csv(reports/"backtest_bets.csv",index=False); gdf.to_csv(reports/"backtest_predictions.csv",index=False); summary={"start_season":int(start_season),"end_season":int(end_season),"history_start":int(history_start),"advanced_features":adv_meta,"overall":_group_summary(bdf),"by_market":{str(k):_group_summary(v) for k,v in bdf.groupby("market")} if len(bdf) else {},"by_signal":{str(k):_group_summary(v) for k,v in bdf.groupby("signal")} if len(bdf) else {}}
    edge_rows=[]
    if len(bdf):
        bins=[0,2,3,4,5,6,8,10,15,999]; labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]; bdf["edge_bucket"]=pd.cut(bdf.edge,bins=bins,labels=labels,right=False)
        for k,v in bdf.groupby("edge_bucket",observed=True): edge_rows.append({"edge_bucket":str(k),**_group_summary(v)})
    pd.DataFrame(edge_rows).to_csv(reports/"backtest_by_edge.csv",index=False); pd.DataFrame([{"signal":k,**v} for k,v in summary["by_signal"].items()]).to_csv(reports/"backtest_by_signal.csv",index=False)
    if len(gdf):
        y=gdf.home_win.to_numpy(dtype=float); p=gdf.home_win_probability.to_numpy(dtype=float); summary["prediction"]={"games":int(len(gdf)),"margin_mae":float(np.mean(np.abs(gdf.pred_margin_home-gdf.actual_margin_home))),"total_mae":float(np.mean(np.abs(gdf.pred_total-gdf.actual_total))),"brier":brier_score(y,p),"log_loss":log_loss_score(y,p),"ece":expected_calibration_error(y,p)}; pd.DataFrame(calibration_table(y,p)).to_csv(reports/"calibration.csv",index=False)
    else: summary["prediction"]={"games":0}
    o=summary["overall"]; ci=o.get("roi_ci_95") or [None,None]; summary["evidence"]={"status":"ESTABLISHED SAMPLE" if o["bets"]>=500 else "DEVELOPING SAMPLE" if o["bets"]>=100 else "SMALL SAMPLE","positive_roi_95ci":bool(ci[0] is not None and ci[0]>0),"positive_average_clv":bool(o.get("avg_clv") is not None and o["avg_clv"]>0),"note":"Historical results do not guarantee future profitability. Opening/archive lines are used only when present and sportsbook prices never enter score features."}; (reports/"backtest_summary.json").write_text(json.dumps(summary,indent=2)); return bdf,summary
