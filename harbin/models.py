from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NON_FEATURES={"game_id","season","week","date","away_id","home_id","away_team","home_team","target_margin_home","target_total"}

def feature_columns(df): return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
def _ridge(alpha=22.): return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])
def _boost(): return Pipeline([("impute",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(max_depth=3,learning_rate=.04,max_iter=280,l2_regularization=9,min_samples_leaf=28,random_state=26))])
def _fit_pair(train,cols,target,baseline):
    y=train[target]-train[baseline]; r,b=_ridge(),_boost(); r.fit(train[cols],y); b.fit(train[cols],y); return r,b
def _residual(pair,X): r,b=pair; return .62*r.predict(X)+.38*b.predict(X)
def choose_blend_weight(target,baseline,residual,grid=None):
    target=np.asarray(target,float); baseline=np.asarray(baseline,float); residual=np.asarray(residual,float); grid=np.asarray(grid if grid is not None else np.linspace(0,1,21)); best=(0.,float(mean_absolute_error(target,baseline)))
    for w in grid:
        mae=float(mean_absolute_error(target,baseline+float(w)*residual))
        if mae<best[1]-1e-10: best=(float(w),mae)
    return best

def _ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); edges=np.linspace(0,1,bins+1); e=0.
    for i in range(bins):
        m=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else edges[i+1]+1e-12))
        if m.any(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)
def _fit_prob_calibrator(margins,y):
    X=np.asarray(margins,float).reshape(-1,1); y=np.asarray(y,int); lr=LogisticRegression(C=.35,max_iter=2000).fit(X,y); raw=lr.predict_proba(X)[:,1]; iso=IsotonicRegression(out_of_bounds="clip").fit(raw,y); return lr,iso
def _predict_prob(cal,margins):
    lr,iso=cal; raw=lr.predict_proba(np.asarray(margins,float).reshape(-1,1))[:,1]; return np.clip(iso.predict(raw),.01,.99)
def _walkfolds(d,cols,target,baseline):
    rows=[]
    for season in sorted(pd.to_numeric(d.season,errors="coerce").dropna().astype(int).unique()):
        tr=d[d.season<season]; va=d[d.season==season]
        if len(tr)<800 or len(va)<100: continue
        pair=_fit_pair(tr,cols,target,baseline); res=_residual(pair,va[cols]); w,mae=choose_blend_weight(va[target],va[baseline],res); pred=va[baseline].to_numpy(float)+w*res
        rows.append({"season":int(season),"rows":int(len(va)),"mae":float(mae),"rmse":float(mean_squared_error(va[target],pred)**.5),"baseline_mae":float(mean_absolute_error(va[target],va[baseline])),"weight":float(w)})
    return rows

def train_models(df):
    if len(df)<500: raise RuntimeError(f"Need at least 500 historical FBS games; got {len(df)}")
    d=df.sort_values(["season","week","date","game_id"]).reset_index(drop=True); cols=feature_columns(d); n=len(d)
    # nested chronological split: core fit / weight tune / probability calibration / untouched evaluation
    a=int(n*.64); b=int(n*.75); c=int(n*.91); core,tune,calib,ev=d.iloc[:a],d.iloc[a:b],d.iloc[b:c],d.iloc[c:]
    metrics={}; sig={}; weights={}; specs={"margin":("target_margin_home","baseline_margin"),"total":("target_total","baseline_total")}
    for name,(target,baseline) in specs.items():
        pair=_fit_pair(core,cols,target,baseline); rr=_residual(pair,tune[cols]); w,_=choose_blend_weight(tune[target],tune[baseline],rr); weights[name]=w
        pair2=_fit_pair(d.iloc[:b],cols,target,baseline); ep=ev[baseline].to_numpy(float)+w*_residual(pair2,ev[cols]); metrics[f"{name}_baseline_mae"]=float(mean_absolute_error(ev[target],ev[baseline])); metrics[f"{name}_mae"]=float(mean_absolute_error(ev[target],ep)); metrics[f"{name}_rmse"]=float(mean_squared_error(ev[target],ep)**.5); metrics[f"{name}_blend_weight"]=float(w); metrics[f"{name}_eval_rows"]=int(len(ev)); sig[name]=float(max(6,np.std(ev[target].to_numpy(float)-ep,ddof=1))); folds=_walkfolds(d,cols,target,baseline); metrics[f"{name}_walkforward_folds"]=folds; metrics[f"{name}_walkforward_mae_mean"]=float(np.mean([x["mae"] for x in folds])) if folds else None; metrics[f"{name}_walkforward_mae_std"]=float(np.std([x["mae"] for x in folds])) if folds else None; metrics[f"{name}_walkforward_improved_folds"]=int(sum(x["mae"]<=x["baseline_mae"] for x in folds))
    # calibrate winner probabilities on a disjoint slice using pre-calibration score model
    mp=_fit_pair(d.iloc[:b],cols,"target_margin_home","baseline_margin"); cm=calib.baseline_margin.to_numpy(float)+weights["margin"]*_residual(mp,calib[cols]); cal=_fit_prob_calibrator(cm,(calib.target_margin_home>0).astype(int)); em=ev.baseline_margin.to_numpy(float)+weights["margin"]*_residual(mp,ev[cols]); ep=_predict_prob(cal,em); ey=(ev.target_margin_home>0).astype(int).to_numpy(); metrics["win_brier"]=float(brier_score_loss(ey,ep)); metrics["win_log_loss"]=float(log_loss(ey,ep,labels=[0,1])); metrics["win_ece"]=_ece(ey,ep); metrics["win_calibration_eval_rows"]=int(len(ev))
    return {"columns":cols,"margin":_fit_pair(d,cols,"target_margin_home","baseline_margin"),"total":_fit_pair(d,cols,"target_total","baseline_total"),"metrics":metrics,"margin_sigma":sig["margin"],"total_sigma":sig["total"],"margin_weight":weights["margin"],"total_weight":weights["total"],"probability_calibrator":cal,"validation":"nested chronological core/tune/calibration/evaluation + season walk-forward + zero-weight guard"}
def predict_models(bundle,frame):
    if frame.empty:return np.array([]),np.array([])
    X=frame[bundle["columns"]]; m=frame.baseline_margin.to_numpy(float)+bundle["margin_weight"]*_residual(bundle["margin"],X); t=frame.baseline_total.to_numpy(float)+bundle["total_weight"]*_residual(bundle["total"],X); return np.clip(m,-48,48),np.clip(t,28,92)
def predict_home_probabilities(bundle,margins): return _predict_prob(bundle["probability_calibrator"],margins)
