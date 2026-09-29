from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import brier_score, log_loss_score, expected_calibration_error

NON_FEATURES={"game_id","season","week","date","away_team","home_team","target_margin_home","target_total"}


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def _ridge(alpha=22.0):
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])


def _boost():
    return Pipeline([("impute",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(max_depth=3,learning_rate=.04,max_iter=280,l2_regularization=9.0,min_samples_leaf=28,random_state=26))])


def _fit_pair(train,cols,target,baseline):
    y=train[target]-train[baseline]; r,b=_ridge(),_boost(); r.fit(train[cols],y); b.fit(train[cols],y); return r,b


def _residual_predict(pair,X):
    r,b=pair; return .62*r.predict(X)+.38*b.predict(X)


def choose_blend_weight(target,baseline,residual,grid=None):
    target=np.asarray(target,dtype=float); baseline=np.asarray(baseline,dtype=float); residual=np.asarray(residual,dtype=float); grid=np.asarray(grid if grid is not None else np.linspace(0,1,21)); best_w=0.; best=float(mean_absolute_error(target,baseline))
    for w in grid:
        mae=float(mean_absolute_error(target,baseline+float(w)*residual))
        if mae<best-1e-10: best_w,best=float(w),mae
    return best_w,best


def _inner_weight(train,cols,target,baseline):
    if len(train)<500: return 0.0
    cut=max(300,int(len(train)*.82)); cut=min(cut,len(train)-100); core,tune=train.iloc[:cut],train.iloc[cut:]; pair=_fit_pair(core,cols,target,baseline); residual=_residual_predict(pair,tune[cols]); w,_=choose_blend_weight(tune[target],tune[baseline],residual); return w


def _season_walkforward(d,cols,target,baseline,max_folds=4):
    folds=[]; seasons=sorted(int(x) for x in d.season.dropna().unique())
    for season in seasons[-max_folds:]:
        tr=d[d.season<season].copy(); va=d[d.season==season].copy()
        if len(tr)<700 or len(va)<100: continue
        w=_inner_weight(tr,cols,target,baseline); pair=_fit_pair(tr,cols,target,baseline); pred=va[baseline].to_numpy(dtype=float)+w*_residual_predict(pair,va[cols]); folds.append({"season":season,"rows":int(len(va)),"mae":float(mean_absolute_error(va[target],pred)),"rmse":float(mean_squared_error(va[target],pred)**.5),"baseline_mae":float(mean_absolute_error(va[target],va[baseline])),"weight":float(w)})
    return folds


def train_models(df: pd.DataFrame):
    """Nested chronological core -> tuning -> calibration -> untouched evaluation."""
    if len(df)<600: raise RuntimeError(f"Need at least 600 historical FBS games; got {len(df)}")
    d=df.sort_values(["season","week","date","game_id"]).reset_index(drop=True); cols=feature_columns(d); n=len(d); i1=max(350,int(n*.70)); i2=max(i1+120,int(n*.82)); i3=max(i2+100,int(n*.91)); i3=min(i3,n-100); i2=min(i2,i3-80); i1=min(i1,i2-100); core,tune,cal,ev=d.iloc[:i1],d.iloc[i1:i2],d.iloc[i2:i3],d.iloc[i3:]
    if min(len(tune),len(cal),len(ev))<80: raise RuntimeError("Not enough rows for nested chronological validation blocks")
    metrics={}; sigma={}; weights={}; specs={"margin":("target_margin_home","baseline_margin"),"total":("target_total","baseline_total")}
    for name,(target,baseline) in specs.items():
        pair_core=_fit_pair(core,cols,target,baseline); tune_res=_residual_predict(pair_core,tune[cols]); weight,_=choose_blend_weight(tune[target],tune[baseline],tune_res); weights[name]=weight; pre_eval=pd.concat([core,tune,cal],ignore_index=True); pair_pre=_fit_pair(pre_eval,cols,target,baseline); pred=ev[baseline].to_numpy(dtype=float)+weight*_residual_predict(pair_pre,ev[cols]); base=ev[baseline].to_numpy(dtype=float); metrics[f"{name}_baseline_mae"]=float(mean_absolute_error(ev[target],base)); metrics[f"{name}_mae"]=float(mean_absolute_error(ev[target],pred)); metrics[f"{name}_rmse"]=float(mean_squared_error(ev[target],pred)**.5); metrics[f"{name}_blend_weight"]=float(weight); metrics[f"{name}_eval_rows"]=int(len(ev)); sigma[name]=float(max(6.0,np.std(ev[target].to_numpy(dtype=float)-pred,ddof=1))); folds=_season_walkforward(d,cols,target,baseline); metrics[f"{name}_walkforward_folds"]=folds
        if folds:
            metrics[f"{name}_walkforward_mae_mean"]=float(np.mean([x["mae"] for x in folds])); metrics[f"{name}_walkforward_mae_std"]=float(np.std([x["mae"] for x in folds])); metrics[f"{name}_walkforward_improved_folds"]=int(sum(x["mae"]<=x["baseline_mae"] for x in folds))
    mw=weights["margin"]; score_train=pd.concat([core,tune],ignore_index=True); pair_cal=_fit_pair(score_train,cols,"target_margin_home","baseline_margin"); cal_margin=cal["baseline_margin"].to_numpy(dtype=float)+mw*_residual_predict(pair_cal,cal[cols]); y_cal=(cal["target_margin_home"].to_numpy(dtype=float)>0).astype(int); calibrator=None
    if len(np.unique(y_cal))==2:
        calibrator=Pipeline([("scale",StandardScaler()),("logit",LogisticRegression(C=.65,max_iter=1000))]); calibrator.fit(cal_margin.reshape(-1,1),y_cal); pre_ev=pd.concat([core,tune,cal],ignore_index=True); pair_ev=_fit_pair(pre_ev,cols,"target_margin_home","baseline_margin"); ev_margin=ev["baseline_margin"].to_numpy(dtype=float)+mw*_residual_predict(pair_ev,ev[cols]); p=calibrator.predict_proba(ev_margin.reshape(-1,1))[:,1]; y=(ev["target_margin_home"].to_numpy(dtype=float)>0).astype(int); metrics["win_brier"]=brier_score(y,p); metrics["win_log_loss"]=log_loss_score(y,p); metrics["win_ece"]=expected_calibration_error(y,p); metrics["win_calibration_eval_rows"]=int(len(y))
    return {"columns":cols,"margin":_fit_pair(d,cols,"target_margin_home","baseline_margin"),"total":_fit_pair(d,cols,"target_total","baseline_total"),"metrics":metrics,"margin_sigma":sigma["margin"],"total_sigma":sigma["total"],"margin_weight":weights["margin"],"total_weight":weights["total"],"win_calibrator":calibrator,"validation":"nested chronological core/tune/calibration/evaluation + season walk-forward + zero-weight guard"}


def predict_models(bundle,frame: pd.DataFrame):
    if frame.empty: return np.array([]),np.array([])
    X=frame[bundle["columns"]]; margin=frame["baseline_margin"].to_numpy(dtype=float)+bundle["margin_weight"]*_residual_predict(bundle["margin"],X); total=frame["baseline_total"].to_numpy(dtype=float)+bundle["total_weight"]*_residual_predict(bundle["total"],X); return np.clip(margin,-48,48),np.clip(total,28,92)


def predict_home_probabilities(bundle,margins):
    m=np.asarray(margins,dtype=float); cal=bundle.get("win_calibrator")
    if cal is not None: return cal.predict_proba(m.reshape(-1,1))[:,1]
    sigma=max(6,float(bundle["margin_sigma"])); return np.array([.5*(1+math.erf(float(x)/(sigma*math.sqrt(2)))) for x in m],dtype=float)
