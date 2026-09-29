from __future__ import annotations

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


def _season_walkforward(d,cols,target,baseline,max_folds=3):
    seasons=sorted(int(x) for x in d.season.dropna().unique()); folds=[]
    for season in seasons[-max_folds:]:
        tr=d[d.season<season]; va=d[d.season==season]
        if len(tr)<500 or len(va)<100: continue
        pair=_fit_pair(tr,cols,target,baseline); resid=_residual_predict(pair,va[cols]); w,mae=choose_blend_weight(va[target],va[baseline],resid)
        folds.append({"season":season,"rows":int(len(va)),"mae":mae,"baseline_mae":float(mean_absolute_error(va[target],va[baseline])),"weight":w})
    return folds


def train_models(df: pd.DataFrame):
    if len(df)<300: raise RuntimeError(f"Need at least 300 historical FBS games; got {len(df)}")
    d=df.sort_values(["season","week","date","game_id"]).reset_index(drop=True); cols=feature_columns(d); cut=max(250,int(len(d)*.82)); cut=min(cut,len(d)-100); tr,va=d.iloc[:cut],d.iloc[cut:]
    metrics={}; sigma={}; weights={}; holdout={}
    specs={"margin":("target_margin_home","baseline_margin"),"total":("target_total","baseline_total")}
    for name,(target,baseline) in specs.items():
        pair=_fit_pair(tr,cols,target,baseline); raw=va[baseline].to_numpy(dtype=float); residual=_residual_predict(pair,va[cols]); full=raw+residual; weight,selected_mae=choose_blend_weight(va[target],raw,residual); selected=raw+weight*residual
        metrics[f"{name}_baseline_mae"]=float(mean_absolute_error(va[target],raw)); metrics[f"{name}_unshrunk_mae"]=float(mean_absolute_error(va[target],full)); metrics[f"{name}_mae"]=selected_mae; metrics[f"{name}_rmse"]=float(mean_squared_error(va[target],selected)**.5); metrics[f"{name}_blend_weight"]=weight; metrics[f"{name}_holdout_rows"]=int(len(va)); sigma[name]=float(max(6,np.std(va[target].to_numpy(dtype=float)-selected,ddof=1))); weights[name]=weight; holdout[name]=selected
        folds=_season_walkforward(d,cols,target,baseline); metrics[f"{name}_walkforward_folds"]=folds
        if folds:
            metrics[f"{name}_walkforward_mae_mean"]=float(np.mean([x["mae"] for x in folds])); metrics[f"{name}_walkforward_mae_std"]=float(np.std([x["mae"] for x in folds]))
    # Probability calibrator: fit on first half of chronological holdout, score on second half.
    margin_pred=np.asarray(holdout["margin"],dtype=float); y=(va["target_margin_home"].to_numpy(dtype=float)>0).astype(int); split=max(100,len(va)//2); calibrator=None
    if len(va)-split>=80 and len(np.unique(y[:split]))==2:
        calibrator=Pipeline([("scale",StandardScaler()),("logit",LogisticRegression(C=.65,max_iter=1000))]); calibrator.fit(margin_pred[:split].reshape(-1,1),y[:split]); p=calibrator.predict_proba(margin_pred[split:].reshape(-1,1))[:,1]; yt=y[split:]; metrics["win_brier"]=brier_score(yt,p); metrics["win_log_loss"]=log_loss_score(yt,p); metrics["win_ece"]=expected_calibration_error(yt,p); metrics["win_calibration_rows"]=int(len(yt))
    return {"columns":cols,"margin":_fit_pair(d,cols,"target_margin_home","baseline_margin"),"total":_fit_pair(d,cols,"target_total","baseline_total"),"metrics":metrics,"margin_sigma":sigma["margin"],"total_sigma":sigma["total"],"margin_weight":weights["margin"],"total_weight":weights["total"],"win_calibrator":calibrator,"validation":"chronological 82/18 holdout + season walk-forward folds + shrinkage guard"}


def predict_models(bundle,frame: pd.DataFrame):
    if frame.empty: return np.array([]),np.array([])
    X=frame[bundle["columns"]]; margin=frame["baseline_margin"].to_numpy(dtype=float)+bundle["margin_weight"]*_residual_predict(bundle["margin"],X); total=frame["baseline_total"].to_numpy(dtype=float)+bundle["total_weight"]*_residual_predict(bundle["total"],X); return np.clip(margin,-48,48),np.clip(total,28,92)


def predict_home_probabilities(bundle,margins):
    m=np.asarray(margins,dtype=float)
    cal=bundle.get("win_calibrator")
    if cal is not None: return cal.predict_proba(m.reshape(-1,1))[:,1]
    sigma=max(6,float(bundle["margin_sigma"])); return .5*(1+np.vectorize(__import__('math').erf)(m/(sigma*np.sqrt(2))))
