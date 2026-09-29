from __future__ import annotations

import math
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

def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]

def _ridge(alpha=22.):
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])

def _boost():
    return Pipeline([("impute",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(max_depth=3,learning_rate=.04,max_iter=280,l2_regularization=9,min_samples_leaf=28,random_state=26))])

def _fit_pair(train,cols,target,baseline):
    y=train[target]-train[baseline]
    r,b=_ridge(),_boost(); r.fit(train[cols],y); b.fit(train[cols],y)
    return r,b

def _residual(pair,X):
    r,b=pair
    return .62*r.predict(X)+.38*b.predict(X)

def choose_blend_weight(target,baseline,residual,grid=None):
    target=np.asarray(target,float); baseline=np.asarray(baseline,float); residual=np.asarray(residual,float)
    grid=np.asarray(grid if grid is not None else np.linspace(0,1,21))
    best=(0.,float(mean_absolute_error(target,baseline)))
    for w in grid:
        mae=float(mean_absolute_error(target,baseline+float(w)*residual))
        if mae<best[1]-1e-10:
            best=(float(w),mae)
    return best

def release_weight_guard(tuned_weight, baseline_mae, tuned_mae, folds, min_fold_win_rate=.50):
    """Fail closed when a learned residual does not survive independent release checks.

    A zero weight is a legitimate production model: it means the independent baseline
    beat the learned correction.  Season folds must have their weight tuned strictly
    inside the training period; the held-out season is never used to select its weight.
    """
    w=float(tuned_weight)
    if w<=1e-12:
        return 0.0, True, "tuning selected the independent baseline"
    valid=[f for f in (folds or []) if f.get("mae") is not None and f.get("baseline_mae") is not None]
    fold_rate=(sum(float(f["mae"]) < float(f["baseline_mae"]) for f in valid)/len(valid)) if valid else 0.0
    release_better=float(tuned_mae) < float(baseline_mae)-1e-9
    if release_better and fold_rate>=float(min_fold_win_rate):
        return w, True, f"learned residual passed release holdout and {fold_rate:.0%} of leak-free season folds"
    reasons=[]
    if not release_better: reasons.append("release holdout did not beat baseline")
    if fold_rate<float(min_fold_win_rate): reasons.append(f"only {fold_rate:.0%} of leak-free season folds beat baseline")
    return 0.0, False, "; ".join(reasons) or "release guard failed"

def _ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); edges=np.linspace(0,1,bins+1); e=0.
    for i in range(bins):
        m=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else edges[i+1]+1e-12))
        if m.any(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)

def _fit_prob_calibrator(margins,y):
    X=np.asarray(margins,float).reshape(-1,1); y=np.asarray(y,int)
    lr=LogisticRegression(C=.35,max_iter=2000).fit(X,y)
    raw=lr.predict_proba(X)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip").fit(raw,y)
    return lr,iso

def _predict_prob(cal,margins):
    lr,iso=cal
    raw=lr.predict_proba(np.asarray(margins,float).reshape(-1,1))[:,1]
    return np.clip(iso.predict(raw),.01,.99)

def _walkfolds(d,cols,target,baseline):
    """Leak-free expanding-season validation.

    Each fold tunes the residual weight on a chronological tail of *prior* seasons,
    then evaluates the next season once. The evaluation season is never used to fit
    either the regressors or its blend weight.
    """
    rows=[]
    seasons=sorted(pd.to_numeric(d.season,errors="coerce").dropna().astype(int).unique())
    for season in seasons:
        tr=d[pd.to_numeric(d.season,errors="coerce")<season].copy()
        va=d[pd.to_numeric(d.season,errors="coerce")==season].copy()
        if len(tr)<800 or len(va)<100: continue
        cut=max(600,int(len(tr)*.80)); cut=min(cut,len(tr)-100)
        core,tune=tr.iloc[:cut],tr.iloc[cut:]
        if len(tune)<75: continue
        pair=_fit_pair(core,cols,target,baseline)
        tune_res=_residual(pair,tune[cols])
        w,_=choose_blend_weight(tune[target],tune[baseline],tune_res)
        # Deliberately keep the pair fitted before the tuning tail. This makes the
        # season score fully out of sample without a second fit using tune outcomes.
        pred=va[baseline].to_numpy(float)+w*_residual(pair,va[cols])
        rows.append({
            "season":int(season),"rows":int(len(va)),
            "mae":float(mean_absolute_error(va[target],pred)),
            "rmse":float(mean_squared_error(va[target],pred)**.5),
            "baseline_mae":float(mean_absolute_error(va[target],va[baseline])),
            "weight":float(w),"weight_tuned_on":"prior-season chronological tail",
        })
    return rows

def train_models(df):
    if len(df)<500:
        raise RuntimeError(f"Need at least 500 historical FBS games; got {len(df)}")
    d=df.sort_values(["season","week","date","game_id"]).reset_index(drop=True)
    cols=feature_columns(d); n=len(d)
    # Four chronological roles: core fit / blend tune / probability calibration / release holdout.
    # The final slice is not used to tune the candidate weight. It is a fail-closed release gate.
    a=max(300,int(n*.64)); b=max(a+75,int(n*.75)); c=max(b+75,int(n*.91)); c=min(c,n-75)
    core,tune,calib,release=d.iloc[:a],d.iloc[a:b],d.iloc[b:c],d.iloc[c:]
    metrics={}; sig={}; tuned_weights={}; release_weights={}; release_reasons={}
    specs={"margin":("target_margin_home","baseline_margin"),"total":("target_total","baseline_total")}
    pre_pairs={}
    for name,(target,baseline) in specs.items():
        tune_pair=_fit_pair(core,cols,target,baseline)
        tune_res=_residual(tune_pair,tune[cols])
        tuned_w,_=choose_blend_weight(tune[target],tune[baseline],tune_res)
        tuned_weights[name]=float(tuned_w)

        pre_pair=_fit_pair(d.iloc[:b],cols,target,baseline); pre_pairs[name]=pre_pair
        release_base=release[baseline].to_numpy(float)
        release_res=_residual(pre_pair,release[cols])
        tuned_pred=release_base+tuned_w*release_res
        baseline_mae=float(mean_absolute_error(release[target],release_base))
        tuned_mae=float(mean_absolute_error(release[target],tuned_pred))
        folds=_walkfolds(d,cols,target,baseline)
        release_w,guard_passed,guard_reason=release_weight_guard(tuned_w,baseline_mae,tuned_mae,folds)
        release_weights[name]=float(release_w); release_reasons[name]=guard_reason
        deployed_pred=release_base+release_w*release_res

        metrics[f"{name}_baseline_mae"]=baseline_mae
        metrics[f"{name}_tuned_model_mae"]=tuned_mae
        metrics[f"{name}_mae"]=float(mean_absolute_error(release[target],deployed_pred))
        metrics[f"{name}_rmse"]=float(mean_squared_error(release[target],deployed_pred)**.5)
        metrics[f"{name}_tuned_weight"]=float(tuned_w)
        metrics[f"{name}_release_weight"]=float(release_w)
        metrics[f"{name}_blend_weight"]=float(release_w)  # backwards-compatible deployed weight
        metrics[f"{name}_release_guard_passed"]=bool(guard_passed)
        metrics[f"{name}_release_guard_reason"]=guard_reason
        metrics[f"{name}_eval_rows"]=int(len(release))
        metrics[f"{name}_walkforward_folds"]=folds
        metrics[f"{name}_walkforward_mae_mean"]=float(np.mean([x["mae"] for x in folds])) if folds else None
        metrics[f"{name}_walkforward_mae_std"]=float(np.std([x["mae"] for x in folds])) if folds else None
        metrics[f"{name}_walkforward_improved_folds"]=int(sum(x["mae"]<x["baseline_mae"] for x in folds))
        metrics[f"{name}_walkforward_fold_count"]=int(len(folds))
        sig[name]=float(max(6,np.std(release[target].to_numpy(float)-deployed_pred,ddof=1)))

    # Probability calibration is evaluated honestly with the candidate margin weight,
    # before the release gate is allowed to alter production behavior.
    margin_pre=pre_pairs["margin"]
    cm=calib.baseline_margin.to_numpy(float)+tuned_weights["margin"]*_residual(margin_pre,calib[cols])
    eval_cal=_fit_prob_calibrator(cm,(calib.target_margin_home>0).astype(int))
    em=release.baseline_margin.to_numpy(float)+tuned_weights["margin"]*_residual(margin_pre,release[cols])
    ep=_predict_prob(eval_cal,em); ey=(release.target_margin_home>0).astype(int).to_numpy()
    metrics["win_brier"]=float(brier_score_loss(ey,ep))
    metrics["win_log_loss"]=float(log_loss(ey,ep,labels=[0,1]))
    metrics["win_ece"]=_ece(ey,ep)
    metrics["win_calibration_eval_rows"]=int(len(release))
    metrics["win_calibration_weight"]=float(tuned_weights["margin"])

    # If the margin release gate ever falls back, refit only the production calibrator
    # on the dedicated calibration slice using the deployed weight. Evaluation metrics
    # above remain tied to the untouched candidate path and are not overwritten.
    prod_cm=calib.baseline_margin.to_numpy(float)+release_weights["margin"]*_residual(margin_pre,calib[cols])
    prod_cal=_fit_prob_calibrator(prod_cm,(calib.target_margin_home>0).astype(int))

    return {
        "columns":cols,
        "margin":_fit_pair(d,cols,"target_margin_home","baseline_margin"),
        "total":_fit_pair(d,cols,"target_total","baseline_total"),
        "metrics":metrics,
        "margin_sigma":sig["margin"],"total_sigma":sig["total"],
        "margin_weight":release_weights["margin"],"total_weight":release_weights["total"],
        "margin_tuned_weight":tuned_weights["margin"],"total_tuned_weight":tuned_weights["total"],
        "probability_calibrator":prod_cal,
        "validation":"nested chronological core/tune/calibration/release holdout + leak-free expanding-season walk-forward + fail-closed baseline fallback",
    }

def predict_models(bundle,frame):
    if frame.empty: return np.array([]),np.array([])
    X=frame[bundle["columns"]]
    m=frame.baseline_margin.to_numpy(float)+bundle["margin_weight"]*_residual(bundle["margin"],X)
    t=frame.baseline_total.to_numpy(float)+bundle["total_weight"]*_residual(bundle["total"],X)
    return np.clip(m,-48,48),np.clip(t,28,92)

def predict_home_probabilities(bundle,margins):
    return _predict_prob(bundle["probability_calibrator"],margins)
