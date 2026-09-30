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

from .walkforward import chronological_partition, sort_chronologically


NON_FEATURES={"game_id","season","week","date","away_id","home_id","away_team","home_team","target_margin_home","target_total"}
LEAKAGE_FRAGMENTS=(
    "target_","market_","closing_","close_","moneyline","_odds","odds_",
    "quant_","stake_","clv","final_","postgame_","result_",
)


def feature_leakage_columns(columns):
    bad=[]
    for c in columns:
        lc=str(c).lower()
        if c in NON_FEATURES or any(x in lc for x in LEAKAGE_FRAGMENTS):
            bad.append(c)
    return bad


def feature_columns(df):
    """Select only numeric, populated, non-constant, pregame-safe model inputs."""
    cols=[]
    for c in df.columns:
        if c in NON_FEATURES or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(x in str(c).lower() for x in LEAKAGE_FRAGMENTS):
            continue
        s=pd.to_numeric(df[c],errors="coerce")
        if s.notna().sum()<max(25,int(len(df)*.02)):
            continue
        if s.nunique(dropna=True)<2:
            continue
        cols.append(c)
    return cols


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
    """Legacy fail-closed helper retained for compatibility.

    Stage 2 no longer calls this helper during model selection because doing so would
    consume the final evaluation block. Selection now ends on the tune block; the final
    evaluation block is diagnostic only.
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


class _IdentityProbabilityMap:
    def predict(self, values):
        return np.asarray(values, dtype=float)


class _ConstantMarginClassifier:
    def __init__(self, probability):
        self.probability=float(np.clip(probability,.01,.99))

    def predict_proba(self, X):
        n=len(np.asarray(X))
        p=np.full(n,self.probability,dtype=float)
        return np.column_stack([1-p,p])


def _fit_prob_calibrator(margins,y):
    x=np.asarray(margins,float)
    y=np.asarray(y,int)
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    if len(x)<10:
        p=(float(y.sum())+1.)/(len(y)+2.) if len(y) else .5
        return _ConstantMarginClassifier(p),_IdentityProbabilityMap()
    if len(np.unique(y))<2:
        p=(float(y.sum())+1.)/(len(y)+2.)
        return _ConstantMarginClassifier(p),_IdentityProbabilityMap()
    lr=LogisticRegression(C=.35,max_iter=2000).fit(x.reshape(-1,1),y)
    raw=lr.predict_proba(x.reshape(-1,1))[:,1]
    # Isotonic is intentionally reserved for a reasonably sized OOF sample. On
    # smaller calibration blocks Platt scaling is lower-variance and fails safer.
    minority=min(int((y==0).sum()),int((y==1).sum()))
    if len(y)>=120 and minority>=20 and len(np.unique(np.round(raw,8)))>=8:
        iso=IsotonicRegression(out_of_bounds="clip",y_min=.01,y_max=.99).fit(raw,y)
    else:
        iso=_IdentityProbabilityMap()
    return lr,iso


def _predict_prob(cal,margins):
    lr,iso=cal
    raw=lr.predict_proba(np.asarray(margins,float).reshape(-1,1))[:,1]
    return np.clip(iso.predict(raw),.01,.99)


def _walkfolds(d,cols,target,baseline):
    """Leak-free expanding-season diagnostics; never used to select final weights."""
    rows=[]
    seasons=sorted(pd.to_numeric(d.season,errors="coerce").dropna().astype(int).unique())
    for season in seasons:
        tr=d[pd.to_numeric(d.season,errors="coerce")<season].copy()
        va=d[pd.to_numeric(d.season,errors="coerce")==season].copy()
        if len(tr)<800 or len(va)<100:
            continue
        try:
            core,tune,_,_,_=chronological_partition(tr,min_core_rows=300,min_section_rows=50)
        except RuntimeError:
            continue
        tune_pair=_fit_pair(core,cols,target,baseline)
        tune_res=_residual(tune_pair,tune[cols])
        w,_=choose_blend_weight(tune[target],tune[baseline],tune_res)
        pair=_fit_pair(tr,cols,target,baseline)
        pred=va[baseline].to_numpy(float)+w*_residual(pair,va[cols])
        rows.append({
            "season":int(season),"rows":int(len(va)),
            "mae":float(mean_absolute_error(va[target],pred)),
            "rmse":float(mean_squared_error(va[target],pred)**.5),
            "baseline_mae":float(mean_absolute_error(va[target],va[baseline])),
            "weight":float(w),"weight_tuned_on":"strictly prior whole-week tune block",
        })
    return rows


def _eval_guard(weight, baseline_mae, model_mae):
    if float(weight)<=1e-12:
        return True,"tuning selected baseline weight 0; evaluation remained untouched"
    if float(model_mae)<float(baseline_mae)-1e-9:
        return True,"selected residual beat baseline on untouched chronological evaluation"
    return False,"selected residual did not beat baseline on untouched chronological evaluation; weight was not retuned on evaluation"


def train_models(df, validation_diagnostics=True):
    if len(df)<500:
        raise RuntimeError(f"Need at least 500 historical FBS games; got {len(df)}")
    d=sort_chronologically(df)
    cols=feature_columns(d)
    leaked=feature_leakage_columns(cols)
    if leaked:
        raise RuntimeError(f"Pregame feature contract violation: {', '.join(map(str,leaked))}")
    if len(cols)<8:
        raise RuntimeError(f"Feature stack is unexpectedly thin; only {len(cols)} usable numeric pregame features")

    core,tune,calib,evaluation,partition=chronological_partition(d)
    metrics={"temporal_partition":partition,"selection_uses_evaluation":False}
    sig={}; tuned_weights={}; pre_calib_pairs={}; pre_eval_pairs={}; production_pairs={}
    specs={"margin":("target_margin_home","baseline_margin"),"total":("target_total","baseline_total")}

    pre_calib=pd.concat([core,tune],ignore_index=True)
    pre_eval=pd.concat([core,tune,calib],ignore_index=True)
    for name,(target,baseline) in specs.items():
        selection_pair=_fit_pair(core,cols,target,baseline)
        tune_res=_residual(selection_pair,tune[cols])
        tuned_w,tune_mae=choose_blend_weight(tune[target],tune[baseline],tune_res)
        tuned_weights[name]=float(tuned_w)
        tune_baseline_mae=float(mean_absolute_error(tune[target],tune[baseline]))

        cal_pair=_fit_pair(pre_calib,cols,target,baseline); pre_calib_pairs[name]=cal_pair
        cal_pred=calib[baseline].to_numpy(float)+tuned_w*_residual(cal_pair,calib[cols])
        cal_err=calib[target].to_numpy(float)-cal_pred
        sigma=float(np.std(cal_err,ddof=1)) if len(cal_err)>1 else 6.0
        sig[name]=float(max(6.0,sigma if math.isfinite(sigma) else 6.0))

        eval_pair=_fit_pair(pre_eval,cols,target,baseline); pre_eval_pairs[name]=eval_pair
        eval_base=evaluation[baseline].to_numpy(float)
        eval_pred=eval_base+tuned_w*_residual(eval_pair,evaluation[cols])
        baseline_mae=float(mean_absolute_error(evaluation[target],eval_base))
        model_mae=float(mean_absolute_error(evaluation[target],eval_pred))
        guard_passed,guard_reason=_eval_guard(tuned_w,baseline_mae,model_mae)
        folds=_walkfolds(d,cols,target,baseline) if validation_diagnostics else []

        metrics[f"{name}_tuning_rows"]=int(len(tune))
        metrics[f"{name}_tuning_baseline_mae"]=tune_baseline_mae
        metrics[f"{name}_tuning_mae"]=float(tune_mae)
        metrics[f"{name}_tuned_weight"]=float(tuned_w)
        metrics[f"{name}_calibration_rows"]=int(len(calib))
        metrics[f"{name}_calibration_mae"]=float(mean_absolute_error(calib[target],cal_pred))
        metrics[f"{name}_baseline_mae"]=baseline_mae
        metrics[f"{name}_tuned_model_mae"]=model_mae
        metrics[f"{name}_mae"]=model_mae
        metrics[f"{name}_rmse"]=float(mean_squared_error(evaluation[target],eval_pred)**.5)
        metrics[f"{name}_release_weight"]=float(tuned_w)
        metrics[f"{name}_blend_weight"]=float(tuned_w)
        metrics[f"{name}_release_guard_passed"]=bool(guard_passed)
        metrics[f"{name}_release_guard_reason"]=guard_reason
        metrics[f"{name}_eval_rows"]=int(len(evaluation))
        metrics[f"{name}_walkforward_folds"]=folds
        metrics[f"{name}_walkforward_mae_mean"]=float(np.mean([x["mae"] for x in folds])) if folds else None
        metrics[f"{name}_walkforward_mae_std"]=float(np.std([x["mae"] for x in folds])) if folds else None
        metrics[f"{name}_walkforward_improved_folds"]=int(sum(x["mae"]<x["baseline_mae"] for x in folds))
        metrics[f"{name}_walkforward_fold_count"]=int(len(folds))
        production_pairs[name]=_fit_pair(d,cols,target,baseline)

    # Evaluation calibrator: fit only on genuinely out-of-sample calibration margins.
    cm=calib.baseline_margin.to_numpy(float)+tuned_weights["margin"]*_residual(pre_calib_pairs["margin"],calib[cols])
    cy=(calib.target_margin_home>0).astype(int).to_numpy()
    eval_cal=_fit_prob_calibrator(cm,cy)
    em=evaluation.baseline_margin.to_numpy(float)+tuned_weights["margin"]*_residual(pre_eval_pairs["margin"],evaluation[cols])
    ey=(evaluation.target_margin_home>0).astype(int).to_numpy()
    ep=_predict_prob(eval_cal,em)
    metrics["win_brier"]=float(brier_score_loss(ey,ep))
    metrics["win_log_loss"]=float(log_loss(ey,ep,labels=[0,1]))
    metrics["win_ece"]=_ece(ey,ep)
    metrics["win_calibration_eval_rows"]=int(len(evaluation))
    metrics["win_calibration_fit_rows"]=int(len(calib))
    metrics["win_calibration_weight"]=float(tuned_weights["margin"])
    metrics["win_calibration_source"]="out-of-sample calibration block; evaluated on untouched final block"
    metrics["feature_count"]=int(len(cols))
    metrics["feature_columns"]=list(cols)

    # Final live calibrator may use both calibration and evaluation OOF predictions,
    # but only after the untouched evaluation metrics above are frozen.
    prod_margins=np.concatenate([cm,em])
    prod_outcomes=np.concatenate([cy,ey])
    prod_cal=_fit_prob_calibrator(prod_margins,prod_outcomes)
    metrics["win_production_calibration_oof_rows"]=int(len(prod_outcomes))

    return {
        "columns":cols,
        "margin":production_pairs["margin"],
        "total":production_pairs["total"],
        "metrics":metrics,
        "margin_sigma":sig["margin"],"total_sigma":sig["total"],
        "margin_weight":tuned_weights["margin"],"total_weight":tuned_weights["total"],
        "margin_tuned_weight":tuned_weights["margin"],"total_tuned_weight":tuned_weights["total"],
        "probability_calibrator":prod_cal,
        "validation":"nested whole-week chronology: core fit -> tune weight -> OOF calibration -> untouched evaluation; final refit keeps the tune-selected weight and uses OOF probability calibration",
    }


def predict_models(bundle,frame):
    if frame.empty: return np.array([]),np.array([])
    missing=[c for c in bundle["columns"] if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Live feature parity failure; missing: {', '.join(map(str,missing[:12]))}")
    X=frame[bundle["columns"]]
    m=frame.baseline_margin.to_numpy(float)+bundle["margin_weight"]*_residual(bundle["margin"],X)
    t=frame.baseline_total.to_numpy(float)+bundle["total_weight"]*_residual(bundle["total"],X)
    return np.clip(m,-48,48),np.clip(t,28,92)


def predict_home_probabilities(bundle,margins):
    return _predict_prob(bundle["probability_calibrator"],margins)
