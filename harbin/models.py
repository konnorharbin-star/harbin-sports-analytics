from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NON_FEATURES = {"game_id","season","week","date","away_team","home_team","target_margin_home","target_total"}


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def _ridge(alpha=22.0):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _boost():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=.04, max_iter=280,
            l2_regularization=9.0, min_samples_leaf=28, random_state=26,
        )),
    ])


def _fit_pair(train, cols, target, baseline):
    y = train[target] - train[baseline]
    r, b = _ridge(), _boost()
    r.fit(train[cols], y)
    b.fit(train[cols], y)
    return r, b


def _pair_predict(pair, X, baseline):
    r, b = pair
    residual = .62*r.predict(X) + .38*b.predict(X)
    return baseline + residual


def train_models(df: pd.DataFrame):
    if len(df) < 300:
        raise RuntimeError(f"Need at least 300 historical FBS games; got {len(df)}")
    d = df.sort_values(["season","week","date","game_id"]).reset_index(drop=True)
    cols = feature_columns(d)
    cut = int(len(d)*.82)
    tr, va = d.iloc[:cut], d.iloc[cut:]
    metrics, sigma = {}, {}
    specs = {
        "margin": ("target_margin_home", "baseline_margin"),
        "total": ("target_total", "baseline_total"),
    }
    for name, (target, baseline) in specs.items():
        pair = _fit_pair(tr, cols, target, baseline)
        pred = _pair_predict(pair, va[cols], va[baseline].to_numpy())
        raw = va[baseline].to_numpy()
        metrics[f"{name}_mae"] = float(mean_absolute_error(va[target], pred))
        metrics[f"{name}_rmse"] = float(mean_squared_error(va[target], pred)**.5)
        metrics[f"{name}_baseline_mae"] = float(mean_absolute_error(va[target], raw))
        resid = va[target].to_numpy()-pred
        sigma[name] = float(max(6.0, np.std(resid, ddof=1)))
    return {
        "columns": cols,
        "margin": _fit_pair(d, cols, "target_margin_home", "baseline_margin"),
        "total": _fit_pair(d, cols, "target_total", "baseline_total"),
        "metrics": metrics,
        "margin_sigma": sigma["margin"],
        "total_sigma": sigma["total"],
    }


def predict_models(bundle, frame: pd.DataFrame):
    if frame.empty:
        return np.array([]), np.array([])
    X = frame[bundle["columns"]]
    margin = _pair_predict(bundle["margin"], X, frame["baseline_margin"].to_numpy())
    total = _pair_predict(bundle["total"], X, frame["baseline_total"].to_numpy())
    return np.clip(margin, -48, 48), np.clip(total, 28, 92)
