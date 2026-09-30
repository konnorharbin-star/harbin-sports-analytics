from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import feature_leakage_columns


def _summary(train: pd.Series, live: pd.Series) -> dict:
    t = pd.to_numeric(train, errors="coerce")
    l = pd.to_numeric(live, errors="coerce")
    tq = t.dropna().quantile([0.25, 0.5, 0.75]) if t.notna().any() else pd.Series(dtype=float)
    median = float(tq.get(0.5, np.nan)) if len(tq) else None
    iqr = float(tq.get(0.75, np.nan) - tq.get(0.25, np.nan)) if len(tq) else None
    live_median = float(l.dropna().median()) if l.notna().any() else None
    drift_iqr = None
    if median is not None and live_median is not None and iqr is not None and np.isfinite(iqr) and iqr > 1e-12:
        drift_iqr = float(abs(live_median - median) / iqr)
    return {
        "train_nonnull_rate": float(t.notna().mean()),
        "live_nonnull_rate": float(l.notna().mean()),
        "train_median": median,
        "train_iqr": iqr,
        "live_median": live_median,
        "live_median_drift_iqr": drift_iqr,
    }


def audit_feature_stack(train: pd.DataFrame, live: pd.DataFrame, model_columns: list[str], advanced_meta: dict | None = None) -> dict:
    """Audit training/live parity, leakage, missingness, and distribution drift."""
    advanced_meta = advanced_meta or {}
    issues = []
    cols = list(model_columns or [])
    duplicate_train = train.columns[train.columns.duplicated()].tolist()
    duplicate_live = live.columns[live.columns.duplicated()].tolist()
    missing_train = [c for c in cols if c not in train.columns]
    missing_live = [c for c in cols if c not in live.columns]
    leakage = feature_leakage_columns(cols)
    if duplicate_train:
        issues.append({"severity": "ERROR", "code": "duplicate_train_columns", "columns": duplicate_train})
    if duplicate_live:
        issues.append({"severity": "ERROR", "code": "duplicate_live_columns", "columns": duplicate_live})
    if missing_train:
        issues.append({"severity": "ERROR", "code": "missing_train_features", "columns": missing_train})
    if missing_live:
        issues.append({"severity": "ERROR", "code": "missing_live_features", "columns": missing_live})
    if leakage:
        issues.append({"severity": "ERROR", "code": "leakage_features", "columns": leakage})
    manifest = {}
    sparse, live_empty, high_drift = [], [], []
    for c in cols:
        if c not in train.columns or c not in live.columns:
            continue
        s = _summary(train[c], live[c])
        manifest[c] = s
        if s["train_nonnull_rate"] < 0.20:
            sparse.append(c)
        if s["live_nonnull_rate"] == 0.0:
            live_empty.append(c)
        if s["live_median_drift_iqr"] is not None and s["live_median_drift_iqr"] > 4.0:
            high_drift.append(c)
    if sparse:
        issues.append({"severity": "WARNING", "code": "sparse_training_features", "columns": sparse})
    if live_empty:
        issues.append({"severity": "ERROR", "code": "empty_live_features", "columns": live_empty})
    if high_drift:
        issues.append({"severity": "WARNING", "code": "large_live_distribution_shift", "columns": high_drift})
    pair_cov = float(advanced_meta.get("pair_feature_coverage", 0.0) or 0.0)
    dyn_pair_cov = float(advanced_meta.get("dynamic_pair_feature_coverage", 0.0) or 0.0)
    if advanced_meta and dyn_pair_cov < 0.75:
        issues.append({
            "severity": "WARNING", "code": "advanced_dynamic_pair_coverage",
            "value": dyn_pair_cov, "requirement": ">= 0.75",
        })
    errors = sum(x["severity"] == "ERROR" for x in issues)
    warnings = sum(x["severity"] == "WARNING" for x in issues)
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "model_feature_count": len(cols),
        "train_rows": int(len(train)),
        "live_rows": int(len(live)),
        "advanced_pair_feature_coverage": pair_cov,
        "advanced_dynamic_pair_feature_coverage": dyn_pair_cov,
        "asof_policy": advanced_meta.get("asof_policy"),
        "issues": issues,
        "manifest": manifest,
    }


def write_feature_audit(train: pd.DataFrame, live: pd.DataFrame, model_columns: list[str], advanced_meta: dict | None, path, fail_on_error=True) -> dict:
    report = audit_feature_stack(train, live, model_columns, advanced_meta)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    if fail_on_error and report["errors"]:
        codes = ", ".join(i["code"] for i in report["issues"] if i["severity"] == "ERROR")
        raise RuntimeError(f"Feature-stack audit failed: {codes}")
    return report
