from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))


def _distribution_drift(live, hist):
    a = pd.to_numeric(live, errors="coerce").dropna()
    b = pd.to_numeric(hist, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 50:
        return None
    ref_sd = max(float(b.std(ddof=1)), 1e-6)
    live_sd = float(a.std(ddof=1)) if len(a) > 1 else ref_sd
    z = abs(float(a.mean()) - float(b.mean())) / ref_sd
    ratio = live_sd / ref_sd
    score = _clip(100 - min(70, 25 * z + 25 * abs(math.log(max(0.1, ratio)))))
    return {
        "stability_score": round(score, 1),
        "mean_z": round(z, 4),
        "std_ratio": round(ratio, 4),
        "live_mean": float(a.mean()),
        "reference_mean": float(b.mean()),
        "live_std": live_sd,
        "reference_std": ref_sd,
        "live_n": int(len(a)),
        "reference_n": int(len(b)),
    }


def _drift(live, hist):
    """Backward-compatible scalar stability score."""
    d = _distribution_drift(live, hist)
    return None if d is None else d["stability_score"]


def build_live_monitoring(pred, meta, reports_dir="reports"):
    alerts = []
    scores = {}
    coverage = meta.get("market_coverage") or {}
    games = max(1, int(coverage.get("games", 0) or 0))
    scores["market_coverage"] = _clip(
        100
        * min(coverage.get("moneyline", 0), coverage.get("spread", 0), coverage.get("total", 0))
        / games
    )

    adv = meta.get("advanced_features") or {}
    dyn = float(adv.get("dynamic_coverage", adv.get("live_coverage", 0)) or 0)
    feature_count = int(adv.get("dynamic_feature_count", adv.get("feature_count", 0)) or 0)
    scores["advanced_coverage"] = _clip(100 * dyn) * min(1.0, feature_count / 10 if feature_count else 0.0)

    intel = meta.get("market_intelligence") or {}
    scores["multi_book"] = _clip(100 * float(intel.get("multi_book_coverage", 0) or 0))

    ctx = meta.get("current_context") or {}
    context_cov = float(ctx.get("coverage", 0) or 0)
    weather = float(ctx.get("weather_coverage", 0) or 0)
    scores["context"] = _clip(100 * (0.65 * context_cov + 0.35 * weather))

    metrics = meta.get("metrics") or {}
    brier = metrics.get("win_brier")
    ece = metrics.get("win_ece")
    scores["calibration"] = _clip(100 - 220 * max(0, float(brier or 0.25) - 0.20) - 250 * float(ece or 0.10))

    try:
        generated = datetime.fromisoformat(str(meta.get("generated_at")).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds() / 3600
    except Exception:
        age = 999
    scores["freshness"] = _clip(100 - 6 * max(0, age - 1))
    if age > 8:
        alerts.append(f"model output is {age:.1f} hours old")

    reference_path = Path(reports_dir) / "backtest_predictions.csv"
    drift_details = {}
    if reference_path.exists() and not pred.empty:
        try:
            hist = pd.read_csv(reference_path, low_memory=False)
            mappings = [
                ("model_margin_home", "pred_margin_home", "margin"),
                ("model_total", "pred_total", "total"),
                ("calibrated_home_probability", "home_win_probability", "probability"),
            ]
            for live_col, hist_col, name in mappings:
                if live_col in pred and hist_col in hist:
                    d = _distribution_drift(pred[live_col], hist[hist_col])
                    if d is not None:
                        drift_details[name] = d
                        scores["drift_" + name] = d["stability_score"]
                        if d["stability_score"] < 55:
                            alerts.append(
                                f"{name} prediction distribution shifted materially from the historical walk-forward reference "
                                f"({d['mean_z']:.2f}σ mean shift; stability {d['stability_score']:.1f}/100)"
                            )
        except Exception as exc:
            alerts.append(f"drift reference unavailable: {type(exc).__name__}")

    drift_scores = [d["stability_score"] for d in drift_details.values()]
    scores["distribution_stability"] = round(float(np.mean(drift_scores)), 1) if drift_scores else 65.0

    missing = float(pred.isna().mean().mean()) if not pred.empty else 1.0
    scores["output_completeness"] = _clip(100 * (1 - min(0.5, missing) / 0.5))

    weights = {
        "market_coverage": 0.13,
        "advanced_coverage": 0.17,
        "multi_book": 0.09,
        "context": 0.10,
        "calibration": 0.18,
        "freshness": 0.09,
        "distribution_stability": 0.14,
        "output_completeness": 0.10,
    }
    total = sum(weights[k] * scores.get(k, 0) for k in weights)

    if scores["market_coverage"] < 90:
        alerts.append("verified ML/spread/total coverage below 90%")
    if scores["advanced_coverage"] < 70:
        alerts.append("real pregame advanced-efficiency coverage below target")
    if scores["multi_book"] < 50:
        alerts.append("multi-book consensus is limited; configure THE_ODDS_API_KEY for free multi-book supplementation")
    if scores["context"] < 60:
        alerts.append("injury/weather/travel context coverage is limited")

    status = "ALERT" if total < 60 or any(d["stability_score"] < 40 for d in drift_details.values()) else "WARN" if alerts else "OK"
    return {
        "status": status,
        "live_readiness_score": round(total, 1),
        "scores": scores,
        "drift_details": drift_details,
        "drift_reference": str(reference_path) if reference_path.exists() else None,
        "alerts": list(dict.fromkeys(alerts)),
        "meaning": "operational/model-monitoring score; not a profitability guarantee",
    }


def write_live_monitoring(pred, meta, output="outputs/live_monitoring.json", reports_dir="reports"):
    report = build_live_monitoring(pred, meta, reports_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report
