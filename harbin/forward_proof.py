from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import brier_score, expected_calibration_error
from .policy import _cluster_bootstrap_roi


def _stats(df: pd.DataFrame):
    if df.empty:
        return {"bets": 0, "roi": None, "units": 0.0, "roi_ci_95": [None, None], "avg_clv_proxy": None}
    p = pd.to_numeric(df.get("profit"), errors="coerce").dropna()
    ci = _cluster_bootstrap_roi(df.rename(columns={"clv_proxy": "clv"})) if len(p) else [None, None]
    clv = pd.to_numeric(df.get("clv_proxy"), errors="coerce").dropna()
    return {
        "bets": int(len(p)),
        "roi": float(p.mean()) if len(p) else None,
        "units": float(p.sum()) if len(p) else 0.0,
        "roi_ci_95": ci,
        "avg_clv_proxy": float(clv.mean()) if len(clv) else None,
    }


def build_forward_evidence(
    graded_path="reports/live_graded_predictions.csv",
    out_path="reports/forward_evidence.json",
    min_bets=300,
    min_week_blocks=12,
):
    """Evaluate only genuinely archived pre-kickoff paper decisions.

    This never enables real-money mode automatically. Its strongest outcome is
    ``ELIGIBLE_FOR_HUMAN_RELEASE_REVIEW`` after a substantial forward sample.
    """
    p = Path(graded_path)
    report = {
        "status": "INSUFFICIENT_SAMPLE",
        "real_money_allowed": False,
        "criteria": {
            "minimum_forward_bets": int(min_bets),
            "minimum_distinct_week_blocks": int(min_week_blocks),
            "roi": "> 0",
            "roi_95pct_lower_bound": "> 0",
            "average_clv_proxy": "> 0",
            "brier": "<= 0.24",
            "ece": "<= 0.08",
        },
    }
    if not p.exists():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        return report
    df = pd.read_csv(p, low_memory=False)
    if df.empty:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        return report

    bets = df[df.get("quant_signal", pd.Series("PASS", index=df.index)).astype(str).str.upper() != "PASS"].copy()
    overall = _stats(bets)
    by_market = {
        str(k): _stats(g.copy())
        for k, g in bets.groupby("quant_market", dropna=False)
    } if len(bets) and "quant_market" in bets.columns else {}
    blocks = int(bets[["season", "week"]].drop_duplicates().shape[0]) if {"season", "week"}.issubset(bets.columns) else 0

    brier = ece = None
    if {"calibrated_home_probability", "actual_margin_home"}.issubset(df.columns):
        ph = pd.to_numeric(df["calibrated_home_probability"], errors="coerce")
        margin = pd.to_numeric(df["actual_margin_home"], errors="coerce")
        mask = ph.notna() & margin.notna() & margin.ne(0)
        if int(mask.sum()) >= 50:
            y = (margin[mask] > 0).astype(int).to_numpy()
            prob = ph[mask].clip(.001, .999).to_numpy()
            brier = float(brier_score(y, prob))
            ece = float(expected_calibration_error(y, prob, 10))

    ci = overall.get("roi_ci_95") or [None, None]
    checks = {
        "sample": overall["bets"] >= int(min_bets),
        "week_blocks": blocks >= int(min_week_blocks),
        "positive_roi": overall.get("roi") is not None and float(overall["roi"]) > 0,
        "positive_roi_lower_bound": ci[0] is not None and float(ci[0]) > 0,
        "positive_clv": overall.get("avg_clv_proxy") is not None and float(overall["avg_clv_proxy"]) > 0,
        "calibration": brier is not None and ece is not None and brier <= .24 and ece <= .08,
    }
    if all(checks.values()):
        status = "ELIGIBLE_FOR_HUMAN_RELEASE_REVIEW"
    elif overall["bets"] >= 100:
        status = "DEVELOPING"
    else:
        status = "INSUFFICIENT_SAMPLE"

    report.update({
        "status": status,
        "real_money_allowed": False,
        "overall": overall,
        "by_market": by_market,
        "distinct_week_blocks": blocks,
        "calibration": {"brier": brier, "ece": ece},
        "checks": checks,
        "note": "Forward evidence is based on archived pre-kickoff paper snapshots. Even when all automated checks pass, real-money promotion requires explicit human review.",
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report
