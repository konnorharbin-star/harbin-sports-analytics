from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .entry_provenance import verified_entry_mask

MIN_SEGMENT_BETS = 50


def _segment(df, col):
    out = {}
    if col not in df.columns:
        return out
    for k, g in df.groupby(col, dropna=False):
        p = pd.to_numeric(g.get("profit"), errors="coerce").dropna()
        c = pd.to_numeric(g.get("clv"), errors="coerce").dropna()
        out[str(k)] = {
            "bets": int(len(p)),
            "roi": float(p.mean()) if len(p) else None,
            "units": float(p.sum()) if len(p) else 0.0,
            "avg_clv": float(c.mean()) if len(c) else None,
        }
    return out


def _max_drawdown(profits):
    x = pd.to_numeric(pd.Series(profits), errors="coerce").fillna(0).to_numpy(float)
    if not len(x):
        return 0.0
    curve = np.cumsum(x); peak = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    return float(np.max(peak - curve))


def _block_bootstrap(df, seed=26, n_boot=3000):
    if df.empty or len(df) < 30:
        return [None, None]
    if {"season", "week"}.issubset(df.columns):
        blocks = []
        for _, g in df.groupby(["season", "week"], sort=False):
            p = pd.to_numeric(g.get("profit"), errors="coerce").dropna()
            if len(p): blocks.append((float(p.sum()), int(len(p))))
        if len(blocks) >= 8:
            rng = np.random.default_rng(seed); vals = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(blocks), size=len(blocks))
                units = sum(blocks[i][0] for i in idx); bets = sum(blocks[i][1] for i in idx)
                vals.append(units / max(1, bets))
            return [float(v) for v in np.quantile(vals, [.025, .975])]
    p = pd.to_numeric(df.get("profit"), errors="coerce").dropna().to_numpy(float)
    if len(p) < 30: return [None, None]
    rng = np.random.default_rng(seed); vals = [float(rng.choice(p, size=len(p), replace=True).mean()) for _ in range(n_boot)]
    return [float(v) for v in np.quantile(vals, [.025, .975])]


def _overall(df):
    if df.empty:
        return {"bets":0,"wins":0,"losses":0,"pushes":0,"win_rate":None,"units":0.0,"roi":None,"max_drawdown":0.0,"avg_clv":None,"positive_clv_rate":None,"clv_samples":0,"roi_ci_95":[None,None]}
    p = pd.to_numeric(df.get("profit"), errors="coerce").dropna(); r = pd.to_numeric(df.get("result"), errors="coerce")
    c = pd.to_numeric(df.get("clv"), errors="coerce").dropna(); wins = int((r > 0).sum()); losses = int((r < 0).sum())
    return {
        "bets": int(len(p)), "wins": wins, "losses": losses, "pushes": int((r == 0).sum()),
        "win_rate": float(wins / max(1, wins + losses)), "units": float(p.sum()), "roi": float(p.mean()) if len(p) else None,
        "max_drawdown": _max_drawdown(p), "avg_clv": float(c.mean()) if len(c) else None,
        "positive_clv_rate": float((c > 0).mean()) if len(c) else None, "clv_samples": int(len(c)),
        "roi_ci_95": _block_bootstrap(df),
    }


def _promotion_sample(bets: pd.DataFrame):
    """Only explicit affirmative verification can enter promotion evidence.

    Legacy used_distinct_open merely describes archive fields and cannot prove
    the quote existed at the decision time.
    """
    if "entry_quote_verified" not in bets.columns:
        return bets.iloc[0:0].copy(), "unverified"
    flag = verified_entry_mask(bets)
    return bets.loc[flag].copy(), "entry_quote_verified"


def _positive_segments(segments):
    return sum(
        1 for x in segments.values()
        if int(x.get("bets", 0) or 0) >= MIN_SEGMENT_BETS
        and x.get("roi") is not None and float(x["roi"]) > 0
        and x.get("avg_clv") is not None and float(x["avg_clv"]) > 0
    )


def build_evidence_report(summary_path="reports/backtest_summary.json", bets_path="reports/backtest_bets.csv", out_path="reports/evidence_report.json"):
    try: summary = json.loads(Path(summary_path).read_text())
    except Exception: summary = {}
    bets = pd.read_csv(bets_path, low_memory=False) if Path(bets_path).exists() else pd.DataFrame()
    promotion, provenance = _promotion_sample(bets) if len(bets) else (pd.DataFrame(), "unverified")
    overall = _overall(promotion); all_archive = summary.get("overall") or _overall(bets)
    by_market = _segment(promotion, "market") if len(promotion) else {}
    by_signal = _segment(promotion, "signal") if len(promotion) else {}
    by_season = _segment(promotion, "season") if len(promotion) else {}
    by_week = _segment(promotion, "week") if len(promotion) else {}
    n = int(overall.get("bets", 0) or 0); roi = overall.get("roi"); clv = overall.get("avg_clv"); ci = overall.get("roi_ci_95") or [None, None]
    positive_markets = _positive_segments(by_market); positive_seasons = _positive_segments(by_season)
    verified = provenance == "entry_quote_verified" and n > 0
    robust = n >= 1000 and ci[0] is not None and float(ci[0]) > 0 and clv is not None and float(clv) > 0 and positive_markets >= 2 and positive_seasons >= 2 and verified
    if robust: status = "ROBUST"
    elif n >= 500 and roi is not None and float(roi) > 0 and clv is not None and float(clv) > 0 and verified: status = "VALIDATED"
    elif n >= 150: status = "DEVELOPING"
    else: status = "UNPROVEN"
    report = {
        "status": status,
        "overall": overall,
        "all_archive_overall": all_archive,
        "promotion_sample": {
            "entry_quote_verified": bool(verified),
            "provenance_field": provenance,
            "verified_bets": n,
            "all_archive_bets": int(len(bets)),
            "excluded_unverified_bets": int(max(0, len(bets) - len(promotion))),
            "positive_markets": int(positive_markets),
            "positive_seasons": int(positive_seasons),
            "min_segment_bets": MIN_SEGMENT_BETS,
        },
        "by_market": by_market, "by_signal": by_signal, "by_season": by_season, "by_week": by_week,
        "criteria": {
            "robust": "1000+ verified opening-entry bets, week-block ROI 95% CI lower bound > 0, positive CLV, and >=2 markets plus >=2 seasons each with 50+ bets, positive ROI and positive CLV",
            "validated": "500+ verified opening-entry bets with positive ROI and CLV; not sufficient for PRODUCTION",
            "developing": "150+ verified opening-entry bets",
            "unproven": "below developing sample or entry timing is not independently verified",
        },
        "note": "Only time-valid archived opening entries qualify for promotion evidence. Archive-final fallbacks remain research diagnostics and cannot establish production readiness. Historical results do not guarantee future profit.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); Path(out_path).write_text(json.dumps(report, indent=2)); return report
