"""Forward validation for displayed CFB STRONG/BET/LEAN tiers.

This module is diagnostic only. It evaluates the exact market-specific tier labels
published by the Cooper-style board using flat 1u risk and stored timestamp-safe
forward snapshots. It never changes fair-score projections or promotes deployment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .market import american_implied, replica_ml_label, replica_spread_label, replica_total_label, roi

MARKETS = ("moneyline", "spread", "total")
TIERS = ("STRONG", "BET", "LEAN")


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _wilson_interval(wins: int, losses: int, z: float = 1.96) -> list[float | None]:
    n = wins + losses
    if n <= 0:
        return [None, None]
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
        / denom
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _bootstrap_roi(values: pd.Series, seed: int = 2026, draws: int = 4000) -> list[float | None]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) < 20:
        return [None, None]
    rng = np.random.default_rng(seed)
    sims = np.empty(draws)
    for idx in range(draws):
        sims[idx] = rng.choice(x, size=len(x), replace=True).mean()
    low, high = np.quantile(sims, [0.025, 0.975])
    return [float(low), float(high)]


def _status(summary: Mapping[str, object]) -> str:
    n = int(summary.get("graded_bets") or 0)
    roi = _number(summary.get("flat_roi"))
    clv = _number(summary.get("avg_execution_clv"))
    gap = _number(summary.get("calibration_gap"))
    roi_ci = summary.get("roi_ci_95")
    low = high = None
    if isinstance(roi_ci, (list, tuple)) and len(roi_ci) >= 2:
        low = _number(roi_ci[0])
        high = _number(roi_ci[1])

    if n < 20:
        return "EARLY_SAMPLE"
    if n < 50:
        return "DEVELOPING"
    if (
        n >= 100
        and low is not None
        and low > 0
        and clv is not None
        and clv > 0
        and gap is not None
        and abs(gap) <= 0.08
    ):
        return "VALIDATED"
    if (
        n >= 50
        and high is not None
        and high < 0
        and clv is not None
        and clv < 0
    ):
        return "UNDERPERFORMING"
    if roi is not None and roi > 0 and clv is not None and clv >= 0:
        return "PROMISING"
    return "UNVALIDATED"


def expected_display_tier(
    market: str,
    row: Mapping[str, object],
) -> str:
    """Recompute the tier implied by the exact stored displayed line/price."""

    market = str(market or "").lower()
    home = str(row.get("home_team") or "")
    side = str(row.get("quant_side") or "")
    probability = _number(row.get("model_probability"))
    if market == "moneyline":
        odds = _number(row.get("execution_odds"))
        if probability is None or odds is None:
            return ""
        badge, _ = replica_ml_label(probability, odds)
        return badge if roi(probability, odds) > 0 else ""

    if market == "spread":
        margin = _number(row.get("model_margin_home"))
        line = _number(row.get("quant_price"))
        if margin is None or line is None or side not in {home, str(row.get("away_team") or "")}:
            return ""
        home_line = line if side == home else -line
        badge, _ = replica_spread_label(margin, home_line)
        return badge

    if market == "total":
        model_total = _number(row.get("model_total"))
        line = _number(row.get("quant_price"))
        if model_total is None or line is None:
            return ""
        badge, _ = replica_total_label(model_total, line)
        return badge
    return ""


def summarize_tier_rows(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "graded_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": None,
            "hit_rate_ci_95": [None, None],
            "flat_units": 0.0,
            "flat_roi": None,
            "roi_ci_95": [None, None],
            "avg_odds": None,
            "avg_break_even_probability": None,
            "avg_model_probability": None,
            "calibration_gap": None,
            "brier": None,
            "avg_edge": None,
            "avg_ev": None,
            "avg_execution_clv": None,
            "positive_execution_clv_rate": None,
            "execution_clv_samples": 0,
            "status": "EARLY_SAMPLE",
            "validated": False,
        }

    result = pd.to_numeric(frame.get("result"), errors="coerce")
    profit = pd.to_numeric(frame.get("flat_profit", frame.get("profit")), errors="coerce")
    wins = int((result > 0).sum())
    losses = int((result < 0).sum())
    pushes = int((result == 0).sum())
    decisions = wins + losses
    hit_rate = wins / decisions if decisions else None

    odds = pd.to_numeric(frame.get("execution_odds"), errors="coerce").dropna()
    break_even = []
    for value in odds:
        try:
            break_even.append(float(american_implied(float(value))))
        except Exception:
            continue

    probabilities = pd.to_numeric(frame.get("model_probability"), errors="coerce")
    binary = pd.DataFrame({"p": probabilities, "result": result}).dropna()
    binary = binary[binary["result"] != 0].copy()
    if len(binary):
        outcomes = (binary["result"] > 0).astype(float)
        avg_model_probability = float(binary["p"].mean())
        actual_rate = float(outcomes.mean())
        calibration_gap = avg_model_probability - actual_rate
        brier = float(((binary["p"] - outcomes) ** 2).mean())
    else:
        avg_model_probability = None
        calibration_gap = None
        brier = None

    clv = pd.to_numeric(frame.get("execution_clv"), errors="coerce").dropna()
    edge = pd.to_numeric(frame.get("model_edge"), errors="coerce").dropna()
    ev = pd.to_numeric(frame.get("model_ev"), errors="coerce").dropna()

    summary: dict[str, object] = {
        "graded_bets": int(len(frame)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": hit_rate,
        "hit_rate_ci_95": _wilson_interval(wins, losses),
        "flat_units": float(profit.sum()) if profit.notna().any() else 0.0,
        "flat_roi": float(profit.mean()) if profit.notna().any() else None,
        "roi_ci_95": _bootstrap_roi(profit),
        "avg_odds": float(odds.mean()) if len(odds) else None,
        "avg_break_even_probability": (
            float(np.mean(break_even)) if break_even else None
        ),
        "avg_model_probability": avg_model_probability,
        "calibration_gap": calibration_gap,
        "brier": brier,
        "avg_edge": float(edge.mean()) if len(edge) else None,
        "avg_ev": float(ev.mean()) if len(ev) else None,
        "avg_execution_clv": float(clv.mean()) if len(clv) else None,
        "positive_execution_clv_rate": (
            float((clv > 0).mean()) if len(clv) else None
        ),
        "execution_clv_samples": int(len(clv)),
    }
    summary["status"] = _status(summary)
    summary["validated"] = summary["status"] == "VALIDATED"
    return summary


def build_tier_performance(frame: pd.DataFrame, clean_frame: pd.DataFrame | None = None) -> dict[str, object]:
    """Build posted and clean market × tier forward-validation matrices."""

    if frame.empty:
        return {
            "schema_version": 1,
            "graded_tier_bets": 0,
            "validation_eligible_bets": 0,
            "excluded_from_validation": 0,
            "status": "EARLY_SAMPLE",
            "by_market": {},
            "by_tier": {},
            "by_market_tier": {},
            "matrix": [],
            "clean_matrix": [],
            "clean_by_market_tier": {},
            "validated_cells": 0,
            "underperforming_cells": 0,
            "methodology": (
                "posted matrix uses the first timestamp-safe displayed tier per game/market; "
                "actual American odds when available; spread/total use explicit -110 "
                "fallback only when the stored display snapshot lacks a price"
            ),
        }

    data = frame.copy()
    data["market"] = data.get("market", "").astype(str).str.lower()
    data["tier"] = data.get("tier", "").astype(str).str.upper()
    data = data[data["market"].isin(MARKETS) & data["tier"].isin(TIERS)].copy()
    if "tier_consistent" not in data.columns:
        data["tier_consistent"] = True
    if "validation_eligible" not in data.columns:
        data["validation_eligible"] = True
    data["tier_consistent"] = data["tier_consistent"].fillna(False).astype(bool)
    data["validation_eligible"] = data["validation_eligible"].fillna(False).astype(bool)

    if clean_frame is None:
        clean = data[data["validation_eligible"]].copy()
    else:
        clean = clean_frame.copy()
        if clean.empty:
            clean = pd.DataFrame(columns=data.columns)
        else:
            clean["market"] = clean.get("market", "").astype(str).str.lower()
            clean["tier"] = clean.get("tier", "").astype(str).str.upper()
            clean = clean[
                clean["market"].isin(MARKETS) & clean["tier"].isin(TIERS)
            ].copy()
            if "validation_eligible" in clean.columns:
                clean = clean[clean["validation_eligible"].fillna(False).astype(bool)].copy()

    by_market = {
        market: summarize_tier_rows(data[data.market == market])
        for market in MARKETS
    }
    by_tier = {
        tier: summarize_tier_rows(data[data.tier == tier])
        for tier in TIERS
    }
    nested: dict[str, dict[str, object]] = {}
    clean_nested: dict[str, dict[str, object]] = {}
    matrix: list[dict[str, object]] = []
    clean_matrix: list[dict[str, object]] = []
    for market in MARKETS:
        nested[market] = {}
        clean_nested[market] = {}
        for tier in TIERS:
            summary = summarize_tier_rows(
                data[(data.market == market) & (data.tier == tier)]
            )
            clean_summary = summarize_tier_rows(
                clean[(clean.market == market) & (clean.tier == tier)]
            )
            nested[market][tier] = summary
            clean_nested[market][tier] = clean_summary
            matrix.append({"market": market, "tier": tier, **summary})
            clean_matrix.append({"market": market, "tier": tier, **clean_summary})

    validated = sum(row["status"] == "VALIDATED" for row in clean_matrix)
    underperforming = sum(row["status"] == "UNDERPERFORMING" for row in clean_matrix)
    sample = int(len(data))
    clean_sample = int(len(clean))
    status = (
        "VALIDATED_SEGMENTS"
        if validated
        else "TRACKING"
        if clean_sample >= 50
        else "EARLY_SAMPLE"
    )
    return {
        "schema_version": 1,
        "graded_tier_bets": sample,
        "validation_eligible_bets": clean_sample,
        "excluded_from_validation": sample - clean_sample,
        "inconsistent_badge_rows": int((~data["tier_consistent"]).sum()),
        "unverified_price_rows": int((~data.get("price_verified", pd.Series(False, index=data.index)).fillna(False).astype(bool)).sum()),
        "status": status,
        "by_market": by_market,
        "by_tier": by_tier,
        "by_market_tier": nested,
        "matrix": matrix,
        "clean_by_market_tier": clean_nested,
        "clean_matrix": clean_matrix,
        "validated_cells": validated,
        "underperforming_cells": underperforming,
        "methodology": (
            "posted matrix uses the first timestamp-safe displayed tier per game/market; "
            "clean matrix uses the first pre-kickoff tier that satisfies price provenance "
            "and badge-consistency requirements; flat 1u risk; P/L is independent of "
            "Kelly/portfolio sizing; actual stored American "
            "odds are used when available; spread/total fall back to -110 only when "
            "the stored display snapshot has no price; pushes are excluded from "
            "hit-rate/calibration denominators; VALIDATED requires >=100 bets, "
            "positive 95% ROI lower bound, positive average execution CLV and "
            "|calibration gap| <= 8 percentage points; only rows whose stored "
            "display badge recomputes from the stored displayed line/price and whose "
            "execution price is verified may contribute to validation status"
        ),
    }


def refresh_display_market_tiers(row: Mapping[str, object]) -> dict[str, object]:
    """Recompute displayed Cooper-style badges after best-line selection."""

    out: dict[str, object] = {}
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
    home_probability = _number(row.get("calibrated_home_probability"))

    ml_side = str(row.get("ml_team") or "")
    if home_probability is not None and ml_side in {home, away}:
        if ml_side == home:
            odds = _number(row.get("best_home_ml"))
            if odds is None:
                odds = _number(row.get("home_ml"))
            book = row.get("best_home_ml_book")
            quote_at = row.get("best_home_ml_quote_at")
            quote_time_source = row.get("best_home_ml_quote_time_source")
            probability = home_probability
        else:
            odds = _number(row.get("best_away_ml"))
            if odds is None:
                odds = _number(row.get("away_ml"))
            book = row.get("best_away_ml_book")
            quote_at = row.get("best_away_ml_quote_at")
            quote_time_source = row.get("best_away_ml_quote_time_source")
            probability = 1.0 - home_probability
        if odds is not None:
            badge, edge = replica_ml_label(probability, odds)
            ev = roi(probability, odds)
            out.update(
                {
                    "ml_odds": odds,
                    "ml_badge": badge if ev > 0 else "",
                    "ml_edge_pp": edge,
                    "ml_est_roi": ev,
                    "ml_book": book,
                    "ml_quote_at": quote_at,
                    "ml_quote_time_source": quote_time_source,
                }
            )

    spread_side = str(row.get("spread_team") or "")
    spread_line = _number(row.get("spread_line"))
    model_margin = _number(row.get("model_margin_home"))
    if spread_line is not None and model_margin is not None and spread_side in {home, away}:
        home_line = spread_line if spread_side == home else -spread_line
        badge, _ = replica_spread_label(model_margin, home_line)
        out["spread_badge"] = badge
        if spread_side == home:
            out["spread_book"] = row.get("best_home_spread_book")
            out["spread_quote_at"] = row.get("best_home_spread_quote_at")
            out["spread_quote_time_source"] = row.get("best_home_spread_quote_time_source")
        else:
            out["spread_book"] = row.get("best_away_spread_book")
            out["spread_quote_at"] = row.get("best_away_spread_quote_at")
            out["spread_quote_time_source"] = row.get("best_away_spread_quote_time_source")

    total_line = _number(row.get("market_total"))
    model_total = _number(row.get("model_total"))
    if total_line is not None and model_total is not None:
        badge, _ = replica_total_label(model_total, total_line)
        out["total_badge"] = badge
        direction = str(row.get("total_dir") or "").upper()
        if direction == "O":
            out["total_book"] = row.get("best_over_book")
            out["total_quote_at"] = row.get("best_over_quote_at")
            out["total_quote_time_source"] = row.get("best_over_quote_time_source")
        elif direction == "U":
            out["total_book"] = row.get("best_under_book")
            out["total_quote_at"] = row.get("best_under_quote_at")
            out["total_quote_time_source"] = row.get("best_under_quote_time_source")

    return out
