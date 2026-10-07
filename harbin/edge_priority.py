"""Priority diagnostics for historically supported CFB edges.

This layer does not create model edge, change probabilities, alter fair lines, or
increase stake. It ranks already-supported candidates by how much current line and
price room remains before the historical support boundary disappears.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _regime_min_edge(row) -> float | None:
    if _finite(row.get("regime_min_edge")):
        return float(row.get("regime_min_edge"))
    band = str(row.get("regime_band") or "")
    if "-" not in band:
        return None
    try:
        return float(band.split("-", 1)[0])
    except Exception:
        return None


def _bet_to_spread_line(line, edge, min_edge) -> float | None:
    """Worst half-point spread that still preserves the regime's minimum edge."""

    if not (_finite(line) and _finite(edge) and _finite(min_edge)):
        return None
    raw_boundary = float(line) - max(0.0, float(edge) - float(min_edge))
    # The selected side loses edge as its displayed line moves numerically lower.
    # Round upward to the next half-point so the returned threshold never falls
    # below the minimum supported model-market disagreement.
    return math.ceil(raw_boundary * 2.0 - 1e-12) / 2.0


def classify_edge_priority(row) -> str:
    reliability = str(row.get("edge_reliability_status") or "")
    price_status = str(row.get("price_evidence_status") or "UNKNOWN")
    market = str(row.get("market") or "").lower()

    if reliability == "CONTRAINDICATED_SUBGROUP":
        return "REJECT_CONTRAINDICATED"
    if price_status == "OVERPRICED":
        return "REJECT_OVERPRICED"
    if reliability == "PERSISTENT_PARENT_ONLY":
        return "PARENT_ONLY"
    if reliability != "SUPPORTED_SUBGROUP":
        return "WATCH_ONLY"
    if price_status == "PLAUSIBLE":
        return "PLAUSIBLE_PRICE"
    if price_status != "CONFIRMED":
        return "WATCH_ONLY"

    price_cushion = (
        float(row.get("conservative_price_margin"))
        if _finite(row.get("conservative_price_margin"))
        else -math.inf
    )
    min_edge = _regime_min_edge(row)
    line_cushion = (
        max(0.0, float(row.get("edge")) - min_edge)
        if _finite(row.get("edge")) and min_edge is not None
        else math.inf
    )

    if market == "spread":
        price_thin = price_cushion < 0.02
        line_thin = line_cushion < 0.50
        if price_thin and line_thin:
            return "CORE_BOTH_THIN"
        if price_thin:
            return "CORE_PRICE_THIN"
        if line_thin:
            return "CORE_LINE_THIN"
        if price_cushion >= 0.05:
            return "ROBUST_CORE"
        return "CORE"

    if price_cushion >= 0.05:
        return "ROBUST_CORE"
    if price_cushion < 0.02:
        return "CORE_PRICE_THIN"
    return "CORE"


def enrich_edge_priority(frame: pd.DataFrame) -> pd.DataFrame:
    """Add practical price/line-room diagnostics to supported edge rows."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(frame.columns) if isinstance(frame, pd.DataFrame) else [])

    out = frame.copy()
    out["regime_min_edge"] = out.apply(_regime_min_edge, axis=1)
    out["line_cushion_points"] = out.apply(
        lambda row: (
            max(0.0, float(row["edge"]) - float(row["regime_min_edge"]))
            if _finite(row.get("edge")) and _finite(row.get("regime_min_edge"))
            else np.nan
        ),
        axis=1,
    )
    out["bet_to_line"] = out.apply(
        lambda row: (
            _bet_to_spread_line(
                row.get("line"),
                row.get("edge"),
                row.get("regime_min_edge"),
            )
            if str(row.get("market") or "").lower() == "spread"
            else np.nan
        ),
        axis=1,
    )
    out["price_cushion_pp"] = pd.to_numeric(
        out.get("conservative_price_margin"), errors="coerce"
    ) * 100.0
    out["edge_priority"] = out.apply(classify_edge_priority, axis=1)

    rank = {
        "ROBUST_CORE": 0,
        "CORE": 1,
        "CORE_LINE_THIN": 2,
        "CORE_PRICE_THIN": 3,
        "CORE_BOTH_THIN": 4,
        "PLAUSIBLE_PRICE": 5,
        "PARENT_ONLY": 6,
        "WATCH_ONLY": 7,
        "REJECT_OVERPRICED": 8,
        "REJECT_CONTRAINDICATED": 9,
    }
    out["_priority_rank"] = out["edge_priority"].map(rank).fillna(99)
    out["_price_cushion_sort"] = pd.to_numeric(
        out.get("conservative_price_margin"), errors="coerce"
    ).fillna(-999.0)
    out["_line_cushion_sort"] = pd.to_numeric(
        out.get("line_cushion_points"), errors="coerce"
    ).fillna(-999.0)
    out["_ev_sort"] = pd.to_numeric(out.get("ev"), errors="coerce").fillna(-999.0)
    out = (
        out.sort_values(
            [
                "_priority_rank",
                "_price_cushion_sort",
                "_line_cushion_sort",
                "_ev_sort",
            ],
            ascending=[True, False, False, False],
        )
        .drop(
            columns=[
                "_priority_rank",
                "_price_cushion_sort",
                "_line_cushion_sort",
                "_ev_sort",
            ]
        )
        .reset_index(drop=True)
    )
    return out


def actionable_priority_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Return supported rows with enough price/line room for priority monitoring."""

    ranked = enrich_edge_priority(frame)
    if ranked.empty:
        return ranked
    return ranked[
        ranked["edge_priority"].isin({"ROBUST_CORE", "CORE"})
    ].reset_index(drop=True)
