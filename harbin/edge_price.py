"""Conservative price checks for historically supported betting edges."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .market import american_implied, fair_american


def wilson_lower_bound(wins: int, losses: int, z: float = 1.96) -> float | None:
    """Two-sided 95% Wilson lower bound for a binary hit rate."""

    wins = int(wins or 0)
    losses = int(losses or 0)
    n = wins + losses
    if n <= 0:
        return None
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - radius)


def price_evidence(odds: object, stats: Mapping[str, object] | None) -> dict[str, object]:
    """Compare current break-even probability with historical evidence.

    CONFIRMED means the 95% Wilson lower bound exceeds the current price's break-even
    probability. PLAUSIBLE means the point estimate clears break-even but its lower
    confidence bound does not. OVERPRICED means even the historical point estimate
    fails to clear the current break-even price.

    This diagnostic never modifies model probability or EV.
    """

    base = {
        "price_evidence_status": "UNKNOWN",
        "price_evidence_sample": 0,
        "historical_price_win_rate": None,
        "historical_price_wilson_lower": None,
        "current_break_even_probability": None,
        "historical_price_margin": None,
        "conservative_price_margin": None,
        "historical_fair_odds_lower_bound": None,
    }
    try:
        price = float(odds)
    except (TypeError, ValueError):
        return base
    if not math.isfinite(price):
        return base

    stats = stats or {}
    wins = int(stats.get("wins", 0) or 0)
    losses = int(stats.get("losses", 0) or 0)
    n = wins + losses
    if n <= 0:
        return base

    win_rate = wins / n
    lower = wilson_lower_bound(wins, losses)
    if lower is None:
        return base

    break_even = float(american_implied(price))
    point_margin = win_rate - break_even
    conservative_margin = lower - break_even
    if conservative_margin > 0:
        status = "CONFIRMED"
    elif point_margin > 0:
        status = "PLAUSIBLE"
    else:
        status = "OVERPRICED"

    return {
        "price_evidence_status": status,
        "price_evidence_sample": n,
        "historical_price_win_rate": win_rate,
        "historical_price_wilson_lower": lower,
        "current_break_even_probability": break_even,
        "historical_price_margin": point_margin,
        "conservative_price_margin": conservative_margin,
        "historical_fair_odds_lower_bound": fair_american(lower),
    }
