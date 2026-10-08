"""Descriptive forward probability calibration for independently confirmed research.

No model fits, parameter updates, wager placement or paid subscriptions.
Only score strictly corroborated, frozen pregame forecasts.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

MIN_ECE_SAMPLE = 100
MIN_BIN_SAMPLE = 10
BIN_EDGES = ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0))


def _finite_probability(value: object) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return p if math.isfinite(p) and 0 < p < 1 else None


def _one_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Win/loss events only. Pushes are excluded and make these *descriptive*
    # probability scores, not proof of calibrated push-adjusted bet prices.
    pairs = []
    games = set()
    omitted_pushes = 0
    for row in rows:
        if row.get("independent_score_verified") is not True:
            continue
        if row.get("result") == "PUSH":
            omitted_pushes += 1
            continue
        if row.get("result") not in ("WIN", "LOSS"):
            continue
        p = _finite_probability(row.get("model_probability"))
        if p is None:
            continue
        pairs.append((p, 1 if row["result"] == "WIN" else 0))
        games.add(str(row.get("game_id") or ""))
    n = len(pairs)
    if not n:
        return {
            "graded_probabilities": 0, "distinct_games": 0,
            "omitted_pushes": omitted_pushes,
            "brier": None, "log_loss": None,
            "mean_predicted_probability": None, "observed_win_rate": None,
            "calibration_ece": None, "reliability_bins": [],
            "status": "NO_VERIFIED_SAMPLE",
            "profitability_proven": False,
        }
    brier = sum((p-y)**2 for p,y in pairs)/n
    ll = -sum(y*math.log(p)+(1-y)*math.log1p(-p) for p,y in pairs)/n
    bins = []
    ece = 0.0
    for a,b in BIN_EDGES:
        values = [(p,y) for p,y in pairs if a <= p < b or (b == 1.0 and p == 1.0)]
        if len(values) < MIN_BIN_SAMPLE:
            continue
        mean_p = sum(p for p,_ in values)/len(values)
        observed = sum(y for _,y in values)/len(values)
        bins.append({
            "lower": a, "upper": b, "count": len(values),
            "mean_probability": mean_p, "observed_win_rate": observed,
        })
        ece += len(values)/n * abs(mean_p-observed)
    return {
        "graded_probabilities": n,
        "distinct_games": len(games),
        "omitted_pushes": omitted_pushes,
        "brier": brier,
        "log_loss": ll,
        "mean_predicted_probability": sum(p for p,_ in pairs)/n,
        "observed_win_rate": sum(y for _,y in pairs)/n,
        "calibration_ece": ece if n >= MIN_ECE_SAMPLE
            and sum(x["count"] for x in bins) == n else None,
        "reliability_bins": bins,
        "status": "DESCRIPTIVE_ONLY" if n >= MIN_ECE_SAMPLE else "INSUFFICIENT_SAMPLE",
        "profitability_proven": False,
    }


def forward_calibration(
    graded: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score independently verified outcomes, grouped by league and market."""
    scores: dict[str, Any] = {}
    for league in ("NFL", "CFB"):
        league_rows = [
            row for row in graded if row.get("league") == league
            and row.get("independent_score_verified") is True
        ]
        markets: dict[str, Any] = {}
        for market in ("moneyline", "spread", "total"):
            markets[market] = _one_group([
                row for row in league_rows if row.get("market") == market
            ])
        scores[league] = {
            "all_markets": _one_group(league_rows),
            "by_market": markets,
        }
    return {
        "metric_scope": "Independently corroborated, first-seen pregame research grades only",
        "limitations": (
            "Pushes are omitted from binary scores. Repeated markets on one game "
            "are correlated; market win probability may include push mass. "
            "Reliability is descriptive, NOT a verified profitable edge or a calibrated "
            "trading probability. No probability refitting or stake authorization."
        ),
        "min_ece_observations": MIN_ECE_SAMPLE,
        "results": scores,
    }
