"""Full FBS weekly-slate inclusion, with fail-closed unvalidated opponents.

Predictions are required for games containing at least one FBS school.
The current statistical model is trained on FBS-vs-FBS games, so an FBS
game against a non-FBS opponent is research-only with NO BET authorization.
"""
from __future__ import annotations

import pandas as pd


def classify(game):
    home = str(getattr(game, "home_division", "") or "").strip().upper()
    away = str(getattr(game, "away_division", "") or "").strip().upper()
    if home == "FBS" and away == "FBS":
        return "FBS_VS_FBS"
    if (home == "FBS") != (away == "FBS"):
        return "FBS_VS_NON_FBS_UNVALIDATED"
    raise ValueError("Fixture is not verified to include an FBS team")


def apply_slate_scope(pred: pd.DataFrame, games):
    """One row per upcoming scheduled FBS match, without promoting FCS edges."""
    ids = [str(g.game_id) for g in games]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game IDs in upcoming FBS slate")
    if pred.empty:
        if ids:
            raise ValueError("Missing projections for upcoming FBS games")
        return pred.copy()
    actual = pred["game_id"].astype(str)
    if actual.duplicated().any() or set(actual) != set(ids):
        raise ValueError("Predictions do not cover all upcoming FBS game IDs")
    scope = {str(g.game_id): classify(g) for g in games}
    result = pred.copy()
    result["fbs_matchup_scope"] = actual.map(scope)
    result["fbs_model_validation"] = result["fbs_matchup_scope"].map({
        "FBS_VS_FBS": "STANDARD_RESEARCH_GATES",
        "FBS_VS_NON_FBS_UNVALIDATED": "NO_BET_UNVALIDATED_OPPONENT_CLASS",
    })
    unvalidated = result["fbs_matchup_scope"].eq("FBS_VS_NON_FBS_UNVALIDATED")
    # The model's ratings/history only cover FBS/FBS. Still show scores,
    # with explicit status and no recommendation, stake, or supported edge.
    for key, value in {
        "quant_signal": "PASS",
        "research_signal": "PASS",
        "production_signal": "PASS",
        "stake_units": 0.0,
        "research_stake_units": 0.0,
        "edge_regime_candidate": False,
        "edge_regime_status": "UNVALIDATED_NON_FBS_OPPONENT",
        "edge_reliability_status": "UNVALIDATED_NON_FBS_OPPONENT",
        "edge_selection_override": False,
    }.items():
        result.loc[unvalidated, key] = value
    return result
