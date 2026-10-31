"""As-of, opponent-form probability challenger for college football (research only).

The target is the error in a PAIR of archived same-game no-vig moneyline quotes,
not an executable bet. Historical model forecasts and quote timestamps are not
independently certified to be immutable pregame observations.

Train = 2023; tune/preselect = 2024; retrospective evaluate = 2025.
No evaluation-season outcome can select a feature, weight, or hyperparameter.
All team-form features for any target week use earlier COMPLETED weeks only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.market_error_challenger import _prob

TRAIN, TUNE, EVAL = 2023, 2024, 2025
MIN_SAMPLE = 200
FAMILIES = (
    "market_cal", "model_only", "market_form3", "score_form3",
    "form3", "form6", "model_form3", "model_form6",
    "model_close_form", "combined", "total_context",
)
PENALTIES = (0.03, 0.3, 3.0, 30.0)
MIN_TUNE_GAIN = 0.0003
SHRINK = 4.0


def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def load_rows(path: Path):
    fields = {"market_type", "game_id", "season", "week", "home_team",
              "away_team", "pred_margin_home", "pred_total", "actual_margin_home",
              "model_probability", "no_vig_probability", "result", "side"}
    rows, ids = [], set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not fields.issubset(reader.fieldnames or ()):
            raise ValueError("Missing chronological football/moneyline fields")
        for rec in reader:
            if rec["market_type"] != "moneyline" or rec["side"] != "home":
                raise ValueError("Require one home-oriented moneyline pair per game")
            gid = rec["game_id"].strip()
            if not gid or gid in ids:
                raise ValueError("Missing or duplicated historical game ID")
            ids.add(gid)
            home, away = rec["home_team"].strip(), rec["away_team"].strip()
            m, q = _prob(rec["no_vig_probability"]), _prob(rec["model_probability"])
            margin, total, actual = (finite(rec[k]) for k in
                                     ("pred_margin_home", "pred_total", "actual_margin_home"))
            if (not home or not away or home == away or m is None or q is None
                    or any(x is None for x in (margin, total, actual))
                    or rec["result"] not in ("win", "loss")):
                raise ValueError("Bad team, game, probability, or score fields")
            if total < abs(margin) or total < 0 or actual == 0:
                raise ValueError("Invalid football projected score or tie")
            y = int(rec["result"] == "win")
            if y != int(actual > 0):
                raise ValueError("Final-score outcome inconsistent with market result")
            season, week = int(rec["season"]), int(rec["week"])
            if season < 2000 or week < 1 or week > 25:
                raise ValueError("Invalid chronological season/week")
            rows.append({"game_id": gid, "season": season, "week": week,
                         "home_team": home, "away_team": away,
                         "market": m, "independent": q,
                         "margin": margin, "total": total,
                         "actual_margin": actual, "y": y})
    return rows


def asof_features(rows):
    """Freeze entire week BEFORE any of its results can update prior-team state."""
    by_week = defaultdict(list)
    for row in rows:
        by_week[(row["season"], row["week"])].append(row)
    out = []
    prior_season = None
    state = defaultdict(list)

    def prior(team, n, key):
        x = state[team][-n:]
        return sum(z[key] for z in x) / (len(x) + SHRINK)

    for (season, week), items in sorted(by_week.items()):
        if season != prior_season:
            state.clear()  # no silent previous-year roster/coach assumptions
            prior_season = season
        frozen = []
        for row in sorted(items, key=lambda r: r["game_id"]):
            h, a = row["home_team"], row["away_team"]
            feats = {}
            for n in (3, 6):
                feats[f"market_form{n}"] = prior(h, n, "market_err") - prior(a, n, "market_err")
                feats[f"score_form{n}"] = (
                    prior(h, n, "margin_err") - prior(a, n, "margin_err")
                ) / 15.0
            frozen.append({**row, **feats})
        out.extend(frozen)
        # Only AFTER all target-week features have been frozen.
        for row in items:
            market_err = row["y"] - row["market"]
            score_err = row["actual_margin"] - row["margin"]
            state[row["home_team"]].append({"market_err": market_err,
                                             "margin_err": score_err})
            state[row["away_team"]].append({"market_err": -market_err,
                                             "margin_err": -score_err})
    return out


def basis(row, family):
    p = row["market"]
    disagreement = row["independent"] - p
    close = 1 - abs(2 * p - 1)
    r3, r6 = row["market_form3"], row["market_form6"]
    s3, s6 = row["score_form3"], row["score_form6"]
    groups = {
        "market_cal": [(p - .5) * close],
        "model_only": [disagreement],
        "market_form3": [r3],
        "score_form3": [s3],
        "form3": [r3, s3],
        "form6": [r6, s6],
        "model_form3": [disagreement, r3, s3],
        "model_form6": [disagreement, r6, s6],
        "model_close_form": [disagreement * close, r6 * close, s6 * close],
        "combined": [disagreement, disagreement * close,
                     r3, s3, (p - .5) * close],
        "total_context": [disagreement, disagreement * (row["total"] - 55) / 40,
                          r3, s3],
    }
    if family not in groups:
        raise ValueError("Unknown, non-frozen research feature family")
    return groups[family]


def fit(rows, family, penalty):
    x = np.asarray([basis(r, family) for r in rows], dtype=float)
    y = np.asarray([r["y"] - r["market"] for r in rows], dtype=float)
    return np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)


def forecast(rows, family=None, beta=None):
    market = np.asarray([r["market"] for r in rows])
    if family is None:
        return market
    x = np.asarray([basis(r, family) for r in rows])
    return np.clip(market + x @ np.asarray(beta), .001, .999)


def evaluate(rows, probabilities):
    y = np.asarray([r["y"] for r in rows], dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(y) or not len(y) or not np.all(np.isfinite(p)):
        raise ValueError("Empty, malformed or misaligned probability evaluation")
    correct = (p >= .5).astype(int) == y
    # Confidence-pool ranking is evaluated WITHIN same-week games (1..N).
    by_week = defaultdict(list)
    for i, row in enumerate(rows):
        by_week[(row["season"], row["week"])].append(i)
    points, max_points = 0, 0
    for indexes in by_week.values():
        sorted_ids = sorted(indexes, key=lambda i: (abs(p[i] - .5), rows[i]["game_id"]))
        points += sum(rank for rank, i in enumerate(sorted_ids, start=1) if correct[i])
        n = len(indexes)
        max_points += n * (n + 1) // 2
    output = {"games": len(y),
              "correct": int(np.sum(correct)),
              "accuracy": float(np.mean(correct)),
              "brier": float(np.mean((p - y) ** 2)),
              "log_loss": float(np.mean(-y * np.log(p) - (1-y) * np.log1p(-p))),
              "confidence_points": int(points),
              "confidence_points_possible": int(max_points)}
    close_games = np.abs(np.asarray([r["market"] for r in rows]) - .5) <= .15
    if np.any(close_games):
        output["close_games"] = int(close_games.sum())
        output["close_accuracy"] = float(np.mean(correct[close_games]))
        output["close_brier"] = float(np.mean((p[close_games] - y[close_games]) ** 2))
    else:
        output["close_games"] = 0
    return output


def experiment(rows):
    enriched = asof_features(rows)
    train = [r for r in enriched if r["season"] == TRAIN]
    tune = [r for r in enriched if r["season"] == TUNE]
    evaluation = [r for r in enriched if r["season"] == EVAL]
    if min(len(train), len(tune), len(evaluation)) < MIN_SAMPLE:
        raise ValueError("Insufficient season-separated matched games")
    baseline_tune = evaluate(tune, forecast(tune))
    trials = []
    for family in FAMILIES:
        for penalty in PENALTIES:
            b = fit(train, family, penalty)
            m = evaluate(tune, forecast(tune, family, b))
            trials.append({"family": family, "penalty": penalty,
                           "tune_accuracy": m["accuracy"],
                           "tune_brier": m["brier"],
                           "tune_log_loss": m["log_loss"],
                           "tune_close_brier": m.get("close_brier")})
    selected = min(trials, key=lambda r: (r["tune_brier"], -r["penalty"], r["family"]))
    gain = baseline_tune["brier"] - selected["tune_brier"]
    use_model = bool(gain >= MIN_TUNE_GAIN and
                     selected["tune_accuracy"] >= baseline_tune["accuracy"] and
                     selected["tune_log_loss"] <= baseline_tune["log_loss"])
    beta = (fit(train + tune, selected["family"], selected["penalty"])
            if use_model else None)
    market = evaluate(evaluation, forecast(evaluation))
    challenger = evaluate(evaluation, forecast(
        evaluation, selected["family"], beta) if use_model else forecast(evaluation))
    return {
        "spec": "cfb_asof_team_form_challenger_v1",
        "train_year": TRAIN, "tune_year": TUNE, "evaluation_year": EVAL,
        "sample_sizes": {str(k): len(v) for k, v in
                         [(TRAIN, train), (TUNE, tune), (EVAL, evaluation)]},
        "tune_market": baseline_tune,
        "tune_best_raw": selected,
        "tune_market_minus_best_brier": gain,
        "selected": {"family": selected["family"], "penalty": selected["penalty"]}
                    if use_model else None,
        "estimated_coefficients": beta.tolist() if beta is not None else None,
        "evaluation_market": market,
        "evaluation_challenger": challenger,
        "evaluation_improvement": {
            "correct_winners": challenger["correct"] - market["correct"],
            "brier_reduction": market["brier"] - challenger["brier"],
            "log_loss_reduction": market["log_loss"] - challenger["log_loss"],
            "confidence_point_gain": (challenger["confidence_points"] -
                                      market["confidence_points"]),
        },
        "all_tuning_candidates": trials,
        "research_only": True,
        "promotion_eligible": False,
        "betting_authorized": False,
        "limitations": [
            "2023-2025 results and archived model outputs were previously inspected: 2025 is NOT a pristine holdout.",
            "Archived opening-price quote-origin timestamps are unverified and sometimes use final fallbacks.",
            "The as-of feature rules freeze all games in one week before updating team residuals.",
            "No economic betting edge can be inferred from predictive Brier/accuracy changes.",
            "Forward untouched games and week-clustered uncertainty are still required.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("reports/market_form_challenger.json"))
    args = p.parse_args()
    rows = load_rows(args.historical)
    result = experiment(rows)
    result["historical_sha256"] = hashlib.sha256(args.historical.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("sample_sizes", "tune_best_raw", "tune_market_minus_best_brier",
                       "selected", "evaluation_market", "evaluation_challenger",
                       "evaluation_improvement")}, indent=2))


if __name__ == "__main__":
    main()
