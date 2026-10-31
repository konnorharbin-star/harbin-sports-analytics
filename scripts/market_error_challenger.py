"""Market-error probability challenger for CFB. RESEARCH ONLY; no wagers.

Fit adjustments to a *paired, same-game* no-vig market probability and
independent pregame forecast. The training target is (home_won - market_p).
All basis functions are antisymmetric when the home/away orientation flips.

2023 = training, 2024 = model/regularization selection, 2025 = once-only
retrospective evaluation. These seasons have previously been inspected;
there is NO pristine 2025 holdout and NO validated betting edge.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

TRAIN, TUNE, EVAL = 2023, 2024, 2025
FAMILIES = ("disagreement", "close_disagreement", "favorite_bias", "all")
PENALTIES = (1.0, 10.0, 100.0)
MIN_TRAIN, MIN_TUNE, MIN_EVAL = 200, 200, 200
MIN_TUNE_GAIN = 0.00025


def _prob(value):
    try:
        v = float(value)
        return v if math.isfinite(v) and 0 < v < 1 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _american_to_p(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(v) or abs(v) < 100:
        return None
    return -v / (-v + 100) if v < 0 else 100 / (100 + v)


def _market_p(row):
    p = _prob(row.get("no_vig_probability"))
    if p is not None:
        return p
    p = _prob(row.get("consensus_home_novig_probability"))
    if p is not None:
        return p
    home = _american_to_p(row.get("home_ml"))
    away = _american_to_p(row.get("away_ml"))
    return home / (home + away) if home and away else None


def load_paired(path: Path):
    required = {"game_id", "season", "week", "model_probability", "no_vig_probability", "result"}
    rows, keys = [], set()
    with path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not required.issubset(rd.fieldnames or []):
            raise ValueError(f"Missing paired columns: {sorted(required - set(rd.fieldnames or []))}")
        for r in rd:
            if r.get("market_type") != "moneyline" or r.get("side") != "home":
                raise ValueError("Research expects home-oriented, paired moneyline rows")
            key = str(r["game_id"]).strip()
            if not key or key in keys:
                raise ValueError("Blank or duplicate game_id")
            keys.add(key)
            m = _prob(r["no_vig_probability"])
            independent = _prob(r["model_probability"])
            if m is None or independent is None or r["result"] not in ("win", "loss"):
                raise ValueError("Bad market/model probability or result")
            year, week = int(r["season"]), int(r["week"])
            if year < 2000 or not 1 <= week <= 18:
                raise ValueError("Invalid season/week")
            rows.append(dict(game_id=key, season=year, week=week, market=m,
                             independent=independent, y=int(r["result"] == "win")))
    return rows


def basis(row, family):
    m, independent = row["market"], row["independent"]
    d = independent - m
    close = 1 - abs(2 * m - 1)
    if family == "disagreement":
        return [d]
    if family == "close_disagreement":
        return [d, d * close]
    if family == "favorite_bias":
        return [m - .5, (m - .5) * close]
    if family == "all":
        return [d, d * close, (m - .5) * close]
    raise ValueError(f"Unregistered feature family: {family}")


def design(rows, family):
    return np.asarray([basis(r, family) for r in rows], dtype=float)


def fit(rows, family, penalty):
    x = design(rows, family)
    y = np.asarray([r["y"] - r["market"] for r in rows], dtype=float)
    return np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)


def predict(rows, family, beta):
    if not rows:
        return np.asarray([], dtype=float)
    market = np.asarray([r["market"] for r in rows], dtype=float)
    if beta is None:
        return market
    return np.clip(market + design(rows, family) @ np.asarray(beta), 0.001, 0.999)


def metrics(rows, probs):
    if not rows:
        raise ValueError("Cannot score an empty sample")
    y = np.asarray([r["y"] for r in rows], dtype=float)
    p = np.asarray(probs, dtype=float)
    if len(p) != len(y) or not np.all(np.isfinite(p)):
        raise ValueError("Predictions must be finite and match games")
    return {
        "games": len(y),
        "correct": int(np.sum((p >= .5) == y)),
        "accuracy": float(np.mean((p >= .5) == y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p)))),
    }


def experiment(rows):
    train = [r for r in rows if r["season"] == TRAIN]
    tune = [r for r in rows if r["season"] == TUNE]
    evaluate = [r for r in rows if r["season"] == EVAL]
    sizes = {str(y): len(z) for y, z in [(TRAIN, train), (TUNE, tune), (EVAL, evaluate)]}
    if min(len(train), len(tune), len(evaluate)) < min(MIN_TRAIN, MIN_TUNE, MIN_EVAL):
        raise ValueError(f"Not enough distinct matched games by season: {sizes}")

    tune_market = metrics(tune, predict(tune, None, None))
    candidates = []
    for family in FAMILIES:
        for lam in PENALTIES:
            beta = fit(train, family, lam)
            score = metrics(tune, predict(tune, family, beta))
            candidates.append({"family": family, "penalty": lam, "metrics": score})

    # Select only on 2024. Ties and non-improvements choose the market.
    best = min(candidates, key=lambda c: (c["metrics"]["brier"], -c["penalty"], c["family"]))
    improvement = tune_market["brier"] - best["metrics"]["brier"]
    selected = (best if improvement >= MIN_TUNE_GAIN
                and best["metrics"]["accuracy"] >= tune_market["accuracy"] else None)

    # After hyperparameters are frozen, refit on ALL pre-2025 data.
    developmental = train + tune
    beta = fit(developmental, selected["family"], selected["penalty"]) if selected else None
    market_eval = metrics(evaluate, predict(evaluate, None, None))
    challenger_eval = metrics(evaluate, predict(evaluate, selected["family"], beta) if selected
                              else predict(evaluate, None, None))
    return {
        "spec": "cfb_market_error_challenger_v1",
        "status": "RESEARCH_ONLY_NOT_PROVEN",
        "seasons": {"train": TRAIN, "tune": TUNE, "evaluation": EVAL},
        "sample_counts": sizes,
        "tuning_market": tune_market,
        "candidate_tuning_scores": candidates,
        "selected_on_tuning_only": None if not selected else {
            "family": selected["family"], "penalty": selected["penalty"],
            "tune_brier_improvement": improvement, "tune_accuracy": best["metrics"]["accuracy"]},
        "evaluation_market": market_eval,
        "evaluation_challenger": challenger_eval,
        "evaluation_challenger_minus_market_accuracy":
            challenger_eval["accuracy"] - market_eval["accuracy"],
        "evaluation_market_minus_challenger_brier":
            market_eval["brier"] - challenger_eval["brier"],
        "fitted_coefficients": beta.tolist() if beta is not None else None,
        "promotion_eligible": False,
        "bets_authorized": False,
        "limitations": [
            "2023-2025 archive was inspected previously; 2025 is a retrospective evaluation, not pristine prospective evidence.",
            "Archived opening/final-fallback moneylines lack verified bookmaker quote timestamps.",
            "The model may exactly reproduce the market when tuning fails; this is an intended zero-adjustment fallback.",
            "No wagering, ROI, CLV, or market-beating claim follows from this experiment."
        ],
    }


def shadow_live(report, path: Path, audit: Path, output: Path):
    # No publication on stale snapshots. Live shadow NEVER alters the official winner or bet gate.
    snapshot = json.loads(audit.read_text(encoding="utf-8"))
    ts = datetime.fromisoformat(snapshot["generated_at"].replace("Z", "+00:00"))
    age = (datetime.now(UTC) - ts).total_seconds()
    if age < 0 or age > 6 * 3600:
        raise ValueError("Refusing shadow predictions from stale/future model output")
    selected = report["selected_on_tuning_only"]
    beta = report["fitted_coefficients"]
    rows, seen = [], set()
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gid = r.get("game_id", "")
            if not gid or gid in seen:
                raise ValueError("Duplicate live game")
            seen.add(gid)
            kickoff = datetime.fromisoformat(r["date"].replace("Z", "+00:00"))
            if kickoff <= datetime.now(UTC):
                continue
            market = _market_p(r)
            independent = (_prob(r.get("calibrated_home_probability"))
                           or _prob(r.get("model_home_win_probability")))
            if market is None or independent is None:
                continue
            row = {"market": market, "independent": independent}
            challenger = float(predict([row], selected["family"], beta)[0]) if selected else market
            rows.append({"game_id": gid, "kickoff": r["date"],
                         "home_team": r.get("home_team"), "away_team": r.get("away_team"),
                         "market_home_probability": round(market, 8),
                         "independent_home_probability": round(independent, 8),
                         "challenger_home_probability": round(challenger, 8),
                         "market_winner": r.get("home_team") if market >= .5 else r.get("away_team"),
                         "shadow_winner": r.get("home_team") if challenger >= .5 else r.get("away_team"),
                         "research_only": True, "market_quote_origin_verified": False})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        fields = ["game_id", "kickoff", "home_team", "away_team", "market_home_probability",
                  "independent_home_probability", "challenger_home_probability",
                  "market_winner", "shadow_winner", "research_only", "market_quote_origin_verified"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("reports/market_error_challenger.json"))
    p.add_argument("--live", type=Path)
    p.add_argument("--live-audit", type=Path, default=Path("docs/audit_snapshot.json"))
    p.add_argument("--shadow-out", type=Path, default=Path("outputs/market_error_shadow.csv"))
    args = p.parse_args()
    rows = load_paired(args.historical)
    report = experiment(rows)
    report["historical_sha256"] = hashlib.sha256(args.historical.read_bytes()).hexdigest()
    if args.live:
        report["live_shadow_rows"] = shadow_live(report, args.live, args.live_audit, args.shadow_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "sample_counts",
          "selected_on_tuning_only", "evaluation_market", "evaluation_challenger")}, indent=2))


if __name__ == "__main__":
    main()
