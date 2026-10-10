"""Reconstruct paired archive prices for exploratory CFB calibration diagnostics.

Uses the existing canonical archive matcher. No assumed odds and no executable
entry-price or pristine historical holdout claim. Does not alter backtest policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from harbin.backtest_runtime import CanonicalArchiveMarketStore
from scripts.market_first_experiment import decimal, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path, default=Path("reports/backtest_predictions.csv")
    )
    parser.add_argument("--cache", default="/tmp/harbin-cfb-market-diagnostic")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    store = CanonicalArchiveMarketStore(args.cache)
    records, omitted = [], 0
    for row in csv.DictReader(args.predictions.open()):
        game = SimpleNamespace(
            game_id=row["game_id"],
            home_team=row["home_team"],
            away_team=row["away_team"],
        )
        quote = store.quote(game)
        model = probability(row.get("home_win_probability"))
        if not quote or model is None or float(row["actual_margin_home"]) == 0:
            omitted += 1
            continue
        home, away = (
            decimal(quote.get("open_home_ml")),
            decimal(quote.get("open_away_ml")),
        )
        if home is None or away is None:
            omitted += 1
            continue
        records.append(
            {
                "market_type": "moneyline",
                "game_id": row["game_id"],
                "season": row["season"],
                "week": row["week"],
                "side": "home",
                "model_probability": model,
                "no_vig_probability": (1 / home) / (1 / home + 1 / away),
                "result": "win" if float(row["actual_margin_home"]) > 0 else "loss",
                "home_odds": quote["open_home_ml"],
                "away_odds": quote["open_away_ml"],
                "book": quote["book"],
                "entry_timestamp_verified": False,
                "price_stage": "archive_opening_fields_or_final_fallback_unverified",
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No complete paired historical market sample")
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "games": len(records),
        "omitted_missing_pair_or_probability_or_tie": omitted,
        "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(
            (Path(args.cache) / "cfb_line_odds.csv.gz").read_bytes()
        ).hexdigest(),
        "archive_source": "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/csv/cfb_line_odds.csv.gz",
        "verified_entry_timestamps": 0,
        "economic_evidence_eligible": False,
    }
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
