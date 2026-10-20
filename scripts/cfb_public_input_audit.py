"""Read-only current FBS source audit; never writes forecasts or history."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from harbin.context import ContextStore
from harbin.data import SportsDataVerseClient
from harbin.public_input_quality import write_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("reports/current_input_audit"))
    args = parser.parse_args()
    client = SportsDataVerseClient()
    season, week = client.detect() if args.season is None or args.week is None else (args.season, args.week)
    season, week = args.season or season, args.week or week
    frame = client.season_frame(season)
    rows = frame[(frame.week == week) & (~frame.completed.astype(str).str.lower().isin({"true", "1"}))].copy()
    rows = rows[pd.to_datetime(rows.start_date, utc=True) > pd.Timestamp.now(tz="UTC")]
    rows = rows.sort_values(["start_date", "game_id"])
    if args.limit:
        rows = rows.head(args.limit)
    predictions = rows.rename(columns={"start_date": "date"})[["game_id", "season", "week", "date", "home_team", "away_team"]].copy()
    context = ContextStore(season, schedule_frame=frame, cache_dir=args.output / "cache")
    observed, metadata = context.attach(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "source_games.json"
    source.write_text(predictions.to_json(orient="records", indent=2), encoding="utf-8")
    summary = write_quality(source, context.public_input_records, generated_at=datetime.now(timezone.utc).isoformat(), output_dirs=[args.output])
    (args.output / "context_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"season": season, "week": week, **summary, "errors": metadata.get("errors")}, indent=2))
    # A malformed source must not silently earn a passing audit. Genuine gaps
    # in availability/weather remain visible without inventing missing facts.
    if summary["games_checked"] == 0 or summary["exact_events_verified"] != summary["games_checked"]:
        raise SystemExit("Current slate has unverified event identities; inspect the report")


if __name__ == "__main__":
    main()
