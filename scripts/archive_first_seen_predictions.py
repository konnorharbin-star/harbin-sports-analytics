#!/usr/bin/env python3
"""Preserve first-seen pre-kickoff projections across dashboard refreshes.

The immutable ledger never retroactively creates a prediction for a game that
was absent before kickoff. It is distinct from the mutable latest dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def archive(root=Path(".")):
    output = root / "outputs"
    metadata = sorted(output.glob("cfb_model_*_metadata.json"),
                      key=lambda p: p.stat().st_mtime)
    if not metadata:
        raise RuntimeError("Missing model metadata")
    meta = json.loads(metadata[-1].read_text())
    season, week = int(meta["season"]), int(meta["week"])
    csv = output / f"cfb_model_{season}_week{week}.csv"
    current = pd.read_csv(csv, dtype={"game_id": str})
    required = {"game_id", "date", "away_team", "home_team", "away_score",
                "home_score", "model_margin_home", "model_total",
                "win_probability"}
    if not required.issubset(current):
        raise ValueError(f"Prediction missing columns: {required - set(current)}")
    cutoff = pd.to_datetime(meta["pregame_filter"]["cutoff_utc"], utc=True)
    kickoff = pd.to_datetime(current["date"], utc=True, errors="coerce")
    if kickoff.isna().any() or (kickoff <= cutoff).any():
        raise ValueError("Cannot archive a post-kickoff or invalid prediction")
    if current.game_id.duplicated().any():
        raise ValueError("Duplicate game ids in predictions")
    columns = ["season", "week", "first_seen_utc", "game_id", "date",
               "away_team", "home_team", "away_score", "home_score",
               "model_margin_home", "model_total", "win_probability",
               "market_spread_home", "market_total", "quant_signal"]
    current.insert(0, "first_seen_utc", cutoff.isoformat())
    keep = current[[col for col in columns if col in current.columns]].copy()
    # Live model CSV already has season/week. Assign rather than insert, so
    # archiving also works with the actual production schema.
    keep["season"] = season
    keep["week"] = week
    keep = keep[["season", "week"] + [c for c in keep.columns if c not in {"season", "week"}]]
    path = root / "history" / "first_seen_pregame_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pd.read_csv(path, dtype={"game_id": str})
        if previous.duplicated(["season", "game_id"]).any():
            raise ValueError("Duplicate immutable pregame ledger keys")
        keys = set(zip(previous.season.astype(int), previous.game_id.astype(str)))
        keep = keep.loc[~keep.game_id.map(lambda gid: (season, str(gid)) in keys)]
        combined = pd.concat([previous, keep], ignore_index=True)
    else:
        combined = keep
    combined.to_csv(path, index=False)
    summary = {"season": season, "week": week, "newly_archived": len(keep),
               "total_archived": len(combined), "ledger": str(path)}
    (output / "pregame_archive_status.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    archive()
