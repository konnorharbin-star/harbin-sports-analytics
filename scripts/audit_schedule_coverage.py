#!/usr/bin/env python3
"""Audit every source-schedule game against the published pregame slate.

Started/completed games are reported, not quietly treated as missing bets.
Never reintroduce already-started games into pregame predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from harbin.data import SportsDataVerseClient, _bool


def audit(root=Path(".")):
    out = root / "outputs"
    reports = sorted(out.glob("cfb_model_*_metadata.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise RuntimeError("No model metadata found")
    meta = json.loads(reports[-1].read_text())
    season, week = int(meta["season"]), int(meta["week"])
    cutoff = pd.to_datetime(meta["pregame_filter"]["cutoff_utc"], utc=True)
    prediction_path = out / f"cfb_model_{season}_week{week}.csv"
    predictions = pd.read_csv(prediction_path, dtype={"game_id": str})
    predicted_ids = set(predictions["game_id"].astype(str))
    client = SportsDataVerseClient()
    frame = client.season_frame(season)
    regular = frame["season_type"].astype(str).str.lower().isin({"regular", "2"})
    source = frame.loc[regular & (pd.to_numeric(frame["week"], errors="coerce") == week)].copy()
    rows = []
    for _, game in source.iterrows():
        game_id = str(game["game_id"])
        kickoff = pd.to_datetime(game.get("start_date"), errors="coerce", utc=True)
        completed = _bool(game.get("completed", False))
        if game_id in predicted_ids:
            status = "PREDICTED"
        elif completed:
            status = "COMPLETED_EXCLUDED"
        elif pd.isna(kickoff):
            status = "INVALID_KICKOFF"
        elif kickoff <= cutoff:
            status = "STARTED_EXCLUDED"
        else:
            status = "MISSING_UPCOMING"
        rows.append({"game_id": game_id, "away_team": game.get("away_team"),
                     "home_team": game.get("home_team"), "kickoff_utc": game.get("start_date"),
                     "status": status, "model_cutoff_utc": cutoff.isoformat()})
    report = pd.DataFrame(rows, columns=[
        "game_id", "away_team", "home_team", "kickoff_utc", "status", "model_cutoff_utc"])
    if report["game_id"].duplicated().any():
        raise RuntimeError("Duplicate source schedule game IDs")
    unrecognized = predicted_ids - set(report["game_id"])
    missing = report[report.status == "MISSING_UPCOMING"]
    report.to_csv(out / "schedule_coverage_audit.csv", index=False)
    counts = report.status.value_counts().to_dict()
    summary = {"season": season, "week": week, "model_cutoff_utc": cutoff.isoformat(),
               "counts": counts, "missing_upcoming": missing.to_dict("records"),
               "predicted_not_in_source": sorted(unrecognized)}
    (out / "schedule_coverage_audit.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    if len(missing) or unrecognized:
        raise SystemExit("Schedule coverage failure: upcoming game missing from predictions")
    return summary


if __name__ == "__main__":
    audit()
