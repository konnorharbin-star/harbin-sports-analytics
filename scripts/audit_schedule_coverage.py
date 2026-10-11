#!/usr/bin/env python3
"""Audit every source-schedule game against the published pregame slate.

Started/completed games are reported, not quietly treated as missing bets.
Never reintroduce already-started games into pregame predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from harbin.data import SportsDataVerseClient, _bool, fbs_schedule_mask


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
    if source.empty:
        raise RuntimeError("No FBS fixtures found for season/week")
    if not fbs_schedule_mask(source).all():
        raise RuntimeError("Non-FBS matchup in claimed full FBS slate")
    source["fbs_matchup_scope"] = source.apply(
        lambda g: "FBS_VS_FBS" if str(g["home_division"]).strip().lower() == "fbs"
        and str(g["away_division"]).strip().lower() == "fbs"
        else "FBS_VS_NON_FBS_UNVALIDATED", axis=1
    )
    first_seen_path = root / "history" / "first_seen_pregame_predictions.csv"
    first_seen = set()
    if first_seen_path.exists():
        archive = pd.read_csv(first_seen_path, dtype={"game_id": str})
        if {"season", "game_id", "first_seen_utc", "date"}.issubset(archive.columns):
            archive = archive.loc[pd.to_numeric(archive["season"], errors="coerce") == season]
            first_seen = set(archive.loc[
                pd.to_datetime(archive["first_seen_utc"], utc=True, errors="coerce")
                < pd.to_datetime(archive["date"], utc=True, errors="coerce"),
                "game_id",
            ].astype(str))
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
                     "fbs_matchup_scope": game["fbs_matchup_scope"],
                     "first_seen_pregame_archived": game_id in first_seen,
                     "status": status, "model_cutoff_utc": cutoff.isoformat()})
    report = pd.DataFrame(rows, columns=[
        "game_id", "away_team", "home_team", "kickoff_utc",
        "fbs_matchup_scope", "first_seen_pregame_archived", "status", "model_cutoff_utc"])
    if report["game_id"].duplicated().any():
        raise RuntimeError("Duplicate source schedule game IDs")
    unrecognized = predicted_ids - set(report["game_id"])
    missing = report[report.status == "MISSING_UPCOMING"]
    report.to_csv(out / "schedule_coverage_audit.csv", index=False)
    counts = report.status.value_counts().to_dict()
    summary = {"season": season, "week": week, "model_cutoff_utc": cutoff.isoformat(),
               "scope": "EVERY_FBS_INVOLVED_REGULAR_SEASON_MATCHUP",
               "total_scheduled_fbs_games": int(len(source)),
               "fbs_vs_fbs_games": int((source.fbs_matchup_scope == "FBS_VS_FBS").sum()),
               "fbs_vs_non_fbs_games": int(
                   (source.fbs_matchup_scope == "FBS_VS_NON_FBS_UNVALIDATED").sum()),
               "remaining_pregame_games": int((report.status == "PREDICTED").sum() + len(missing)),
               "remaining_pregame_coverage_complete": not bool(len(missing) or unrecognized),
               "previously_archived_before_kickoff": int(
                   report.first_seen_pregame_archived.sum()),
               "started_or_completed_without_archived_pregame": int(
                   (report.status.isin(["STARTED_EXCLUDED", "COMPLETED_EXCLUDED"])
                    & ~report.first_seen_pregame_archived).sum()),
               "counts": counts, "missing_upcoming": missing.to_dict("records"),
               "predicted_not_in_source": sorted(unrecognized)}
    (out / "schedule_coverage_audit.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    if len(missing) or unrecognized:
        raise SystemExit("Schedule coverage failure: upcoming game missing from predictions")
    return summary


if __name__ == "__main__":
    audit()
