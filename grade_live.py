#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from harbin.data import SportsDataVerseClient
from harbin.grading import grade_prediction_history


def main():
    report=grade_prediction_history(SportsDataVerseClient(),"history","reports")
    Path("outputs").mkdir(exist_ok=True); Path("docs").mkdir(exist_ok=True)
    Path("outputs/live_performance.json").write_text(json.dumps(report,indent=2)); Path("docs/live_performance.json").write_text(json.dumps(report,indent=2))
    for name in ("tier_performance.json","tier_performance.csv","tier_validation_clean.csv","live_graded_tiers.csv"):
        src=Path("reports")/name
        if src.exists():
            (Path("outputs")/name).write_text(src.read_text()); (Path("docs")/name).write_text(src.read_text())
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
