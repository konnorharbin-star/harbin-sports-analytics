#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from harbin.data import SportsDataVerseClient
from harbin.grading import grade_prediction_history
from harbin.edge_forward import grade_edge_forward_history
from harbin.edge_timing_forward import grade_timing_forward
from harbin.edge_timing_benchmarks import benchmark_timing_forward


def main():
    client=SportsDataVerseClient()
    report=grade_prediction_history(client,"history","reports")
    edge_forward=grade_edge_forward_history(client,"history/edge_candidate_snapshots.csv","history/market_snapshots.csv","reports")
    report["edge_forward_validation"]=edge_forward
    report["timing_forward_validation"]=grade_timing_forward("history/timing_decisions_v1.csv","history/market_snapshots.csv","history/edge_candidate_snapshots.csv","reports")
    report["timing_baseline_validation"]=benchmark_timing_forward("reports/edge_timing_forward_graded.csv","reports")
    Path("outputs").mkdir(exist_ok=True); Path("docs").mkdir(exist_ok=True)
    Path("outputs/live_performance.json").write_text(json.dumps(report,indent=2)); Path("docs/live_performance.json").write_text(json.dumps(report,indent=2))
    for name in ("tier_performance.json","tier_performance.csv","tier_validation_clean.csv","live_graded_tiers.csv","live_graded_tiers_clean.csv","edge_forward_performance.json","edge_forward_graded.csv","edge_timing_forward_performance.json","edge_timing_forward_graded.csv","edge_timing_baseline_performance.json","edge_timing_baseline_graded.csv"):
        src=Path("reports")/name
        if src.exists():
            (Path("outputs")/name).write_text(src.read_text()); (Path("docs")/name).write_text(src.read_text())
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
