#!/usr/bin/env python3
import argparse
from harbin.backtest import run_backtest

p=argparse.ArgumentParser(description="Leakage-safe historical CFB market backtest")
p.add_argument("--start-season",type=int,default=2023)
p.add_argument("--end-season",type=int,default=2025)
p.add_argument("--history-start",type=int,default=2018)
a=p.parse_args()
bets,summary=run_backtest(a.start_season,a.end_season,a.history_start)
print(f"Backtest complete: {a.start_season}-{a.end_season}")
print("Bets:",summary["overall"]["bets"])
print("ROI:",summary["overall"]["roi"])
print("Units:",summary["overall"]["units"])
print("Average CLV:",summary["overall"]["avg_clv"])
print("ROI 95% bootstrap CI:",summary["overall"]["roi_ci_95"])
print("Evidence:",summary["evidence"]["status"])
print("Open reports/backtest_summary.json for the full report.")
