#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from harbin.backtest_runtime import run_backtest
from harbin.policy import derive_production_policy, DEFAULT_POLICY
from harbin.proof import build_evidence_report

p=argparse.ArgumentParser(description="Leakage-safe historical CFB market backtest")
p.add_argument("--start-season",type=int,default=2023)
p.add_argument("--end-season",type=int,default=2025)
p.add_argument("--history-start",type=int,default=2018)
a=p.parse_args()

bets,summary=run_backtest(a.start_season,a.end_season,a.history_start)

# Never promote a model because evidence generation itself failed. If a source
# produces no historical candidate bets, keep the system in PAPER mode and
# write explicit diagnostics instead of crashing the workflow.
if bets.empty:
    policy=json.loads(json.dumps(DEFAULT_POLICY))
    policy["deployment_mode"]="paper"
    policy["source"]="conservative defaults; no matched historical betting sample"
    Path("reports/production_policy.json").write_text(json.dumps(policy,indent=2))
    evidence={
        "status":"UNPROVEN",
        "overall":summary.get("overall",{}),
        "by_market":{},"by_signal":{},"by_season":{},"by_week":{},
        "criteria":{
            "robust":"1000+ bets, positive ROI 95% CI lower bound, positive CLV, breadth across markets and seasons",
            "validated":"500+ bets with positive ROI and CLV",
            "developing":"150+ bets",
            "unproven":"below developing sample"
        },
        "note":"No matched historical wager sample was produced; production wagering remains disabled."
    }
    Path("reports/evidence_report.json").write_text(json.dumps(evidence,indent=2))
else:
    policy=derive_production_policy()
    evidence=build_evidence_report()

print(f"Backtest complete: {a.start_season}-{a.end_season}")
print("Bets:",summary["overall"]["bets"])
print("ROI:",summary["overall"]["roi"])
print("Units:",summary["overall"]["units"])
print("Average CLV:",summary["overall"]["avg_clv"])
print("ROI 95% bootstrap CI:",summary["overall"]["roi_ci_95"])
print("Evidence:",evidence["status"])
print("Deployment mode:",policy["deployment_mode"])
print("Open reports/backtest_summary.json, reports/evidence_report.json, and reports/production_policy.json.")
