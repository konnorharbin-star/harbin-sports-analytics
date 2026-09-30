#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from harbin.backtest_audit import run_backtest
from harbin.policy import derive_production_policy, DEFAULT_POLICY
from harbin.proof import build_evidence_report

p=argparse.ArgumentParser(description="Leakage-safe historical CFB market backtest")
p.add_argument("--start-season",type=int,default=2023)
p.add_argument("--end-season",type=int,default=2025)
p.add_argument("--history-start",type=int,default=2018)
a=p.parse_args()

bets,summary=run_backtest(a.start_season,a.end_season,a.history_start)

# Evidence is frozen before policy calibration. The policy may read that evidence,
# but the untouched policy-evaluation block can never rewrite historical outcomes.
if bets.empty:
    evidence={
        "status":"UNPROVEN",
        "overall":summary.get("overall",{}),
        "all_archive_overall":summary.get("overall",{}),
        "promotion_sample":{"entry_quote_verified":False,"verified_bets":0,"all_archive_bets":0,"excluded_unverified_bets":0},
        "by_market":{},"by_signal":{},"by_season":{},"by_week":{},
        "criteria":{
            "robust":"1000+ verified opening-entry bets, positive week-block ROI 95% CI lower bound, positive CLV and breadth across markets/seasons",
            "validated":"500+ verified opening-entry bets with positive ROI and CLV",
            "developing":"150+ verified opening-entry bets",
            "unproven":"below developing sample or unverified entry timing"
        },
        "note":"No matched historical wager sample was produced; production wagering remains disabled."
    }
    Path("reports/evidence_report.json").write_text(json.dumps(evidence,indent=2))
    policy=json.loads(json.dumps(DEFAULT_POLICY)); policy["deployment_mode"]="paper"; policy["source"]="conservative defaults; no verified historical betting sample"
    Path("reports/production_policy.json").write_text(json.dumps(policy,indent=2))
else:
    evidence=build_evidence_report()
    policy=derive_production_policy()

print(f"Backtest complete: {a.start_season}-{a.end_season}")
print("Archive bets:",summary["overall"]["bets"])
print("Promotion bets:",evidence.get("overall",{}).get("bets"))
print("Promotion ROI:",evidence.get("overall",{}).get("roi"))
print("Promotion CLV:",evidence.get("overall",{}).get("avg_clv"))
print("Promotion ROI 95% week-block CI:",evidence.get("overall",{}).get("roi_ci_95"))
print("Evidence:",evidence["status"])
print("Deployment mode:",policy["deployment_mode"])
print("Open reports/backtest_summary.json, reports/evidence_report.json, and reports/production_policy.json.")
