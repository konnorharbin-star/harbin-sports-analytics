#!/usr/bin/env python3
import argparse
from harbin.pipeline import run_week

p=argparse.ArgumentParser(description="Run Harbin Sports Analytics CFB model")
p.add_argument("--season",type=int)
p.add_argument("--week",type=int)
p.add_argument("--history-start",type=int)
a=p.parse_args()
pred,meta=run_week(a.season,a.week,a.history_start)
print(f"Harbin CFB v{meta['model_version']} complete: {meta['season']} Week {meta['week']} — {len(pred)} games")
print("Validation:",meta["validation"])
print("Margin MAE:",meta["metrics"].get("margin_mae"),"| Total MAE:",meta["metrics"].get("total_mae"))
print("Calibration Brier:",meta["metrics"].get("win_brier"),"| ECE:",meta["metrics"].get("win_ece"))
print("Advanced coverage:",f"{meta['advanced_features']['live_coverage']:.1%}")
print("Market:",meta["market_status"])
print("Multi-book coverage:",f"{meta['market_intelligence'].get('multi_book_coverage',0):.1%}")
print("Quant picks:",meta["quant_picks"])
print("System health:",meta["health"]["system_health_score"],"/100 (readiness, not a profit guarantee)")
print("Open outputs/README.md for the latest result links.")
