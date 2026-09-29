#!/usr/bin/env python3
import argparse
from harbin.pipeline import run_week

p=argparse.ArgumentParser(description="Run Harbin Sports Analytics CFB v2")
p.add_argument('--season',type=int)
p.add_argument('--week',type=int)
p.add_argument('--history-start',type=int)
a=p.parse_args()
pred,meta=run_week(a.season,a.week,a.history_start)
print(f"Harbin CFB v{meta['model_version']} complete: {meta['season']} Week {meta['week']} — {len(pred)} games")
print(meta['metrics'])
if len(pred):
    cols=['away_team','home_team','away_score','home_score','win_pct','ml_badge','spread_badge','total_badge']
    print(pred[cols].head(20).to_string(index=False))
