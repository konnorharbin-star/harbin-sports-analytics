#!/usr/bin/env python3
import argparse
from harbin_model_v2 import run

p=argparse.ArgumentParser()
p.add_argument('--season',type=int)
p.add_argument('--week',type=int)
p.add_argument('--history-start',type=int)
a=p.parse_args()
pred,meta=run(a.season,a.week,a.history_start)
print(f"Harbin CFB model complete: {meta['season']} Week {meta['week']} — {len(pred)} upcoming games")
print(meta['metrics'])
if len(pred):
    cols=['away_team','home_team','away_score','home_score','win_pct','ml_badge','spread_badge','total_badge']
    print(pred[cols].head(15).to_string(index=False))
