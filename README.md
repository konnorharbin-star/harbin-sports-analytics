# Harbin Sports Analytics

Automated college-football projection, probability, market-comparison, risk, and evidence system that lives in this GitHub repository.

## Run the live model

1. Open **Actions**.
2. Select **CFB Model + Dashboard**.
3. Click **Run workflow**.
4. Enter season/week, or leave blank to auto-detect.
5. Wait for a green check.
6. Open **`outputs/README.md`**.

The latest GitHub Pages dashboard is generated from `docs/`.

## Run the proof layer

Open **Actions → CFB Walk-Forward Backtest → Run workflow**. This produces `reports/backtest_summary.json`, individual historical bets, edge/signal breakdowns, calibration tables, drawdown, ROI confidence intervals, and an opening-to-archive-final CLV proxy when the historical archive contains both prices.

## v4 architecture

The system intentionally separates six jobs that should not be conflated:

- **Fair-score engine** — leakage-safe opponent-adjusted ratings plus advanced football features; sportsbook prices do not enter the score projection.
- **Advanced football layer** — prior-week EPA, passing/rushing EPA, success rate, scoring-opportunity/finishing metrics, third-down efficiency, pace, plays, starting field position, plus available roster-talent/returning-production/continuity priors.
- **Probability layer** — nested chronological calibration with Brier score, log loss, ECE, residual variance, shrinkage guards, and season walk-forward validation.
- **Market intelligence layer** — live ESPN/ESPN Core prices, multi-book consensus when available, best moneyline, dispersion, no-vig probabilities, and first-seen line movement.
- **Risk/context layer** — current injuries/QB availability, weather, travel, rest, altitude, data-quality gating, model-vs-market disagreement, volatility, and capped fractional-Kelly sizing. Current-only context does not leak backward into historical training.
- **Evidence layer** — week-by-week retraining against historical opening/archive-final market data, ATS/ML/total grading, ROI, units, max drawdown, CLV proxy, bootstrap ROI confidence intervals, and calibration.

The public-facing Cooper-style carousel is a separate **replica presentation layer**. Jason Cooper's private scoring formula is not public; this repository does not claim to contain it.

## Key files

- `harbin/ratings.py` — opponent-adjusted pregame state and independent score baseline.
- `harbin/advanced.py` — leakage-safe prior-week advanced and preseason/static features.
- `harbin/models.py` — residual models, nested validation, walk-forward diagnostics, shrinkage, win-probability calibration.
- `harbin/data.py` — schedules/results and verified live/historical market data.
- `harbin/market_intel.py` — multi-book consensus and best-price diagnostics.
- `harbin/context.py` — current injury/QB, weather, rest, travel and altitude context.
- `harbin/pro_market.py` — EV gates, fractional Kelly, stake/risk controls.
- `harbin/line_history.py` — first-seen/current line movement.
- `harbin/grading.py` — ongoing grading of archived v4 decisions.
- `harbin/backtest.py` — historical walk-forward betting proof.
- `harbin/health.py` — system-readiness score; never a profitability score.
- `harbin/render.py` — Cooper-style 14-games-per-page cards.
- `harbin/pipeline.py` — end-to-end production run.

## Data behavior

The free stack is designed around SportsDataverse/cfbfastR datasets, ESPN public endpoints, and Open-Meteo. Missing or blocked sources degrade explicitly. The system does not silently fill sportsbook prices, injuries, weather, or advanced metrics with invented values.

A green workflow means the software ran successfully. It does **not** mean the model is profitable. `outputs/system_health.json` measures engineering/model readiness. Market-beating claims require a sufficiently large, leakage-safe historical and forward sample with ROI, calibration, drawdown, and CLV evidence.
