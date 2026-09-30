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

## v7.1 architecture

The system intentionally separates the jobs that should not be conflated:

- **Fair-score engine** — leakage-safe opponent-adjusted ratings plus advanced football features; sportsbook prices do not enter the score projection.
- **Dynamic football layer** — prior-game EPA, passing/rushing EPA, success rate, explosiveness, scoring-opportunity/finishing metrics, pace, plays, field position and related efficiency features. Current live lookups carry the latest completed state forward through bye weeks without using future-week information.
- **Preseason / roster priors** — available roster talent, returning production and coaching-continuity information.
- **Probability layer** — nested chronological tuning/calibration, Brier score, log loss, ECE, residual variance, fail-closed shrinkage, and expanding-season walk-forward diagnostics.
- **Market intelligence layer** — verified ESPN/ESPN Core markets, optional multi-book consensus via `THE_ODDS_API_KEY`, no-vig probabilities, market dispersion, and line shopping. When multiple executable quotes exist, the quant layer uses the best verified line/price rather than a consensus number as if it were bettable.
- **Risk/context layer** — point-in-time current injuries/QB availability, current roster availability, indoor-aware kickoff weather, travel, rest, altitude, source freshness, context-quality gating, model-vs-market disagreement, volatility and fractional-Kelly sizing.
- **Portfolio layer** — game, team, market, slate and correlated kickoff-window exposure caps.
- **Evidence layer** — week-by-week historical retraining against archived market data, ML/ATS/total grading, ROI, units, max drawdown, CLV proxy, confidence intervals, calibration, and time-split threshold validation.
- **Release layer** — explicit RESEARCH/PAPER/SHADOW/PRODUCTION states. Production is impossible unless engineering checks, dynamic feature coverage, multi-book breadth, historical evidence and independent live/shadow evidence all pass.

The public-facing Cooper-style carousel is a separate **replica presentation layer**. Jason Cooper's private scoring formula is not public; this repository does not claim to contain it.

## Fail-closed behavior

The system is designed to refuse confidence rather than manufacture it:

- If a learned score adjustment fails its untouched release holdout, its weight falls back to zero.
- If a betting market cannot validate a threshold on an independent time split with non-negative holdout ROI and CLV, that market is disabled for live quant signals.
- A historically weak week is excluded only when it is independently negative in both prior data and holdout data with minimum sample sizes.
- Missing verified sportsbook data displays **NO LINE**; it is not imputed or invented.
- Missing or stale injury/roster/weather context lowers usable context coverage and increases uncertainty risk; a source name by itself does not earn context credit.
- Current-only injury, roster and weather inputs are disabled for historical-season runs instead of being backfilled into old games.
- A green GitHub workflow means the software ran. It does **not** mean a market edge is proven.
- Approved real stake remains zero unless the hard PRODUCTION gate passes.

## Key files

- `harbin/ratings.py` — opponent-adjusted pregame state and independent score baseline.
- `harbin/advanced.py` — leak-free dynamic advanced metrics plus preseason/static features.
- `harbin/models.py` — residual models, nested validation, season walk-forward diagnostics, shrinkage and win-probability calibration.
- `harbin/data.py` — schedules/results and verified live/historical market data.
- `harbin/market_intel.py` — multi-book consensus, executable best-line/best-price diagnostics and dispersion.
- `harbin/context.py` — current point-in-time injury/QB/roster, indoor-aware weather, rest, travel, altitude, cache freshness and context quality.
- `harbin/pro_market.py` — validated EV gates and odds-aware fractional Kelly.
- `harbin/portfolio.py` — concentration and correlated-kickoff exposure controls.
- `harbin/policy.py` — time-split production policy calibration and fail-closed market/week gates.
- `harbin/line_history.py` — first-seen/current line movement.
- `harbin/grading.py` — ongoing grading of archived live/shadow decisions.
- `harbin/backtest.py` — historical walk-forward betting proof.
- `harbin/release_gate.py` — hard deployment criteria.
- `harbin/health.py` — engineering/model-readiness score; never a profitability score.
- `harbin/render.py` — Cooper-style 14-games-per-page cards.
- `harbin/pipeline.py` — end-to-end v7.1 production run.
- `docs/STAGE4_WEATHER_QB_INJURY_ROSTER.md` — Stage 4 timing, freshness, roster, injury, weather and leakage contract.

## Data behavior

The free stack is designed around SportsDataverse/cfbfastR datasets, ESPN public endpoints, and Open-Meteo. An optional external multi-book odds source is supported through the `THE_ODDS_API_KEY` GitHub secret; if it is not configured, the system reports the missing breadth honestly instead of pretending a single provider is a consensus market.

Current-only injuries, current roster availability, weather and market quotes do not leak backward into historical score training. Historical backtests use only information available at the simulated decision time. Current context is a post-prediction risk/confidence layer until enough historical point-in-time context exists to validate directional score adjustments.

## What “10/10” means here

Engineering components can reach full readiness when their hard checks pass. **Profitability cannot be assigned a perfect score by code changes.** It must be earned by a sufficiently large, leakage-safe out-of-sample and forward sample with positive CLV, positive ROI across multiple markets/seasons, acceptable drawdown, calibrated probabilities, and a confidence interval that clears zero. The release gate is intentionally designed to keep the model in research/paper/shadow mode until that evidence exists.
