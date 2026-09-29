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

Open **Actions → CFB Walk-Forward Backtest → Run workflow**. This produces the historical wager ledger, walk-forward prediction diagnostics, calibration, drawdown, CLV proxy, production-policy research, holdout validation, and the v7 release audit.

Important reports:

- `reports/backtest_summary.json` — raw historical signal performance and prediction diagnostics.
- `reports/production_policy.json` — evidence-gated market thresholds and blocked historical regimes.
- `reports/policy_validation.json` — final holdout audit of the enabled policy only.
- `reports/evidence_report.json` — broad historical evidence summary.
- `reports/v7_audit.json` — unresolved research/production blockers and real-money release gate.

## v7 architecture

The system deliberately separates prediction quality, market comparison, bankroll/risk controls, and betting proof. A strong score model is not automatically a profitable betting model.

- **Fair-score engine** — leakage-safe opponent-adjusted ratings plus advanced football features; sportsbook prices do not enter the core score projection.
- **Advanced football layer** — prior-week EPA, passing/rushing EPA, success rate, explosiveness, finishing/scoring opportunity metrics, pace, field position, plus available roster-talent/returning-production/continuity priors. Team identity is matched by both ESPN ID and canonical team name to prevent silent zero-coverage failures.
- **Probability layer** — chronological calibration with Brier score, log loss, ECE, residual variance, shrinkage guards, and season walk-forward diagnostics.
- **Market intelligence layer** — verified ESPN/ESPN Core live markets, best-price/consensus support when multiple books are available, no-vig probabilities, line movement, and market-disagreement diagnostics.
- **Risk/context layer** — current injuries/QB availability, weather, travel, rest, altitude, data-quality gating, model volatility, market disagreement, and capped fractional-Kelly sizing. Current-only context never leaks into historical training.
- **Evidence layer** — historical week-by-week retraining and grading for moneyline/spread/total markets, ROI, units, drawdown, CLV proxy, calibration, and week-cluster bootstrap uncertainty intervals.
- **Evidence-gated production policy** — a market is disabled by default. It only becomes eligible for PAPER signals when a strict time-split holdout has at least 100 qualifying bets, positive ROI with a positive 95% lower confidence bound, and non-negative CLV. Historically harmful week regimes can be blocked automatically.
- **Release gate** — real-money mode is intentionally disabled. Promotion requires positive forward evidence, stable calibration, meaningful CLV, independent market verification, and no unresolved data-quality alerts.

The public-facing Cooper-style carousel is a separate **replica presentation layer**. Jason Cooper's private scoring formula is not public; this repository does not claim to contain it.

## Key files

- `harbin/ratings.py` — opponent-adjusted pregame state and independent score baseline.
- `harbin/advanced.py` — leakage-safe advanced/static features and robust team identity matching.
- `harbin/models.py` — residual models, validation, walk-forward diagnostics, shrinkage, probability calibration.
- `harbin/data.py` — schedules/results and verified live/historical market data.
- `harbin/market_intel.py` — multi-book consensus and best-price diagnostics.
- `harbin/context.py` — current injury/QB, weather, rest, travel and altitude context.
- `harbin/pro_market.py` — EV gates, evidence-aware market selection, fractional Kelly and stake controls.
- `harbin/policy.py` — strict market validation, bootstrap evidence gates and regime filters.
- `harbin/evidence_v7.py` — holdout policy validation and release audit.
- `harbin/line_history.py` — first-seen/current line movement.
- `harbin/grading.py` — ongoing grading of archived decisions.
- `harbin/backtest.py` / `harbin/backtest_runtime.py` — historical walk-forward betting proof.
- `harbin/health.py` — evidence-aware system-readiness score; never a profitability score.
- `harbin/render.py` — Cooper-style 14-games-per-page cards.
- `harbin/pipeline.py` — end-to-end production run.

## What “10/10” means here

The engineering target is an institutional-style research process: reproducible data, no temporal leakage, calibrated probabilities, market-aware validation, conservative risk controls, monitoring, and explicit release gates. It does **not** mean a guaranteed edge or guaranteed profit.

A component that depends on evidence or an external data source cannot honestly be scored at 10/10 merely by adding code. In particular:

- Multi-book intelligence requires at least one independent second sportsbook source. The existing optional free-tier integration can use `THE_ODDS_API_KEY` when configured.
- Live monitoring becomes statistically meaningful only after enough forward snapshots and graded decisions accumulate.
- Profitability is considered unproven until the enabled policy passes historical holdout **and** forward-paper evidence with uncertainty bounds and CLV.

## Data behavior

The free stack is designed around SportsDataverse/cfbfastR datasets, ESPN public endpoints, Open-Meteo, and optional supplemental market sources. Missing or blocked sources degrade explicitly. The system does not silently fabricate sportsbook prices, injuries, weather, or advanced metrics.

A green workflow means the software ran successfully. It does **not** mean the model is profitable. `outputs/system_health.json` measures engineering/model readiness. Market-beating claims require sufficiently large leakage-safe historical and forward samples with positive ROI uncertainty bounds, calibration, drawdown, and CLV evidence.
