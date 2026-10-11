# Harbin Sports Analytics

Automated college-football projection, probability, market-comparison, risk, monitoring, and evidence system that lives in this GitHub repository.

## Cross-sport command center

The [read-only model command center](docs/command-center.html) brings the CFB and [NFL](https://github.com/konnorharbin-star/harbin-nfl-analytics) published audit, evidence, monitoring, and GitHub Actions status into one place. It does not approve bets or override release gates. [Operations and deployment guide](docs/COMMAND_CENTER.md).

## Free same-book near-kickoff market research

The [free read-only line movement audit](docs/FREE_SAMEBOOK_MOVEMENT.md) compares each archived pregame research observation against later market snapshots **from the same sportsbook** only. It explicitly distinguishes point-line movement from raw American-odds movement, reports source and timing failures, and never treats a public near-kickoff quote as an official bookmaker closing line, a verified executable wager, or proof of profitability.

## Free forward research grading

The [free forward grading workflow](docs/FREE_FORWARD_GRADING.md) reviews historical first-seen research watchlist entries against each model's already-public postgame grade ledger. Outputs are explicitly **hypothetical**, do not represent placed wagers, and cannot verify an actual execution price. This is a read-only, free-to-operate research feature.

## Run every FBS game with one click (or automatically)

The model already runs on GitHub Actions **three times daily** plus an
additional Saturday run. Each run processes **the entire upcoming FBS
week**—every game involving an FBS team, including FBS vs FCS—rather
than requiring game-by-game prompts. All FBS/FCS score projections are
labelled research-only and cannot produce an approved betting signal.

1. Open [Actions → CFB Model + Dashboard](https://github.com/konnorharbin-star/harbin-sports-analytics/actions/workflows/cfb-model.yml).
2. Click **Run workflow**, select `main`, and leave season/week blank
   for the next upcoming week (or specify both to choose a full week).
3. Open the run summary and `outputs/schedule_coverage_audit.json`.
   The job fails loudly if even one future FBS game is missing.
4. See `outputs/README.md` or the latest generated dashboard for
   the full slate and the betting release gates.

See the [full FBS slate runbook](docs/FULL_FBS_SLATE_RUNBOOK.md) for
automation schedule, started-game exclusions, immutable original
predictions, FBS/FCS scope and public-data caveats.

The GitHub Pages dashboard is generated from `docs/`.

## Run the proof layer

Open **Actions → CFB Walk-Forward Backtest → Run workflow**. This produces `reports/backtest_summary.json`, individual historical bets, edge/signal breakdowns, calibration tables, drawdown, ROI confidence intervals, and an opening-to-archive-final CLV proxy when the historical archive contains both prices.

## v7.3 platform architecture

The system intentionally separates the jobs that should not be conflated:

- **Fair-score engine** — leakage-safe opponent-adjusted ratings plus advanced football features; sportsbook prices do not enter the score projection.
- **Dynamic football layer** — prior-game EPA, passing/rushing EPA, success rate, explosiveness, scoring-opportunity/finishing metrics, pace, plays, field position and related efficiency features. Current live lookups carry the latest completed state forward through bye weeks without using future-week information.
- **Preseason / roster priors** — available roster talent, returning production and coaching-continuity information.
- **Probability layer** — nested chronological tuning/calibration, Brier score, log loss, ECE, residual variance, fail-closed shrinkage, and expanding-season walk-forward diagnostics.
- **Market intelligence layer** — verified ESPN/ESPN Core markets, optional multi-book consensus via `THE_ODDS_API_KEY`, no-vig probabilities, market dispersion, and line shopping. When multiple executable quotes exist, the quant layer uses the best verified line/price rather than a consensus number as if it were bettable.
- **Risk/context layer** — point-in-time current injuries/QB availability, current roster availability, indoor-aware kickoff weather, travel, rest, altitude, source freshness, context-quality gating, model-vs-market disagreement, volatility and fractional-Kelly sizing.
- **Portfolio/execution layer** — unit-based drawdown throttles, executable-price provenance, slate/game/team/market/book/kickoff caps, bet-count limits, and zero approved stake outside a fully open production path.
- **Evidence layer** — week-by-week historical retraining against archived market data, ML/ATS/total grading, ROI, units, max drawdown, CLV proxy, confidence intervals, calibration, and time-split threshold validation.
- **Release layer** — explicit RESEARCH/PAPER/SHADOW/PRODUCTION states. Production is impossible unless engineering checks, dynamic feature coverage, multi-book breadth, historical evidence and independent live/shadow evidence all pass.
- **Reporting/observability layer** — one canonical audit snapshot, distribution-drift diagnostics, compact forward run history, human-readable run reports, and cross-file publication reconciliation before dashboard state is committed.

The public-facing Cooper-style carousel is a separate **replica presentation layer**. Jason Cooper's private scoring formula is not public; this repository does not claim to contain it.

## Fail-closed behavior

The system is designed to refuse confidence rather than manufacture it:

- If a learned score adjustment fails its untouched release holdout, its weight falls back to zero.
- If a betting market cannot validate a threshold on an independent time split with non-negative holdout ROI and CLV, that market is disabled for live quant signals.
- A historically weak week is excluded only when it is independently negative in both prior data and holdout data with minimum sample sizes.
- Missing verified sportsbook data displays **NO LINE**; it is not imputed or invented.
- Missing or stale injury/roster/weather context lowers usable context coverage and increases uncertainty risk; a source name by itself does not earn context credit.
- Current-only injury, roster and weather inputs are disabled for historical-season runs instead of being backfilled into old games.
- Missing executable sportsbook provenance blocks portfolio approval rather than assuming a price is bettable.
- A current unit drawdown at the configured hard stop forces approved portfolio stake to zero.
- A stale production gate cannot approve stake if the independent live/shadow grading ledger is missing.
- Reporting independently rejects non-zero approved stake unless release, policy, and portfolio modes all reconcile to production.
- Dashboard publication stops if the public metadata/release/portfolio/monitoring bundle disagrees with the canonical audit snapshot.
- A green GitHub workflow means the software ran. It does **not** mean a market edge is proven.
- Approved real stake remains zero unless every hard PRODUCTION gate and Stage 5 execution control is satisfied.

## Key files

- `harbin/ratings.py` — opponent-adjusted pregame state and independent score baseline.
- `harbin/advanced.py` — leak-free dynamic advanced metrics plus preseason/static features.
- `harbin/models.py` — residual models, nested validation, season walk-forward diagnostics, shrinkage and win-probability calibration.
- `harbin/data.py` — schedules/results and verified live/historical market data.
- `harbin/market_intel.py` — multi-book consensus, executable best-line/best-price diagnostics and dispersion.
- `harbin/context.py` — current point-in-time injury/QB/roster, indoor-aware weather, rest, travel, altitude, cache freshness and context quality.
- `harbin/pro_market.py` — validated EV gates and odds-aware fractional Kelly.
- `harbin/portfolio.py` — bankroll throttles, executable-quote checks, concentration caps and production stake approval.
- `harbin/policy.py` — time-split production policy calibration plus fail-closed market/week and portfolio defaults.
- `harbin/line_history.py` — first-seen/current line movement.
- `harbin/grading.py` — ongoing grading of archived live/shadow decisions.
- `harbin/backtest.py` — historical walk-forward betting proof.
- `harbin/monitoring.py` — operational readiness and explicit historical-reference distribution drift.
- `harbin/reporting.py` — canonical audit snapshot, run report, trend ledger and publication reconciliation.
- `harbin/release_gate.py` — hard deployment criteria.
- `harbin/health.py` — engineering/model-readiness score; never a profitability score.
- `harbin/render.py` — Cooper-style 14-games-per-page cards.
- `harbin/pipeline.py` — end-to-end v7.1 core model run.
- `validate_publication.py` — workflow-level verification that public dashboard files describe one coherent run.
- `docs/STAGE4_WEATHER_QB_INJURY_ROSTER.md` — Stage 4 timing, freshness, roster, injury, weather and leakage contract.
- `docs/STAGE5_PORTFOLIO_BANKROLL_EXECUTION.md` — Stage 5 unit-bankroll, execution provenance and concentration-control contract.
- `docs/STAGE6_REPORTING_MONITORING.md` — Stage 6 dashboard, drift, audit-snapshot and publication-integration contract.

## Data behavior

The free stack is designed around SportsDataverse/cfbfastR datasets, ESPN public endpoints, and Open-Meteo. An optional external multi-book odds source is supported through the `THE_ODDS_API_KEY` GitHub secret; if it is not configured, the system reports the missing breadth honestly instead of pretending a single provider is a consensus market.

Current-only injuries, current roster availability, weather and market quotes do not leak backward into historical score training. Historical backtests use only information available at the simulated decision time. Current context is a post-prediction risk/confidence layer until enough historical point-in-time context exists to validate directional score adjustments.

Stage 5 bankroll protection is unit-based. The repository does not infer a user's dollar bankroll, does not place wagers, and cannot promote the model into production; it can only reduce or halt risk after the release gate and production policy have been satisfied.

Stage 6 monitoring compares live prediction distributions with the stored walk-forward reference when enough observations are available. Drift is an operational diagnostic, not evidence of profitability and not an automatic retraining instruction.

## What “10/10” means here

Engineering components can reach full readiness when their hard checks pass. **Profitability cannot be assigned a perfect score by code changes.** It must be earned by a sufficiently large, leakage-safe out-of-sample and forward sample with positive CLV, positive ROI across multiple markets/seasons, acceptable drawdown, calibrated probabilities, and a confidence interval that clears zero. The release gate is intentionally designed to keep the model in research/paper/shadow mode until that evidence exists.

### Prospective paper recommendation records

[Operating contract](docs/RECOMMENDATION_OPERATIONS.md) · [NFL + CFB dashboard](docs/operations.html).

Communicated BETs require a permanent pre-kickoff GitHub receipt; legacy research signals are separate. No real wagers are placed.
