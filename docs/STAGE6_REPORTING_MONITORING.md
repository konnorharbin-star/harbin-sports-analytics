# Stage 6 — Reporting, Monitoring, and Publication Reconciliation

Stage 6 turns the live model outputs into one auditable reporting surface and adds a final integration barrier between a successful model run and a published dashboard.

## Canonical audit snapshot

`harbin.reporting` now builds `outputs/audit_snapshot.json` and copies the exact same document to `docs/audit_snapshot.json`.

The snapshot reconciles:

- platform/model identity and season/week;
- prediction row count;
- chronological model-validation metrics;
- data-quality contracts;
- operational monitoring and distribution drift;
- hard release-gate state;
- historical backtest evidence;
- independent live/shadow evidence;
- Stage 5 bankroll/portfolio controls;
- approved stake versus release/policy mode.

The public audit dashboard reads this single snapshot instead of independently joining many JSON files in the browser. That removes a class of stale-file and mixed-run reporting errors.

## Publication reconciliation

`validate_publication_files()` compares the generated output bundle with the public `docs/` bundle. It verifies that:

- the private and public audit snapshots are byte-equivalent at the JSON-object level;
- season, week, and platform version agree with public metadata;
- release state agrees with the public release gate;
- approved units agree with the portfolio summary;
- live-readiness score agrees with monitoring output;
- public data-quality output exists and parses.

`validate_publication.py` exposes this check as a command-line integration gate. The `CFB Model + Dashboard` workflow runs it after `run_week.py` and before committing generated state. A reconciliation failure therefore stops publication.

## Fail-closed stake reconciliation

The reporting layer independently verifies that non-zero approved stake can only appear when all of the following are simultaneously true:

1. the hard release gate says `PRODUCTION`;
2. `production_eligible` is true;
3. the calibrated policy deployment mode is `production`;
4. the Stage 5 portfolio mode is `production`.

It also verifies that approved stake cannot exceed the portfolio allocation. These checks do not replace Stage 5 controls; they are an independent reporting-layer reconciliation.

## Distribution drift

`harbin.monitoring` now reports explicit drift diagnostics for current margin, total, and calibrated-win-probability distributions against the historical walk-forward prediction reference when enough observations exist.

For each monitored distribution the output contains:

- stability score;
- standardized mean shift;
- live/reference standard deviations;
- volatility ratio;
- live and reference sample sizes.

A stability score below 55 raises an operational alert. Missing historical reference data remains visible; it is not fabricated.

Distribution drift is an observability signal, not evidence of betting profitability and not by itself a reason to retrain a model.

## Forward run history

Each completed reporting bundle appends a compact record to `history/audit_snapshots_v1.jsonl`. Repeated publication of the exact same model timestamp/season/week does not append a duplicate.

The record tracks release state, publication status, live-readiness score, distribution stability, approved units, and data-quality status. The audit snapshot carries recent records and can surface a material readiness drop versus the recent median.

## Human-readable run report

Every production run also writes:

- `outputs/RUN_REPORT.md`
- `docs/run_report.md`

The report summarizes model validation, monitoring, portfolio state, historical evidence, live evidence, blockers, and alerts while explicitly separating software/model readiness from profitability evidence.

## Dashboard

`docs/audit.html` is now a single-snapshot audit view covering:

- hard release gates;
- publication reconciliation;
- distribution drift;
- model validation;
- historical and live evidence;
- portfolio execution state;
- current alerts and blockers.

## Integration tests

`tests/test_stage6_reporting.py` verifies:

- a normal PAPER state reconciles successfully;
- impossible non-zero production stake fails reporting reconciliation;
- the canonical output/public snapshot bundle is identical;
- cross-file season/week drift is detected;
- material prediction-distribution drift produces diagnostics and an alert.

The repository-wide test workflow remains the merge gate.

## Stage boundary

Stage 6 closes dashboard/reporting and full publication integration. Stage 7 is the final repository-wide audit: re-check every stage, generated workflow, release assumption, and remaining production blocker before any production-ready claim is considered.
