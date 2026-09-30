# Stage 1 — Data and Feature Stack

Stage 1 makes the model inputs reproducible, current, and leakage-safe before any betting-validation work is trusted.

## Pregame data sources

- SportsDataverse / ESPN CFB schedules and final scores
- SportsDataverse advanced team efficiency data
- roster-talent priors
- returning-production priors
- coaching-continuity priors

Sportsbook data, closing lines, CLV, graded results, stake outputs, and postgame fields are explicitly blocked from the predictive training feature set.

## As-of rule

For a Week `W` prediction, dynamic efficiency features contain only observations from weeks `< W`.

Each advanced metric now has two versions:

1. season-to-date pregame mean
2. pregame recent-form EWMA with alpha `0.35`

No Week `W` observation is allowed to update the state until after the Week `W` pregame feature row has been materialized.

At a season boundary, prior-season values can seed the next season only through a 0.62 carry factor. Talent, returning production, and coaching continuity are treated as preseason priors.

## Cache freshness

Historical advanced seasons may be cached indefinitely. The current season is refreshed at least every 6 hours. Roster/prior files are refreshed at least every 24 hours.

If a refresh endpoint fails but a valid cached copy exists, the run can use the stale copy, but that fallback is recorded in metadata. The system no longer silently freezes a current-season file forever.

## Identity and coverage

Team joins prefer numeric ESPN team IDs and fall back to canonicalized team names. Each live matchup now receives explicit feature-coverage fields:

- `advanced_home_coverage`
- `advanced_away_coverage`
- `advanced_pair_coverage`
- `advanced_dynamic_pair_coverage`

These values let downstream risk controls distinguish a fully observed game from a sparse feature row.

## Training feature contract

`harbin.models.feature_columns` only admits numeric, non-constant, sufficiently populated pregame columns. It rejects names associated with:

- sportsbook / market fields
- closing-line information
- moneyline / odds fields
- CLV
- final / postgame / result fields
- quant recommendations and stake outputs
- training targets

Live prediction also fails closed if any trained feature is absent from the live frame.

## Feature audit

`harbin.feature_audit` provides a reusable train-vs-live audit covering:

- train/live feature parity
- leakage-name detection
- duplicate columns
- sparse features
- live-empty features
- large live-vs-train median shifts measured in training IQRs
- advanced pair coverage

Structural failures are errors. Distribution shift is surfaced as a warning because a real football environment can legitimately move away from historical medians.

## CI coverage

`tests/test_stage1_features.py` protects:

- cache TTL behavior
- strict pregame as-of semantics
- EWMA recent-form construction
- matchup feature-coverage fields
- leakage blocking
- train/live parity audit behavior

Stage 1 is considered complete only when the repository test suite and a full model workflow both pass on `main`.
