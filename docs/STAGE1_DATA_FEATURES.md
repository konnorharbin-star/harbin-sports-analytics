# Stage 1 — Data and Feature Stack

Stage 1 makes the model inputs reproducible, current, and leakage-safe before betting-validation results are trusted.

## Pregame sources

- SportsDataverse / ESPN CFB schedules and final scores
- SportsDataverse advanced team efficiency data
- roster-talent priors
- returning-production priors
- coaching-continuity priors

Sportsbook data, closing lines, CLV, graded results, stake outputs, and postgame fields are explicitly blocked from predictive training features.

## As-of rule

For a Week `W` prediction, dynamic efficiency features contain only observations from weeks `< W`.

Each advanced metric has two pregame versions:

1. season-to-date mean
2. recent-form EWMA with alpha `0.35`

No Week `W` observation updates state until after the Week `W` snapshot is created. The store also materializes the next-week snapshot and walks backward across byes or source lag, so a missing upstream row does not zero out the live feature stack.

At a season boundary, prior-season values may seed the next season only through a 0.62 carry factor. Talent, returning production, and coaching continuity are preseason priors.

## Cache freshness

Historical advanced seasons may be cached indefinitely. The current season refreshes at least every 6 hours. Roster/prior files refresh at least every 24 hours.

If refresh fails but a valid cached copy exists, the stale copy can be used, but the fallback and error are recorded in metadata.

## Identity and coverage

Team joins prefer numeric ESPN team IDs and fall back to canonicalized team names. Every live matchup receives:

- `advanced_home_coverage`
- `advanced_away_coverage`
- `advanced_pair_coverage`
- `advanced_dynamic_pair_coverage`

These fields let downstream monitoring distinguish fully observed games from sparse feature rows.

## Training feature contract

`harbin.models.feature_columns` only admits numeric, non-constant, sufficiently populated pregame columns. It rejects fields associated with:

- sportsbook / market prices
- closing-line information
- moneyline / odds fields
- CLV
- final / postgame / result fields
- quant recommendations and stake outputs
- training targets

Live prediction fails closed if any trained feature is absent from the live frame.

## Feature audit

`harbin.feature_audit` provides a reusable train-vs-live audit for:

- feature parity
- leakage-name detection
- duplicate columns
- sparse features
- all-missing live features
- large live-vs-train median shifts measured in training IQRs
- advanced pair coverage

Structural failures are errors. Distribution shift is a warning because real football environments can legitimately move away from historical medians.

## CI protection

`tests/test_stage1_features.py` verifies cache TTLs, strict pregame as-of semantics, EWMA recent form, bye-week fallback, matchup coverage, leakage blocking, and train/live parity auditing.

Stage 1 is complete only after the full repository test suite and a full model workflow pass on `main`.
