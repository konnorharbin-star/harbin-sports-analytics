# Stage 4 — Weather, QB, Injury, and Roster Context

Stage 4 hardens the model's **current-game context layer** without leaking present-day information into historical score training. Fair-score projections remain market- and current-context-independent; current injuries, roster availability, and forecasts are attached after prediction and are used to reduce confidence and stake when information is adverse or incomplete.

## Point-in-time boundary

`harbin.context.ContextStore` now has an explicit `as_of` timestamp. Current injury, roster, and forecast context is enabled only for the active CFB season. Historical season runs fail closed rather than applying today's injury report or a present-day weather API to old games.

Injury rows are also filtered to reports no later than `as_of` and no later than the target week before any availability calculation is made.

## Injury and quarterback state

SportsDataverse's ESPN injury feed can contain multiple reports for the same athlete during a season. Stage 4 no longer sums every historical report.

For each athlete, the context layer:

1. discards future-dated and future-week rows;
2. keeps the latest admissible report;
3. treats a later ACTIVE/AVAILABLE/CLEARED report as superseding an older OUT report;
4. derives a bounded severity from status text;
5. decays old non-season-ending reports and expires very old stale reports;
6. preserves explicit season-ending conditions;
7. records QB-specific risk separately from general injury load.

The live output includes home/away injury counts, QB injury risk, and injury-report age diagnostics.

## Current roster availability

Stage 4 adds the current SportsDataverse ESPN season roster feed. The layer records roster size, active/inactive counts when the upstream schema exposes status, QB depth when a position field is available, and a bounded roster-availability risk.

Missing status or position fields are **not invented**. Unknown roster status does not automatically mark a player inactive. Roster coverage and data quality report the limitation instead.

Preseason talent, returning production, and coaching continuity remain in `harbin.advanced`; Stage 4 does not duplicate those priors or inject current roster state into historical model fitting.

## Weather and venue handling

Venue metadata now prefers the SportsDataverse ESPN teams feed and falls back to the legacy team-info feed when necessary. Indoor venues explicitly suppress outdoor weather risk.

For outdoor games within the live forecast horizon, Open-Meteo provides kickoff-hour temperature, precipitation probability/amount, weather code, wind, and gusts. Present-day forecasts are never requested for historical kickoffs.

Weather risk is bounded and transparent. It affects the downstream risk multiplier; it does not arbitrarily rewrite the fair score.

## Rest, travel, and altitude

Existing rest, travel-distance, venue-elevation, and altitude-change diagnostics remain. Stage 4 preserves them while separating their risk contribution from availability and weather.

The output now includes team-specific availability risk and `availability_edge_home` for audit. A positive value means the away team carries more measured availability risk than the home team; it is diagnostic, not an unvalidated point-spread adjustment.

## Freshness and fail-closed quality

Injury, roster, and team metadata caches now have explicit TTLs. A failed refresh may use a stale cached file, but that state receives a freshness penalty. Missing core inputs lower context quality instead of receiving credit merely because one context source exists.

The context layer computes injury-source, roster, venue, weather and travel coverage plus a bounded `quality_score`. For backward compatibility, `coverage` now represents usable context quality, so the existing release and monitoring consumers fail closed when core context is missing or stale. Per-game `context_quality_score` is written into the full prediction output.

Missing or stale context is also treated as uncertainty in `context_risk`; it reduces downstream risk sizing instead of being mistaken for a clean zero-risk environment.

## Why current context does not shift the fair score yet

A directional score adjustment for an injured quarterback or severe weather can be tempting, but a fixed hand-written point value would create an unvalidated model inside the model. Stage 4 therefore uses current context for risk/confidence and exposes team-side diagnostics while leaving fair-score estimates unchanged.

A future score adjustment should be trained and validated only after enough historical **point-in-time** injury/roster/weather snapshots exist to estimate the effect without leakage.

## CI protection

`tests/test_stage4_context.py` verifies that:

- a later cleared status supersedes an older OUT report;
- future injury reports/weeks cannot enter the current view;
- stale non-season-ending reports expire while explicit season-ending reports persist;
- roster inactive share and QB depth are summarized without fabricating missing status;
- indoor venues suppress weather risk;
- missing/stale core context lowers quality;
- health scoring uses measured context quality rather than source names.

The full repository test suite must pass before Stage 4 is merged to `main`.

## Stage boundary

Stage 4 does not redesign bankroll allocation, slate/team concentration caps, or portfolio correlation controls. Those remain Stage 5.
