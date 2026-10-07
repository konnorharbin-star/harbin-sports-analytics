# CFB edge timing diagnostics — research only

`harbin/edge_timing.py` extends the **already holdout-supported, price-confirmed**
edge priority board. It does not generate new bets, override a subgroup veto,
adjust fair scores/probabilities/EV, or authorize staking.

## Comparable evidence

- Current quotes require explicit book and timestamp and must be no more than
  120 minutes old. Post-kickoff, missing, stale, and future-dated quotes fail closed.
- Prior observations must match **game ID, spread market, selected team, and book**.
  Comparisons never mix books or reverse home/away spread sign.
- Sources are (a) `history/edge_candidate_snapshots.csv` with source quote
  timestamps and (b) `history/market_snapshots.csv`, specifically the per-book
  spread quotes inside `market_quotes_json`. In (b), `captured_at` is when we
  **observed** an offer; it is **not** an exchange/sportsbook quote-publication
  timestamp. Missing per-book prices are excluded.
- Current and historical evidence must be pre-kickoff, non-future relative to
  evaluation time, and at least 10 minutes apart. The previous observation must
  be within 12 hours of the current quote. Historic snapshots fetched after the
  decision time cannot become prior observations.
- No movement is inferred from a single quote or different books.

## Research actions

- `BET_NOW_RESEARCH`: confirmed priority CORE/ROBUST_CORE candidate with
  valid quote, conservative positive price margin, remaining bet-to-line room,
  and a recent observed same-book worsening of at least 0.5 spread points or
  (when the spread is stable) 1 percentage point of implied break-even price.
- `WAIT_MONITOR`: the analogous recent improvement; **not a validated forecast**
  that prices will keep improving.
- `NO_TIMING_SIGNAL`: insufficient comparable observations, effectively flat
  prices, or a market direction the module has not calibrated.
- `PASS`: non-priority, stale/missing, past kickoff, exceeded bet-to threshold,
  or missing conservative price cushion.

The action labels are research **execution-monitoring hypotheses**, not live
bet instructions. All bets remain subject to the existing release gate (zero
approved production units in RESEARCH).

## Exports

`outputs/edge_priority.csv`, `outputs/edge_actionable.csv` and their Docs
copies include `timing_action`, `timing_reason`, prior same-book odds and
spread, changes, quote age, source, and observed timestamp. The edge dashboard
displays timing next to edge priority. The machine-readable
`outputs/edge_timing_report.json` provides a descriptive *first-to-last
observed pre-kickoff* survival rate when enough finished games exist.

The last captured quote is **not an official closing line**. This report does
not establish CLV, profitability, or superiority of BET NOW over WAIT.
Actual timing-strategy validation requires independent forward observations,
chronological holdouts, same-book executable-price comparisons, and sufficient
sample sizes. Do not promote these heuristics on the basis of in-sample
description alone.

## Prospective strategy validation (schema v1)

The model now appends `history/timing_decisions_v1.csv` at the moment it
computes a **clean, positive-cushion, pregame** BET_NOW_RESEARCH or WAIT_MONITOR
signal. The first such observation per `game_id + market + side` wins and
is never replaced by a later price, sportsbook or revised projection. No
historical model snapshots are re-labeled retroactively as timing signals.

The independent `grade_live.py` job publishes the following:
- `outputs/edge_timing_forward_graded.csv` (also in `reports/` and `docs/`):
  one immutable initial decision per row, plus matched forward same-book offers.
- `outputs/edge_timing_forward_performance.json` (also in `reports/` and
  `docs/`): sample counts, missingness, ambiguous line/price trades, action-
  specific directional accuracy and descriptive Wilson intervals.

**Pre-registered observations:** The primary endpoint is the **first** observed
same-book quote from 6 to 9 hours after the frozen decision, observed before
kickoff. A time window without an eligible quote stays missing; we never take
the best price from the window. The secondary endpoint is the **last**
observed same-book quote in the final 60 minutes before kickoff. If no such
snapshot exists we report missing, not the last earlier snapshot as "close."
This is an *observed near-kickoff proxy*, not an official closing market.

Quotes at other books are never matched. An observed spread improvement
(e.g. +0.5 points for the backed team) combined with worse American odds
is a mixed tradeoff and **not scored as an automatic timing win**. For the
same line, at least one percentage point in implied break-even price is
required for a directional improvement/deterioration. No return, bet sizing,
CLV, or timing-based profit is inferred from line improvement alone.

**Research-only gate:** Report remains PENDING/EARLY/COLLECTING until at least
100 conclusive observations, 80 distinct games, 8 distinct kickoff weeks and
20 conclusive observations *of each action*. Even then the only transition
is `REVIEW_READY_NOT_APPROVED`: full independent statistical and economic
validation, reliability/availability assessment and human review are necessary.
No automatic strategy promotion or production staking occurs.

The directional Wilson confidence intervals are descriptive and do not adjust
for within-game clustering or data-availability selection. Missing snapshots,
sportsbook quote timestamps, stale prices, and observation frequency must
be monitored before drawing an operational conclusion.
