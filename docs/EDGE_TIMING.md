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
