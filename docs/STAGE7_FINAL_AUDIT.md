# Stage 7 — Final Repository Audit and Evidence Integrity

Stage 7 is the repository-wide audit after Stages 1–6. Its purpose is not to add another predictive feature. It verifies that chronology, market timing, forward evidence, execution provenance, release gates, and published reporting all tell the same story before any production-ready claim is possible.

## Audit conclusion

The engineering stack can pass this audit while the betting model remains **RESEARCH**, **PAPER**, or **SHADOW**. A green workflow is evidence that the software contract executed successfully; it is not evidence of a profitable betting edge.

Production remains impossible unless the hard release gate independently clears model calibration, verified-entry historical evidence, multi-book breadth, portfolio-verified forward evidence, and the execution/risk controls.

## Stage-by-stage audit

### Stage 1 — data and pregame features

Verified contracts:

- historical training consumes completed games only;
- target-week games are excluded from training history;
- ratings and dynamic advanced state are snapshotted before the current week is consumed;
- byes carry the latest already-known state forward rather than importing a future game;
- sportsbook prices are not fair-score model features;
- current-only roster, injury, and weather information is not backfilled into historical score training.

### Stage 2 — walk-forward model selection and calibration

Verified contracts:

- model partitions use whole chronological week boundaries;
- residual weight tuning occurs before probability calibration;
- the final evaluation block is untouched by model selection;
- learned residual weights can be zero;
- production refits preserve the tune-selected weight rather than retuning on evaluation outcomes;
- MAE/RMSE and Brier/log-loss/ECE are published separately from betting results.

### Stage 3 — market, CLV, and multi-book execution

Verified contracts:

- score projections are generated before market comparison;
- one coherent quote is retained per sportsbook;
- best prices/lines preserve sportsbook provenance;
- closing snapshots must be strictly before kickoff;
- realized P&L uses the stored executable American price when one exists.

Stage 7 closes two additional integrity gaps: selected quotes now carry their own update timestamp, and spread/total Quant wagers are not created from a synthetic `-110` price when an actual executable price is missing.

### Stage 4 — weather, QB, injury, and roster context

Verified contracts:

- current context is a post-prediction risk/confidence layer;
- injury reports are deduplicated point-in-time by athlete;
- stale/missing context reduces quality rather than creating fake information;
- indoor games do not receive outdoor weather penalties;
- historical runs do not import current-only context.

### Stage 5 — portfolio and bankroll controls

Verified contracts:

- no dollar bankroll is invented;
- bankroll state is unit-based;
- soft drawdowns shrink risk and a hard stop can force approved stake to zero;
- game/team/market/book/kickoff/slate/bet-count caps are deterministic;
- a production candidate requires an executable book, line/price, and sufficient market provenance;
- actual approved stake stays zero outside an open production path.

Stage 7 strengthens execution checks by requiring a parseable selected-quote timestamp, a strictly pre-kickoff quote, an explicit market-book count, and a configurable quote-age limit.

### Stage 6 — monitoring and publication reconciliation

Verified contracts:

- the public audit page consumes one canonical snapshot;
- output and public snapshot copies must reconcile;
- season/week/platform/release/monitoring/portfolio fields are cross-checked before publication;
- model distribution drift is monitored against the historical walk-forward reference;
- forward audit history is append-only at the reporting layer;
- publication validation is a workflow gate before generated state is committed.

## Stage 7 findings and fixes

### 1. Policy evaluation was participating in policy selection

The prior production-policy code ranked thresholds on earlier seasons and then searched the final season for a threshold that passed. That made the nominal holdout part of selection.

Stage 7 changes policy calibration to:

1. **development** — rank threshold candidates;
2. **tune** — choose among candidates and determine repeated weak-week exclusions;
3. **untouched evaluation** — evaluate the frozen policy only.

Changing only untouched-evaluation outcomes can change whether a market is enabled, but it cannot change the selected threshold or excluded weeks.

## 2. Historical promotion evidence could include archive-final fallbacks

The archive is allowed to use a final quote when an opening quote is unavailable for research diagnostics. Such a fallback cannot prove that the simulated wager was available at decision time.

Stage 7 therefore records market-specific `entry_quote_verified` provenance. Only explicit, non-null archived opening line/price fields for the selected sportsbook/market may enter:

- production policy calibration;
- promotion evidence;
- ROBUST historical release criteria.

Fallback/final archive rows remain visible in all-archive diagnostics and are counted as excluded from promotion evidence.

## 3. Forward evidence was based on raw signals rather than the capped portfolio

The legacy forward grader used the first actionable raw Quant signal. A raw signal can be reduced or removed by Stage 5 portfolio caps, so it is not equivalent to a simulated executed portfolio decision.

Stage 7 adds `history/portfolio_decisions_v1.csv`. It is appended only after portfolio controls run and records cap-constrained PAPER/SHADOW/BET decisions. The independent live grader now prefers this ledger and marks evidence `portfolio_verified=true` only when the portfolio ledger is the source.

The release gate cannot use legacy raw-signal grading to reach PRODUCTION.

## 4. Forward timing now fails closed

A portfolio decision qualifies for independent grading only when both the decision timestamp and kickoff parse successfully and:

`decision_at < kickoff`

Missing, malformed, or post-kickoff decision times are excluded rather than treated as acceptable legacy rows.

## 5. Executable quote timing now fails closed

For production approval, Stage 7 requires:

- actual executable American odds;
- spread/total line when applicable;
- sportsbook provenance;
- explicit market-book count;
- selected quote timestamp;
- selected quote strictly before kickoff;
- quote not in the future beyond a small clock-skew allowance;
- quote age within the configured limit (default 60 minutes).

PAPER/SHADOW planning may still show a hypothetical cap-constrained allocation when an execution field fails, but `execution_ready` is false and real approved units remain zero.

## 6. Historical breadth is now defined by qualified segments

ROBUST historical evidence requires at least 1,000 verified opening-entry bets, positive normalized CLV, and a week-block ROI 95% confidence-interval lower bound above zero. Breadth additionally requires at least two markets and at least two seasons that each have at least 50 bets, positive ROI, and positive CLV.

This definition lives in the evidence report and is consumed by the hard release gate rather than being recomputed with weaker criteria in a second place.

## CI protection

`tests/test_stage7_final_audit.py` verifies that:

- untouched policy evaluation cannot change selected thresholds;
- unverified archive entries cannot enter promotion evidence;
- non-portfolio forward grading cannot satisfy the production release gate;
- post-kickoff decisions are excluded from forward evidence;
- a missing spread price cannot manufacture a Quant wager;
- stale quote timestamps cannot receive production-approved stake.

Existing Stage 1–6 regression tests remain part of the full repository suite.

## What Stage 7 does not do

Stage 7 does not declare the model profitable, does not place wagers, does not custody funds, does not bypass the release gate, and does not convert a readiness score into expected return.

As of the Stage 7 implementation, the last published Stage 6 state was still **RESEARCH** because the chronological calibration ECE exceeded its release threshold, historical ROI uncertainty still crossed zero, historical breadth was insufficient, and independent portfolio-verified live/shadow evidence had not accumulated. Those are empirical blockers, not software defects, and code changes cannot legitimately manufacture them away.
