# Stage 5 — Portfolio, Bankroll, and Execution Controls

Stage 5 turns individual Harbin Quant candidates into a bounded slate-level risk plan. It does **not** create a new predictive signal, change the fair-score model, or bypass the existing RESEARCH/PAPER/SHADOW/PRODUCTION release gate.

## Unit-based bankroll contract

The repository does not know the user's dollar bankroll, so Stage 5 does not invent one. Risk is expressed in betting **units**.

`reports/live_graded_bets.csv` is the independent forward/shadow ledger used for the bankroll throttle. Its flat-stake profit series is converted into:

- cumulative units;
- current drawdown from the running peak;
- maximum observed drawdown;
- trailing-window ROI;
- trailing execution CLV when available.

The release gate remains responsible for deciding whether the live sample is large enough for production. The bankroll layer only reduces or halts risk; it cannot promote the model.

## Drawdown throttle

The default portfolio policy has a soft drawdown threshold and a hard stop.

- Below the soft threshold, the unit multiplier is 1.0.
- Between the soft threshold and hard stop, both proposed stake and portfolio caps are reduced.
- At or beyond the hard stop, production approval is zero.
- A recent window with jointly poor ROI and non-positive CLV receives an additional conservative multiplier.

A later recovery can restore risk because the throttle uses **current** drawdown from the equity peak, not the worst drawdown that ever occurred.

These thresholds are risk-management defaults, not evidence of an optimal betting strategy.

## Production fail-closed rules

A row can receive non-zero `portfolio_stake_units` only when all of the following are true:

1. the hard release gate says `PRODUCTION`;
2. the validated policy says `production`;
3. the independent live/shadow grading ledger exists;
4. the bankroll hard stop is inactive;
5. the candidate has a supported market and side;
6. executable American odds are present;
7. spread/total candidates have an executable line;
8. sportsbook provenance is present when required by policy;
9. the row survives all concentration caps.

A stale release-gate JSON cannot create approved stake by itself if the live ledger has disappeared.

PAPER and SHADOW modes may still show cap-constrained hypothetical allocations, but `portfolio_stake_units` remains zero.

## Concentration controls

Candidates are processed deterministically by risk-adjusted EV score. Stage 5 enforces:

- maximum units for the full slate;
- maximum units per game;
- maximum directional team exposure;
- maximum market exposure;
- maximum sportsbook exposure;
- maximum correlated kickoff-window exposure;
- maximum number of bets;
- minimum executable allocation size.

Directional moneyline/spread bets charge exposure to the selected team, not automatically to both teams. Totals charge both teams because the wager is exposed to the scoring environment of the entire game.

All unit caps scale down with the bankroll drawdown multiplier. This prevents a large candidate pool from refilling a nominal 5-unit slate after the system has decided to reduce risk.

## Sportsbook execution provenance

`quant_book`, `quant_price`, and `quant_odds` are treated as execution fields rather than display-only metadata. A candidate without the required executable sportsbook provenance is blocked instead of receiving an approved stake.

The book cap is applied after line shopping. This prevents every candidate from concentrating at one sportsbook merely because that book happens to own many best prices.

Stage 5 does not automate sportsbook login, wager submission, or settlement.

## Output fields

`outputs/portfolio_card.csv` now includes:

- original `paper_stake_units`;
- `bankroll_adjusted_units`;
- cap-constrained `portfolio_candidate_units`;
- actual `portfolio_stake_units` (non-zero only in production);
- `execution_ready`;
- `portfolio_action`;
- `portfolio_limit_reason`.

`outputs/portfolio_summary.json` adds:

- unit-bankroll risk state;
- base and effective unit caps;
- approved-bet count;
- execution-blocked count;
- book allocation;
- team-exposure allocation;
- cap-hit diagnostics;
- an explicit production block reason when a production gate is open but bankroll/execution prerequisites fail.

## Policy compatibility

`harbin.policy.load_policy` now deep-merges the portfolio block. Older generated policy files therefore inherit new Stage 5 safety defaults instead of silently deleting them when they contain only the legacy portfolio keys.

## CI protection

`tests/test_stage5_portfolio.py` verifies that:

- legacy policy files inherit new safety defaults;
- missing sportsbook provenance cannot be approved;
- single-book concentration caps bind;
- drawdown hard stops block production;
- soft drawdowns shrink unit budgets;
- directional team exposure does not charge the opponent;
- maximum bet count prioritizes higher-ranked candidates deterministically;
- an apparently open production gate cannot approve stake without the live grading ledger.

The full repository test suite must pass before Stage 5 is merged to `main`.

## Stage boundary

Stage 5 outputs a bounded execution plan only. It does not place wagers, custody funds, infer a dollar bankroll, or optimize portfolio weights from an in-sample covariance matrix. Any future automated execution layer would require separate credentials, controls, reconciliation, and audit design.
