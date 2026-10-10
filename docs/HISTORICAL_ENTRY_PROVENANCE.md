# Historical quote provenance: research vs. promotion

**Status: Research only.** A positive simulated return on the historical archive
must not be described as an executable or proven edge without credible prices at
a documented decision time.

## Why archive opening fields are insufficient

The upstream CFB odds archive contains opening lines and prices, but those
fields alone do not establish the sportsbook's original publication time.
A historical collector's download timestamp is not the time the sportsbook
offered a quote. A game's kickoff timestamp is not a quote timestamp.

The archive therefore remains suitable for research on predictive error and
hypothesis generation. It cannot, by itself, demonstrate that a backtested bet
was available and executable when the model would have selected it.

## Strict admission requirements

For **each selected sportsbook and market**:

1. Require a consistent identified sportsbook and a complete original opening
   price for both relevant sides: home/away moneyline, both team spreads, or
   over/under. Duplicates of the same side or a quote from a different book
   cannot fill missing data.
2. Require a provider-origin opening quote publication timestamp, not merely
   an ingestion or download timestamp. The currently accepted explicit fields
   are `opening_quote_published_at`, `provider_opening_quote_at`, and
   `sportsbook_opening_quote_at`.
3. Parse both side timestamps in UTC and require each to precede the game's
   kickoff. Missing, malformed, and post-kickoff times are disqualifying.
4. Require `entry_quote_verified` to contain a literal affirmative flag.
   Text `False`, `0`, `unknown`, blank, or null cannot become a verified
   entry by boolean casting. `used_distinct_open` is not a substitute.
5. Apply the same fail-closed mask to policy calibration, historical evidence,
   and quote-integrity reporting.

The historical backtest now records verification breakdowns by season and
market in `backtest_summary.json` under `quote_integrity`.

## How to generate evidence without subscriptions

Persist **prospective** same-book pregame odds observations in the existing
append-only capture history, including source, named sportsbook, market, side,
handicap, price, observed capture time, independently reported provider quote
time, and kickoff. Ensure the model's prediction and chosen rule are frozen
before the result. Record the subsequent same-book near-kickoff quote without
labeling it an official closing line unless justified.

If a free feed does not report a provider-origin quote timestamp, record
provenance as **unknown** and do not promote those observations. Never create
missing timestamps from a scrape time, an archived opening field, or an
assumption about the market.

## Acceptance and controls

- No changes to the core fair-score forecasts, calibration, or unit sizing.
- No historical or forward profitability claim without untouched,
  time-correct evidence, market-relative validation, CLV, and confidence
  intervals.
- No relaxation of release gates or automated bet placement.

Until the evidence supports a robust improvement, the correct decision is
**NO BET**.
