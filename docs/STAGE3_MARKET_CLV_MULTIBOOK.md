# Stage 3 — Market, CLV, and Multi-Book Execution

Stage 3 turns the existing sportsbook supplement into an auditable execution layer. Score projections remain market-independent; market data is attached only after the fair-score model has produced its predictions.

## Independent sportsbook quotes

`harbin.market_intel.MarketIntelligence` normalizes sportsbook quotes from the available ESPN Core, Action Network, optional The Odds API, and the primary schedule quote.

The layer enforces one coherent quote per sportsbook. In particular, Action Network rows are grouped by `book_id`; moneyline, spread, and total rows from different books are never combined into a synthetic sportsbook quote.

When duplicate observations exist for the same named sportsbook, the freshest quote is preferred, with quote completeness as a deterministic tie-breaker.

## Consensus market

For each game the layer records:

- sportsbook count and source provenance;
- median home spread and median total;
- mean no-vig home win probability across books that have both moneyline sides;
- cross-book standard deviations and ranges;
- a bounded market-consensus quality score;
- serialized normalized quote state for audit/reconstruction.

Consensus values are downstream context only. They do not enter score-model feature construction or model fitting.

## Line shopping

Best executable prices are selected independently by market and side.

- **Moneyline:** highest American price wins.
- **Home spread:** most favorable home line wins; equal lines are broken by the better American price.
- **Away spread:** most favorable away line wins; equal lines are broken by the better away price.
- **Over:** lowest total wins; equal totals are broken by the better over price.
- **Under:** highest total wins; equal totals are broken by the better under price.

Every best-line field has a corresponding sportsbook provenance field. The selected Quant recommendation carries that sportsbook in `quant_book` together with the exact `quant_price` and `quant_odds`.

## Market snapshots

`capture_lines.py` persists hourly audit state without retraining the model. Each snapshot now includes:

- primary-book lines;
- multi-book consensus spread, total, and no-vig moneyline probability;
- best executable line/price by side;
- sportsbook provenance for every best line;
- sportsbook count, contributing source families, and normalized quote JSON.

Near kickoff every scheduled observation is retained even when unchanged. Farther from kickoff, line/price/book changes are retained immediately and unchanged state receives a four-hour heartbeat.

## Closing-line value

A close is valid only when its snapshot has a parseable timestamp strictly before kickoff. Post-kickoff observations are excluded.

CLV uses the latest valid pre-kickoff market snapshot:

- **Spread:** executed line versus consensus closing spread when available, otherwise the primary closing spread.
- **Total:** executed line versus consensus closing total when available, otherwise the primary closing total.
- **Moneyline:** entry consensus no-vig probability versus consensus closing no-vig probability when available. A separate execution CLV compares the actual executed American price with the fair closing probability.

Positive CLV means the entry beat the closing benchmark in the direction of the wager. The live grading report records the CLV source, closing sportsbook count, and closing snapshot timestamp.

## Realized P&L

Grading now uses the actual stored `quant_odds` for moneyline, spread, and total wagers. The old implicit `-110` assumption is retained only as a fallback for legacy spread/total snapshots that predate executable-odds storage.

## CI protection

`tests/test_stage3_market_clv.py` verifies that:

- mixed Action Network book IDs cannot be merged into a synthetic quote;
- best-line selection uses correct side-aware line shopping and price tie-breaks;
- sportsbook provenance is preserved through recommendation selection and snapshots;
- consensus close takes precedence over a single primary book for CLV;
- actual executable odds determine realized profit;
- closing snapshots are strictly pre-kickoff;
- consensus line history records opening and closing multi-book state.

The full repository test suite must pass before Stage 3 is merged to `main`.

## Stage boundary

Stage 3 does not add weather, quarterback, injury, or roster adjustments; those belong to Stage 4. It also does not redesign portfolio caps or bankroll controls; those belong to Stage 5.
