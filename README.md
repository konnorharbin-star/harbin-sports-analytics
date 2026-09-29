# Harbin Sports Analytics

Automated CFB projection platform with a Jason-Cooper-style public display and a separate, more rigorous internal quant layer.

## V2 architecture

The model now uses a leakage-safe opponent-adjusted scoring engine rather than simple raw points-for/against. Each team carries pregame offense, defense, Elo, form, volatility and schedule-strength states. Offensive and defensive states are updated from scoring residuals *against the opponent's rating*, then an ensemble model predicts residual margin and total around an independent baseline score projection.

The sportsbook market is **not** an input to the score projection. Lines are compared only after the model creates a fair score, margin and total.

### Cooper Replica layer

The public card intentionally mirrors the behavior reverse-engineered from Jason Cooper's public screenshots:

- hidden decimal score projections, rounded only for display
- WIN % ≈ normal CDF of projected margin with sigma 16.41
- moneyline badge based on model probability minus raw implied probability
- spread badges at roughly 2 / 4 / 6 points of disagreement
- total badges at roughly 2.5 / 4.5 / 7.5 points of disagreement
- LEAN / BET / STRONG visual labels
- 14 games per 1320×690 carousel page
- dark alternating rows, blue WIN% bar, winner bolding, muted inactive markets

These are reconstructed public behaviors, not claimed proprietary Jason Cooper internals.

### Harbin Quant layer

The CSV/JSON also retains calibrated probability, no-vig market probability, fair odds, estimated ML ROI, both-side ML edge checks, cover probability, total probability, baseline projections and validation metrics.

## Run

Open **Actions → CFB Model + Dashboard → Run workflow**, enter season/week or leave them blank for auto-detect, then click **Run workflow**.

Generated files appear under `outputs/`, including the CSV/JSON and `page1.png`, `page2.png`, etc. The latest browser dashboard is written to `docs/index.html` for GitHub Pages.

## Validation

The workflow runs all tests before generating a card. Historical validation is chronological rather than randomized to avoid time-series leakage.
