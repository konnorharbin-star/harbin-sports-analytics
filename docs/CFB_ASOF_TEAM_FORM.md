# CFB as-of team-form challenger (research only)

## Purpose

Evaluate whether a team's previous outcomes versus pregame forecasts contain **incremental** information beyond the sportsbook's no-vig paired moneyline probability. This is not an excuse to force an underdog; if the historical adjustment fails chronological tuning, the challenger equals the market.

## Inputs and chronology

Uses matched game-level CFB results from `scripts/build_cfb_moneyline_diagnostic.py`, which now retains archived pregame home/away teams, independent model projected margin/total, and final margin. Every future-week feature is computed by `scripts/market_form_challenger.py` using earlier fully completed game **weeks** only, never same-week results.

- **Market surprise** for a team's earlier game: realized win (0/1) minus its then-market no-vig win probability; the away team's surprise has the opposite sign.
- **Score-margin surprise:** actual margin minus the independent model margin, oriented home/away.
- **Team form:** shrunk means of each surprise over the prior 3 or 6 observed paired games, denominator `observations + 4`. Home minus away forms are used to estimate the future home-win market residual.
- Form state resets at the beginning of each season. Entire weeks are frozen before importing that week's results.
- Directional features are antisymmetric under home/away swaps.

## Candidate families and selection

Prespecified candidate families comprise a market favorite-calibration term, independent-vs-market disagreement, short-term team market surprise, previous score errors, close-game interactions, and projected total interactions. Ridge penalty is limited to `{0.03,0.3,3,30}`.

- Train on the **2023** historical season.
- Choose a family and penalty on **2024 only**, requiring at least 0.0003 improvement in 2024 Brier score, no lower 2024 straight-up accuracy, and no higher 2024 log loss.
- If no candidate qualifies, select **market-only**. The zero-adjustment fallback is intentional.
- Refit only the frozen chosen family on **2023+2024** and evaluate once on **2025**, with Brier score, log loss, winner accuracy, close-game accuracy, and confidence-pool points.

The 2023–2025 archive has already been inspected in earlier research. Consequently 2025 is a retrospective scorecard, **not** a pristine holdout. Even a historical improvement would require independent forward validation and week-clustered uncertainty before promotion.

## Run it

GitHub Actions → **CFB Market-Error Research**, or run:

```bash
python -m pytest -q tests/test_market_error_challenger.py tests/test_market_form_challenger.py
python -m scripts.build_cfb_moneyline_diagnostic --out /tmp/cfb_paired_research.csv
python -m scripts.market_form_challenger --historical /tmp/cfb_paired_research.csv --out reports/market_form_challenger.json
```

This outputs an auditable JSON report and adds it as an Actions artifact. It **does not** change official forecasts, stake sizing, pricing policy, release gates, or bets. Original quote timestamps are unverified; no result establishes executable profits.

## Forward requirements

For 2026 and later, log time-stamped first-seen model forecasts and paired contemporaneous market prices *before* kickoff; compare challenger against opening and updated market snapshots at matching observation times. Grade identical future game cohorts, including close games and confidence-pool points. No change becomes the live selection rule until it exceeds the market prospectively.

## First retrospective outcome — October 31, 2026

The PR research workflow executed on a matched sample of 778 games from 2025.
The 2024 tuning sample chose the six-game team-form family, ridge penalty 0.3,
with a tuning Brier improvement of 0.001152. That tuning gain did **not**
generalize to the following season:

| 2025 identical 778 games | Archived market | Form challenger |
| --- | ---: | ---: |
| Correct winners | 562 | 558 |
| Win accuracy | 72.2365% | 71.7224% |
| Brier (lower is better) | 0.1819984 | 0.1835276 |
| Log loss (lower is better) | 0.5387499 | 0.5426183 |
| Confidence-pool points | 17,933 | 17,869 |
| Close-game correct (286) | 57.69% | 56.29% |

**Decision:** REJECT form adjustment. Continue using the market-only benchmark.
No second round of retuning based on 2025 outcomes is permitted. The tested
market observations come from the archival opening-fields/final-fallback price
series and do NOT establish a specific verified bookmaker's closing-line
accuracy or an executable trading advantage.

The code stays in the repository as a **failed research hypothesis with
reproducible evidence**. A genuinely new forecasting idea must be
pre-registered and evaluated on new pregame observations, not retrospectively
optimized to this already-reviewed 2025 result.
