# Harbin Sports Analytics

One GitHub repository for the automated CFB model, prediction history, Jason-Cooper-style display, and GitHub Pages site.

## What it does

- Downloads historical/current FBS scoreboards from ESPN's public JSON endpoint.
- Builds leakage-safe pregame team ratings chronologically.
- Trains independent home-margin and game-total models.
- Produces projected scores and margin-derived WIN %.
- Compares the model to available moneyline, spread, and total markets.
- Applies reconstructed LEAN / BET / STRONG display thresholds.
- Stores timestamped prediction snapshots for line movement / CLV analysis.
- Generates 1320x690 carousel PNGs plus an interactive dark dashboard.
- Runs automatically using GitHub Actions and deploys with GitHub Pages.

## Run it

Open **Actions → CFB Model + Dashboard → Run workflow**. Leave season/week blank for auto-detect, or enter a specific season/week.

The workflow also runs automatically several times per day.

## Important distinction

Jason Cooper's private scoring formula is not public. The visible probability/market/display behavior is reverse engineered from public screenshots; the score engine in this repository is our independently trained implementation.
