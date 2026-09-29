# Harbin Sports Analytics

Automated college-football projection and market-comparison system that lives entirely in this GitHub repository.

## Run the model

1. Open **Actions**.
2. Select **CFB Model + Dashboard**.
3. Click **Run workflow**.
4. Enter a season/week, or leave both blank to auto-detect.
5. Wait for a green check.

Then open **`outputs/README.md`**. It always points to the latest interactive table, CSV/JSON data, and Cooper-style carousel PNGs.

## What v3 fixes

- Live market data is attempted from multiple no-key ESPN hosts before falling back to the SportsDataVerse historical line archive.
- If sportsbook data is unavailable, the model clearly enters **projection-only mode** and does not fabricate betting signals.
- The ML residual layer is automatically shrunk toward the independent baseline; a residual model that is worse on chronological validation can receive a weight of `0.00`.
- Generated output includes explicit market-coverage diagnostics and data-source errors.
- Prediction-history snapshots only append when a projection or market quote actually changes.
- The dashboard and PNGs include their latest Central Time update stamp.
- GitHub Actions uses current Node-24-based checkout/setup actions and cancels overlapping runs.

## Architecture

- `harbin/ratings.py` — leakage-safe opponent-adjusted score/Elo state.
- `harbin/models.py` — margin/total residual models with validation shrinkage guard.
- `harbin/data.py` — schedules/results + live/historical market data fallbacks.
- `harbin/market.py` — Cooper-style replica thresholds plus no-vig/EV math.
- `harbin/render.py` — dark 14-games-per-page HTML/PNG display.
- `harbin/pipeline.py` — end-to-end run, diagnostics, history, and Pages output.
- `run_week.py` — command-line entry point.

## Output

A run for 2026 Week 5 writes the CSV, JSON, HTML, metadata, and four PNG pages under `outputs/`, plus a single `outputs/README.md` landing page.

GitHub Pages is generated from `docs/index.html`.

## Methodology boundary

The visual market layer is a reconstruction from Jason Cooper's public model screenshots. Jason Cooper's private scoring formula has not been published. The underlying score engine here is an independent model and should be judged by its own chronological validation, calibration, closing-line value, and out-of-sample results.
