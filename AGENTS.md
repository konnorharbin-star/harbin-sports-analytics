# Harbin Sports Analytics — Agent Rules

- Never use future information in historical features.
- Use chronological/walk-forward validation; never random-split time-series claims.
- Keep sportsbook prices out of the core score projection. Compare to market only after producing fair margin/total.
- Preserve unrounded model margin and total internally; round only for display.
- Keep the Jason-Cooper-style visible layer separate from improved no-vig/EV diagnostics.
- Do not claim Jason Cooper's private score-generation formula is known.
- Run `python -m pytest -q` before merging changes.
- Never commit secrets, caches, virtualenvs, or binary model artifacts.
