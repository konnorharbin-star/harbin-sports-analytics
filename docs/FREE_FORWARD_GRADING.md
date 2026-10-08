# Free Forward Research Grading

This free, recommendation-only postgame audit evaluates archived football model suggestions. It cannot place wagers, deposit funds, use paid feeds, or connect betting accounts.

## Inputs and cadence

- Immutable first-seen pregame model observations from the public `ops-evidence` branch.
- Public `reports/live_graded_bets.csv` published by each football model. Scores are model-authored grading data, not independently certified official results.
- GitHub Action `Free Forward Research Grading`: manual dispatch plus daily 13:29 UTC schedule. It runs with `contents: read` and uploads JSON and Markdown artifacts only.

## Prespecified evaluation rules

- Freeze the earliest timestamp-valid observation per league, game ID and market (moneyline/spread/total) before knowing the result.
- Exclude missing or stale quotes, missing bookmaker or odds, observations at/after kickoff, invalid probabilities, and inconsistent identity.
- Match postgame results by exact game ID, away/home identity, and kickoff within five minutes. Conflicting results are discarded rather than selected.
- Grade moneyline, spread, and total outcomes using full American-odds payout; pushes return zero profit. Moneyline ties are not graded because settlement varies by book.
- Keep research quality blockers. A graded result does not certify an executable pregame price, validated model EV, or production-ready model.

## Output interpretation

`forward-research-grades.json` and `forward-research-grades.md` show hypothetical one-unit ROI, wins/losses/pushes, unresolved games, and provenance. The candidate set is a selected research watchlist and may not represent all games or markets. Multiple markets for the same game are correlated. Small samples or positive hypothetical returns do not establish a profitable betting edge.

## Free, manual-only operating requirements

- Python standard library, GitHub-hosted infrastructure and the two existing public model result ledgers. no paid data subscriptions.
- This workflow never mutates a repository, never makes sportsbook API calls, and never executes transactions.
- If an input is unavailable, the grader fails visibly rather than inventing a score or fill.
- The first run is triggered by this workflow file being merged to `main`; subsequent collection is daily or manually initiated.

## Run locally

```bash
python -m platform_ops.grade_observations \
  --archive-root /path/to/ops-evidence/history/free-observer \
  --json-out grader-artifacts/forward-research-grades.json \
  --markdown-out grader-artifacts/forward-research-grades.md
```
