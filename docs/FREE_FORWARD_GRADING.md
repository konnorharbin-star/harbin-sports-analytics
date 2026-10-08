# Free Forward Research Grading

This free, recommendation-only postgame audit evaluates archived football model suggestions. It cannot place wagers, deposit funds, use paid feeds, or connect betting accounts.

## Inputs and cadence

- Immutable first-seen pregame model observations from the public `ops-evidence` branch.
- Public `reports/live_graded_bets.csv` published by each football model. These are model-authored grading data, not independently certified as received.
- An independent check from ESPN's public final scoreboard data (unofficial, undocumented endpoint). The grader requires exact completed-final status, home/away identity, game ID or teams, kickoff alignment, and agreement with the model's own recorded final margin and total. ESPN reporting is not official league certification. If ESPN is unavailable or disagrees, the record is not graded.
- GitHub Action `Free Forward Research Grading`: manual dispatch plus daily 13:29 UTC schedule. It runs with `contents: read` and uploads JSON and Markdown artifacts only.

## Prespecified evaluation rules

- Freeze the earliest timestamp-valid observation per league, game ID and market (moneyline/spread/total) before knowing the result.
- Exclude missing or stale quotes, missing bookmaker or odds, observations at/after kickoff, invalid probabilities, and inconsistent identity.
- Match model-authored postgame results by exact game ID, away/home identity, and kickoff within five minutes; also verify ESPN's completed final with strict game identity and kickoff timing. If either source is missing or the two scores disagree, withhold the grade.
- Grade moneyline, spread, and total outcomes using full American-odds payout; pushes return zero profit. Moneyline ties are not graded because settlement varies by book.
- Keep research quality blockers. A graded result does not certify an executable pregame price, validated model EV, or production-ready model.

## Output interpretation

`forward-research-grades.json` and `forward-research-grades.md` show corroborated final-score status, disagreement/missing-source counters, and hypothetical one-unit ROI, wins/losses/pushes, unresolved games, and provenance. The candidate set is a selected research watchlist and may not represent all games or markets. Multiple markets for the same game are correlated. Small samples or positive hypothetical returns do not establish a profitable betting edge.

## Forward probability calibration (research only)

The audit also calculates **Brier score** and **binary log loss** for first-observed recommendations whose results are corroborated by ESPN final scores. It separates NFL and college football, and moneyline/spread/total markets.

- Only independently score-verified win/loss observations count; pushes and missing/invalid probabilities are excluded.
- Output always includes number of verified observations and distinct games. Below 100 scored observations, calibration remains `INSUFFICIENT_SAMPLE`; the expected calibration error (ECE) is hidden.
- At least 100 scored observations are needed before descriptive ECE is shown, and reliability bins with fewer than 10 observations are withheld. This is **not** a production-profitability threshold.
- Binary probability metrics are exploratory because push exclusion changes the sample, multiple market observations can share a game, and model probabilities may include push mass. Neither Brier/log loss nor ECE demonstrates an executable or profitable betting edge.
- The calibration calculations do not retrain the model, adjust odds, change betting recommendations or authorize stakes.

## Free, manual-only operating requirements

- Python standard library, GitHub-hosted infrastructure and the two existing public model result ledgers. no paid data subscriptions.
- This workflow never mutates a repository, never makes sportsbook API calls, and never executes transactions.
- A network outage, canceled/in-progress event, incomplete scoreboard, mismatched final score or ambiguous ESPN game match yields **no grade** for the affected game. Free/public ESPN endpoints are undocumented and can change or impose limits; there is no guaranteed availability or billable fallback.
- This verification is postgame only. It does **not** verify whether any quoted sportsbook price was actually executable at a historical timestamp.
- If the preexisting model results ledger is unavailable, the job fails visibly; missing independent ESPN results cause affected records to remain ungraded, not to be assigned fabricated outcomes.
- The first run is triggered by this workflow file being merged to `main`; subsequent collection is daily or manually initiated.

## Run locally

```bash
python -m platform_ops.grade_observations \
  --archive-root /path/to/ops-evidence/history/free-observer \
  --json-out grader-artifacts/forward-research-grades.json \
  --markdown-out grader-artifacts/forward-research-grades.md
```
