# Free Research Observer — Zero-Wager Policy

**Default and permanent behavior:** The Harbin research observer collects already-published model outputs, validates their timestamps, produces research watchlists, and records immutable evidence. It **cannot place, submit, reserve, or fund bets**. It has no sportsbook authentication, deposits, payment processing, broker accounts, order endpoints, or staking execution. All wagering decisions are the user's own, outside this project.

## What the free observer adds

- **Two existing, public models**: reads CFB and NFL published `docs/audit_snapshot.json` and `docs/latest.csv` from their default `main` branches. It does not modify either model, select new weights, or re-fit probabilities.
- **Price provenance**: preserves league, matchup, market, side, quoted odds, point line, bookmaker identity, and exact quote time when the source actually provides them.
- **Model evidence**: records model-implied probability and independently recalculated EV; flags mismatches, missing quote times, stale quotes, games already kicked off, and unvalidated injury/edge regimes.
- **Fail-closed release status**: copies audit, reconciliation, data quality, historical release blockers and published approved units. Every candidate is explicitly `RESEARCH_ONLY` even if a model reports high EV.
- **Free storage**: saves the output as JSON, a ranked human-readable Markdown research report, portable SQLite, and date-sharded append-only JSONL under a dedicated `ops-evidence` branch. No third-party database subscription.
- **Scheduled collection**: standard GitHub Actions every six hours plus manual dispatch after the workflow is merged. The job uses public HTTP GETs only and attaches JSON + SQLite as a downloadable GitHub Actions artifact.

All tools are Python 3.12 standard library or standard free GitHub Actions. No Odds API key or other paid source is necessary.

## Critical limitations

- These are **observations of already-published model outputs**, not independent verified opening/closing sportsbook records. A source's timestamp may not prove a fill was executable at that moment.
- The observer does **not** validate the sports models themselves or guarantee a market edge. An advertised `quant_ev` can be grossly overconfident; historical profitability and calibration must be demonstrated separately.
- A positive modeled EV can still appear on the watchlist with blockers. The watchlist is a list of candidates to **review**, not a list of approved wagers.
- No post-kickoff betting: all games already started are excluded from forward watchlists.
- A source over 36 hours old is flagged stale; a quote over 90 minutes old is flagged stale. Both remain archived as evidence but are not silently represented as fresh.
- Data goes to the `ops-evidence` Git branch so automated commits **cannot overwrite either model's `main` outputs**. The archive deduplicates identical source versions per UTC day.
- Because GitHub Actions has rate limits, runner availability and artifact retention limits, this is designed to stay within free usage but is not a promise that GitHub provides unlimited compute or storage.
- The history branch is readable to anyone while the repository is public. Do not archive private bankroll data, credentials, personal identifiers or restricted sportsbook content.

## How to run manually

From repository root on Python 3.12:

```bash
python -m platform_ops.free_observer capture \
  --output observer-artifacts/current-research.json \
  --sqlite observer-artifacts/current-research.sqlite \
  --report observer-artifacts/research-report.md
python -m platform_ops.free_observer archive \
  --snapshot observer-artifacts/current-research.json \
  --archive-root local-free-evidence/
```

The capture fails loudly if a published source is unavailable or malformed. It never fills in missing prices, probabilities, bookmakers or quote timestamps.

After merging this branch, open **Actions → Free Model Research Observer → Run workflow** to perform the first collection. This workflow can also run on its six-hour schedule. The workflow appends to the already-created `ops-evidence` branch and uploads the two output files as a 30-day artifact.

## How to audit a candidate

Each `research_watchlist` entry includes `recalculated_ev`, `blockers`, `quoted_at_utc`, `kickoff_utc`, `source_signal` and `qualification`. A candidate is only an analytical observation. Interpret `blockers` before interpreting the EV. If the NFL audit reports FAIL, or its regime/point-in-time context is not validated, the record visibly preserves those blockers.

### Storage contracts

- `snapshot_id`: SHA-256 of both public source digests.
- `source_sha256`: SHA-256 over audit raw bytes + separator + CSV raw bytes.
- `observed_at_utc`: collector UTC timestamp.
- Archive: `ops-evidence/history/free-observer/YYYY/MM/DD.jsonl`, one JSON record per newly observed source version within that day.
- SQLite: `snapshots` and `research_candidates` tables for downstream analysis, with no funds, stakes, orders or user accounts.
- Versioned format: `schema_version=1`.

The system never represents a model/watchlist entry as an executed bet. Any future result-grading tool must use separately sourced outcomes with accurate timestamps and maintain the same read-only separation.
