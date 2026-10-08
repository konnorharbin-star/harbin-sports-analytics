# Free Same-Book Pregame Line Movement

## Purpose

This research-only audit compares the earliest valid pregame recommendation preserved by the [free observer](FREE_RESEARCH_OBSERVER.md) with the **latest eligible pre-kickoff observation at the same sportsbook** already captured by your NFL/CFB model infrastructure.

**It does not verify an official bookmaker closing price, actual bet execution or profitable betting.** It is more conservative than the existing cross-book consensus CLV proxies and does not substitute an entirely different book for a missing quote.

## Source data — free and existing

- Original observation: read-only records in `ops-evidence/history/free-observer/YYYY/MM/DD.jsonl`, including exact game ID, market, side, point line, American odds, sportsbook name, quote timestamp and observation timestamp.
- College football: `history/market_snapshots.csv` on the public CFB model `main` branch; normalized `market_quotes_json` contains an individual provider/book, per-side line/odds and `last_update`. Missing source-origin timestamps and optional `odds_api` source rows are excluded.
- NFL: `history/market_snapshots.csv` on the public NFL model `main` branch; each record contains a complete paired two-way observation, named sportsbook/provider, capture timestamp and market/sides. The NFL quotes have a **capture timestamp**, not a distinct bookmaker-origin `last_update`; this limitation is recorded in output.
- These histories may have gaps, be under-sampled near kickoff, include named providers with uncertain execution ability, or be subject to source terms and public-endpoint availability. A code success does not establish a fillable price.

## Strict acceptance rules

1. First freeze the original research entry from the archived observer using its existing first-seen rule; no retrospective backfill of fake pregame recommendations.
2. Require exact league, game ID, market, side and case-normalized sportsbook name. NFL provider information is kept on the closing observation; the original observer does not freeze source/provider IDs, so no provider-identity certainty is claimed.
3. Original quote must have a proper timezone-aware timestamp before kickoff, a valid American price and a valid point line where applicable.
4. The later quote must be observed **after** the original research observation, be updated after the original price timestamp, and be strictly before kickoff. Both capture and underlying update timestamp must fall within 90 minutes before kickoff. A quote timestamp more than two minutes after capture is rejected.
5. For college football require actual `last_update` source time. A synthetic `captured_at` timestamp is not treated as source-origin evidence. For NFL only verified original paired snapshot capture is available.
6. Reject ambiguous simultaneous final quotes, stale snapshots, missing sportsbooks, invalid odds and no-book/no-line records. **Never** use a consensus market proxy when the matching book is unavailable.
7. Report point-line advantage separately from the American-odds movement. When the point line changes, **do not** pretend the odds difference alone is a comparable implied-probability edge. Only at identical point lines can `same_line_implied_probability_move` be used; this is raw single-side implied probability movement, *not* no-vig probability CLV.

### Definitions

- Spread line advantage: **entry selected-team point spread − later selected-team point spread**.
- Under advantage: **entry total − later total**.
- Over advantage: **later total − entry total**.
- Moneyline / identical-point-line price movement: **later book break-even probability − original book break-even probability**.
- A positive movement is favorable to the entry by definition; no margin conversion is implied and settlement/push likelihood is not modeled in this comparison.

## Output and running

`Free Same-Book Line Movement Research` runs daily at 12:41 UTC, on manual dispatch, and once on merge when the workflow file changes. Uses Python standard library, existing public GitHub sources and included Actions. It has **read-only repo permissions**.

Artifact files: `samebook-movement.json` and `samebook-movement.md`, available in GitHub Actions for up to 30 days.

```bash
python -m platform_ops.samebook_movement \
  --archive-root /path/to/ops-evidence/history/free-observer \
  --json-out movement-artifacts/samebook-movement.json \
  --markdown-out movement-artifacts/samebook-movement.md
```

No sportsbook username, API key, wallet, funding, billing, placement or staking operations are permitted. Data providers remain free; historical commercial vendor endpoints are never contacted by this workflow. If existing public history grows beyond the bounded 30 MB download limit, fail visibly rather than upgrade to a paid service or silently truncate.

**Never automatically bet. Never introduce paid dependencies.** All outputs are for human research only.
