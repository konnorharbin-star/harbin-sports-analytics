# Unified Model Command Center (CFB + NFL)

This is a **read-only** public operations dashboard at [command-center.html](command-center.html), designed to publish with the existing GitHub Pages `docs/` site after the pull request is merged.

## What it actually does

- Reads each repository's `main/docs/audit_snapshot.json` from public GitHub raw content.
- Displays the audit and reconciliation statuses, data quality, season/week, monitoring score, release mode, historical evidence and approved staking units.
- Queries recent public GitHub Actions runs and shows the latest matching **main** workflow result (CFB Model + Dashboard / NFL Model + Operations).
- Shows up to five release blockers and any reconciliation errors per model.
- Shows missing/malformed/stale inputs as UNKNOWN instead of silently reusing old values.
- Links to each repository's existing weekly board, audit report and Actions page.

## Important interpretation rules

- **RESEARCH/PAPER is not permission to place a wager.** A successful workflow means code executed, not that an edge exists.
- The NCAA evidence cell shows *qualifying* historical bets; the NFL evidence cell explicitly displays *research archive* ROI and cannot be treated as verified profitability.
- A raw sportsbook quote count is not a count of game × market model candidates. The NFL reconciliation defect discovered on 2026-10-08 is fixed separately in harbin-nfl-analytics PR #142.
- Approved units read directly from the publication snapshot. The page **never** grants staking authorization, calls a bookmaker, invents prices, or changes either model.
- A report older than 24h is flagged as potentially stale. Sports may have idle days; this flag is informational and not a run scheduler.
- The page fetches public GitHub endpoints client-side. GitHub raw caching, rate limits, Pages deployment delay, and temporary CORS/network failures can affect freshness. Missing or malformed sources fail closed in the display.
- No GitHub token or other secret should be embedded in the page. If repositories are made private, switch to a server-side authenticated aggregator rather than storing credentials in client-side JavaScript.

## Deployment and smoke test

1. Merge the PR after repository tests pass.
2. Confirm that GitHub Pages continues publishing the `docs/` directory from `main`.
3. Open `https://konnorharbin-star.github.io/harbin-sports-analytics/command-center.html` from desktop and mobile.
4. Verify that both audit dates and statuses match the current published `docs/audit_snapshot.json` in each repository.
5. Confirm the NFL status is FAIL when the publication audit contains the known row-count mismatch; it should not turn green just because the workflow run succeeded.
6. If the NFL correction is merged, rerun the normal NFL model workflow to regenerate the audit; the control center then picks up the new published snapshot.
7. Temporarily disable network access in browser developer tools and refresh; the display should change to UNKNOWN rather than showing fabricated metrics.

## Next operational expansions

- Add server-side authenticated snapshots if either repository becomes private.
- Track data-source failures and forecast freshness from the canonical audit contract.
- Add alerting on meaningful state transitions instead of indiscriminate hourly notifications.
- Build a cross-sport schema adapter before onboarding NBA/NCAAB/MLB/NHL, without copying NFL calibration weights or promoting unverified historical results.
