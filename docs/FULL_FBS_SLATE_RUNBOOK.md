# Run the entire FBS slate: one button or automated GitHub schedule

This is **one model run for every not-yet-started game in a selected FBS
week**, not one run per individual football game. Games with at least one
FBS school are included, **including FBS vs FCS opponents**. Games with no
FBS participant are excluded.

## One click on phone or desktop

1. Open https://github.com/konnorharbin-star/harbin-sports-analytics/actions/workflows/cfb-model.yml
2. Tap **Run workflow** on the default `main` branch.
3. Leave season and week blank to auto-detect the upcoming college football
   week, or set `season=2026` and `week=7` to run exactly that entire
   week. No matchup names need to be entered.
4. Open the completed job's **Full FBS slate coverage** step summary.
   `outputs/cfb_model_<season>_week<week>.csv` and the linked dashboard
   contain one row per remaining pregame FBS fixture. `outputs/README.md`
   links to model report cards.
5. The script `scripts/audit_schedule_coverage.py` reconciles the saved
   model rows against the original nationwide schedule; a missing upcoming
   FBS game fails the run instead of silently returning a partial card.

## Scheduling

The existing GitHub Actions model workflow runs **automatically at
11:17, 17:17 and 23:17 UTC every day**, with an extra Saturday
14:47 UTC run. GitHub Actions cron always uses UTC, and occasional
schedule delays may occur. Each run auto-selects the next upcoming week.
These runs do not require ChatGPT to remain open, and the user can rerun
manually at any time using the same Actions button.

The current model projects all **future** FBS-involved games for that
one week. It must **not create post-kickoff or backdated predictions**.
As games begin, later runs exclude started/completed games, while the
`history/first_seen_pregame_predictions.csv` ledger retains their first
pregame forecasts from earlier runs (when available). The coverage audit
reports `COMPLETED_EXCLUDED` and `STARTED_EXCLUDED` separately and counts
started/completed games that lack a first-seen archive. It cannot claim
every completed game was forecast beforehand unless there is evidence.

## Exact data scope and reliability

- The public sportsdataverse/cfbfastR schedule includes a game if
  **either** `home_division` or `away_division` is FBS. This fixes
  the previous bug requiring **both** to be FBS and excluding FBS/FCS.
  Missing division columns fail closed rather than including FCS/FCS.
- The original model remains trained on FBS/FBS history. Its projections
  for FBS/FCS games are necessarily less supported: each is marked
  `FBS_VS_NON_FBS_UNVALIDATED` with
  `fbs_model_validation=NO_BET_UNVALIDATED_OPPONENT_CLASS`. It gets a
  projected score, but **zero stake, no edge promotion and PASS signals**.
- FBS/FBS games keep the original model family, probability calibration,
  injury/market caveats, and existing edge release gates. No model is
  retroactively improved by this coverage change.
- Public sportsbook prices can be missing or stale. The model still
  projects games in **projection-only** mode; no made-up lines, fabricated
  executable prices, API secrets or guaranteed picks.
- Weekly FBS coverage is **not** equivalent to proving a profitable
  betting advantage. Never automatically place sports wagers.

## Key outputs

- `outputs/schedule_coverage_audit.json`: exact season/week, total
  FBS games, FBS/FCS count, predicted future count, started/completed count,
  missed upcoming games, first-seen archive counts and coverage status.
- `outputs/schedule_coverage_audit.csv`: a row per source fixture
  with game ID, teams, kickoff, division scope and precise audit status.
- `outputs/cfb_model_<season>_week<week>.csv`: actual model outputs for
  every pregame FBS fixture; each contains a `fbs_matchup_scope` and
  `fbs_model_validation` classification.
- `history/first_seen_pregame_predictions.csv`: first published
  model snapshots, with opponent category frozen for new entries.

## Manual command (same whole-week model)

```bash
python run_week.py --season 2026 --week 7
python -m scripts.audit_schedule_coverage
python -m scripts.archive_first_seen_predictions
```

The GitHub Action additionally runs the existing full test suite,
market provenance reports, publication validation, release gate and
generated-state publication. The GitHub workflow is the easiest routine
way to run all those steps.

**Important:** the scheduled GitHub Actions model uses its own
existing free public data loaders; it cannot invoke ChatGPT's interactive
web-search session invisibly. Its web-researched quote audit checks prior
immutable webpages captured during separate research sessions.
