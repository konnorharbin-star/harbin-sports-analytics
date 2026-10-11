# Every FBS game's market-edge status — full-slate research audit

The main CFB model now performs one holistic, scheduled run for every remaining
regular-season matchup involving at least one FBS program. A qualified-edge
shortlist alone previously hid **why all other matchups were absent**. The
additional `outputs/full_fbs_edge_scan.csv` and `.json` remedy that visibility
gap, with **exactly one row for every upcoming FBS game**, including FBS/FCS
games that are always unvalidated for wagers.

## The actual distinction

- **The model's predicted winner/score/margin** is calculated independently
  of a sportsbook line using historical training data. These estimates are
  not guaranteed to outperform the market.
- **Raw model EV**, when a quoted selection exists, is a hypothesis generated
  by the original model and may be severely overconfident. If price or model
  selection is missing, raw EV is **null**, not fabricated.
- **Full-slate research status** is one of `RAW_EDGE_HYPOTHESIS_UNVERIFIED`,
  `NO_MODEL_EDGE_HYPOTHESIS`, or
  `UNVALIDATED_OPPONENT_CLASS_NO_BET`. It is **never** an approved bet.
- **Qualified historical edge shortlist** comes from
  `outputs/final_edge_board.csv` and its conservative evidence gate,
  independently applied to market candidates. This list can be much
  shorter than the full slate; `final_edge_decision=NOT_SHORTLISTED`
  is explicitly disclosed for other games.
- **Proven economic edge:** as of this change, none; 3,439 historical CFB
  retrospective entries averaged approximately **−2.04%**, with **zero**
  independently verified point-in-time executable opening quotes in the
  production evidence audit. The prior 2025 subgroup "holdout" was previously
  inspected and cannot be treated as a fresh prospective test. No clean
  forward bet-sample or independently validated sportsbook CLV yet.

The full-slate report does **not** alter stake sizes, production release,
the original model, paper recommendations, opponent ratings or portfolio.
Even a +50% raw EV never becomes an approved wager from this research
report. Raw spread EV may neglect integer spreads' exact push probability;
the separate frozen 3/7 research engine calculates genuine win/push/loss
but has no verified economic advantage yet.

## One-click or scheduled steps

Go to
[Actions → CFB Model + Dashboard](https://github.com/konnorharbin-star/harbin-sports-analytics/actions/workflows/cfb-model.yml)
and click **Run workflow**. Leave season/week empty to select the next
upcoming week, or specify both numbers. The existing recurring schedule
also runs this same complete slate automatically.

At the end of each successful run, open the **Every-FBS-game edge scan**
job summary. Key counts show: total future FBS games scanned, raw research
hypotheses, final evidence shortlist and **zero proven bet authorizations**
until independently verified research establishes an advantage.

For each game, `outputs/full_fbs_edge_scan.csv` discloses: exact projected
score and spread, public/reference market spread when available, point
disagreement, raw model hypothesis (with source book and selection), whether
it made the conservative edge shortlist, conservative EV **only where the
shortlist computed it**, provider quote provenance flags, FBS/FCS class,
release state, and explicitly named blockers.

A check reconciles every game against
`outputs/schedule_coverage_audit.json`; a game that has started is never
backdated to become a pregame wager. If any scheduled upcoming FBS game is
missing, an unexpected game appears, or the scores fail their own margin
consistency, the report **fails**, not silently omits games.

This automated audit reads already-generated public-data-backed model rows
and sourced edge evidence from GitHub Actions. It cannot schedule or invoke
ChatGPT's interactive web search. A separate research conversation can
source additional pages and commit genuine historical webpage observations.

## Developer command

```bash
python -m pytest -q tests/test_walters_full_fbs_edge_scan.py
python -m scripts.walters_full_fbs_edge_scan \
  --shortlist outputs/final_edge_board.csv \
  --coverage outputs/schedule_coverage_audit.json \
  --gate outputs/release_gate.json \
  --out-csv outputs/full_fbs_edge_scan.csv \
  --out-json outputs/full_fbs_edge_scan.json
```
