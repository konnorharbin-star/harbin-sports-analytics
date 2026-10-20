# College football source verification and operating audit

The model's automatic score updates and live grading were archived on October
11, 2026. The owner requested restoring scheduled updates after validation on
October 19. This change restores only the model/dashboard and paper grading
workflows. Other archived workflows remain paused. The model workflow uses no
paid API keys, keeps the full test gate, checks all upcoming FBS-involved games,
and retains the existing release and immutable first-seen forecast controls.

## Audit findings and repairs

The last published model metadata was generated October 11 for two remaining
Week 6 games. It reported 100% current-context coverage. Inspection found:

- Weather and travel could use the designated home team's stadium without
  verifying the actual event venue, including neutral-site games.
- Any populated national injury feed could give every game injury coverage.
- A flexible column search could select `as_of_date` (the export date) instead
  of the athlete's `injury_date`, making an older report appear fresh.
- Unknown roster statuses could count as active. `Unavailable` could be
  classified as available by a substring match.
- Season-roster membership flags could earn game-availability coverage.
- Weather selected the nearest returned hour without a maximum distance or
  validating units and complete values.
- The pipeline dropped detailed context-quality metadata before publishing it.
- A current-season schedule error could silently fall back to last season.

These paths now fail closed or preserve the uncertainty explicitly. Fair-score
weights and historical training formulas are unchanged. Current context still
affects risk/uncertainty only; this repair does not claim improved predictive
accuracy or a proven betting edge.

## Exact event and venue binding

Each upcoming game reads its free ESPN college-football event summary. A
source name or numeric event ID alone is insufficient. Verification requires:

- exact event and competition ID, season, week, scheduled pregame state;
- both ESPN team IDs and matching team names on the expected sides;
- exact timezone-aware kickoff with confirmed time, neutral-site agreement;
- a nonempty actual event venue and agreement with the schedule venue ID;
- capture before kickoff, no future capture, maximum six-hour capture age;
- direct endpoint without redirects, raw response SHA256 plus a separately
  named normalized-payload hash that is rechecked before interpretation.

Weather coordinates must come from the event venue or a unique exact venue-ID
join to team venue metadata. Home-team and city guesses do not qualify. A
renamed stadium with conflicting provider IDs remains unknown until its mapping
is verified. Forecast response coordinates must be within 15 miles of the
requested stadium, accommodating forecast-grid resolution. Units, finite
values, ranges, UTC offset and the unique hour covering kickoff are checked.
An empty or wrong-day forecast does not count as weather coverage.

An exact venue-ID match to explicitly outdoor metadata makes operating roof
position not applicable. An indoor flag alone does not establish a retractable
roof's game-day position or justify suppressing weather risk. Fixed enclosures
without a dedicated verified classification remain flagged conservatively.

## Personnel evidence

Only fresh dated reports for the actual two teams earn reported-injury credit;
both teams need reports for a game's injury-coverage flag. A team with no
reports remains unknown. The export timestamp cannot refresh an old item.
Roster listings, reported injuries, explicit player designations, confirmed
starting QBs and complete game availability are separate states. Unknown
roster members are never counted as verified active. A season-roster `active`
flag does not earn final game-availability coverage.

Official-source research searches each verified matchup against a vetted list
of free conference publishers. Search snippets and links are discovery leads,
not facts. At most 48 distinct official pages are read in a run; exhausted
budgets and retrieval failures appear in the per-game report. Article parsing
requires a unique structured article with an explicit body, correct matchup
and week, a timezone-aware publication/modification clock, pregame capture,
and a unique current team-roster athlete ID. Conditional starter wording,
ambiguous names, conflicting declarations and an explicit OUT report cannot
confirm a starter. No starter is inferred from passing leaders or depth charts.

Current automatic official parsing covers the vetted conference HTML articles
listed in `harbin/official_personnel.py`. It does **not** yet parse conference
PDF reports, arbitrary school websites, or every publisher's custom format.
Those are recorded coverage limits, not evidence that no announcement exists.
For example, the [ACC availability policy](https://admin.theacc.com/sports/2025/8/28/availability-reporting.aspx)
uses conference-specific reports and game-day updates. NFL inactive-list
assumptions are not transplanted into college football.

## Publication and validation

`docs/public_input_quality.html` displays each game's verified event, venue,
fresh reports, weather, roof applicability and remaining gaps.
`public_input_quality.json` retains source URLs, capture clocks, hashes,
rejections, and a SHA256 binding to the exact `docs/latest.json` forecast file.
Publication reconciliation rejects a mismatched forecast or report copy.
Original forecast receipts and historical results are not retrospectively
rewritten by new observations.

The independent live source audit on October 20 at 00:52 UTC matched all 56
upcoming Week 8 FBS games. It verified venue coordinates and kickoff weather
for 52, with four venue-ID/coordinate gaps. No official starter announcements
passed the automated parser. This establishes collector coverage at that
capture, not complete personnel coverage or model accuracy.

Run `python -m scripts.cfb_public_input_audit --output reports/current_input_audit`
for a new read-only source check. It writes no forecasts or historical receipts.
The PR regression workflow runs the full Linux suite and a complete current
model dry run, audits the entire FBS slate and reconciles publication files
without pushing generated data. Main's model workflow repeats tests before
publishing. Real approved stake remains subject to all existing release gates;
no workflow places wagers.
