# Harbin CFB Model Card

## Identity
- Core model version: **4.0.0**
- Repository revision: `064144f324416a8bc15b9103ca140ac201210889`
- Season / Week: **2026 / 5**
- Generated: **Sep 28, 2026 · 11:55 PM CT**

## Intended use
Independent CFB score, margin, total and win-probability estimation; sportsbook comparison; paper/production betting research with explicit uncertainty and risk controls. The Cooper-style table is a presentation layer, not the proprietary formula of any third party.

## Data
- Historical games: **4079**
- Schedule/results: sportsdataverse/cfbfastR-data
- Live odds: ESPN live (site.web.api.espn.com) + ESPN Core supplement
- Advanced feature coverage: **100.0%**
- Multi-book coverage: **0.0%**
- Current context sources: SportsDataverse ESPN injuries, SportsDataverse cfb_team_info

## Validation
- Method: nested chronological core/tune/calibration/evaluation + season walk-forward + zero-weight guard
- Margin MAE: **12.68680249236582** vs baseline **13.720203965537888**
- Total MAE: **12.711073278987822** vs baseline **12.778014944283827**
- Win-probability Brier: **0.15640752116978443**
- Calibration ECE: **0.0620029288208732**

## Operational controls
- Live monitoring score: **82.3/100**
- Portfolio mode: **PAPER**
- Approved units: **0.0**
- Proposed/paper units: **30.51**

## Leakage controls
Features are generated pregame from prior games only; model training is chronological; tuning/calibration/evaluation are separated; historical market evidence is walk-forward. Sportsbook prices are excluded from the score-generation model and are used only after fair scores/probabilities are produced.

## Known limitations
Injury designations and sportsbook feeds can be incomplete or change rapidly. Weather forecasts are uncertain. Historical betting edge may not persist. Public data cannot reproduce an unpublished third-party scoring formula. Do not infer profitability from MAE alone.
