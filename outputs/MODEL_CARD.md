# Harbin CFB Model Card

## Identity
- Core model version: **7.1.0**
- Repository revision: `d04d95927ca9521484694091d087b57e5dd09f90`
- Season / Week: **2026 / 5**
- Generated: **Sep 29, 2026 · 9:16 PM CT**

## Intended use
Independent CFB score, margin, total and win-probability estimation; sportsbook comparison; paper/production betting research with explicit uncertainty and risk controls. The Cooper-style table is a presentation layer, not the proprietary formula of any third party.

## Data
- Historical games: **4079**
- Schedule/results: sportsdataverse/cfbfastR-data
- Live odds: ESPN live (site.web.api.espn.com) + ESPN Core supplement
- Advanced feature coverage: **100.0%**
- Multi-book coverage: **98.2%**
- Current context sources: SportsDataverse ESPN injuries, SportsDataverse ESPN rosters, SportsDataverse ESPN teams, Open-Meteo forecast / indoor venue suppression, Venue geocoding / team metadata

## Validation
- Method: nested whole-week chronology: core fit -> tune weight -> OOF calibration -> untouched evaluation; final refit keeps the tune-selected weight and uses OOF probability calibration
- Margin MAE: **12.657079769201353** vs baseline **13.22104338022683**
- Total MAE: **12.681388088386885** vs baseline **12.693581407531097**
- Win-probability Brier: **0.17194428420231783**
- Calibration ECE: **0.06364117207321737**

## Operational controls
- Live monitoring score: **93.6/100**
- Portfolio mode: **PAPER**
- Approved units: **0.0**
- Proposed/paper units: **4.56**

## Leakage controls
Features are generated pregame from prior games only; model training is chronological; tuning/calibration/evaluation are separated; historical market evidence is walk-forward. Sportsbook prices are excluded from the score-generation model and are used only after fair scores/probabilities are produced.

## Known limitations
Injury designations and sportsbook feeds can be incomplete or change rapidly. Weather forecasts are uncertain. Historical betting edge may not persist. Public data cannot reproduce an unpublished third-party scoring formula. Do not infer profitability from MAE alone.
