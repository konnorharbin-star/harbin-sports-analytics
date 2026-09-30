# Harbin CFB Model Card

## Identity
- Core model version: **7.1.0**
- Repository revision: `03987c699d03f8b9d8746d1936fec8067bf660c6`
- Season / Week: **2026 / 5**
- Generated: **Sep 29, 2026 · 7:59 PM CT**

## Intended use
Independent CFB score, margin, total and win-probability estimation; sportsbook comparison; paper/production betting research with explicit uncertainty and risk controls. The Cooper-style table is a presentation layer, not the proprietary formula of any third party.

## Data
- Historical games: **4079**
- Schedule/results: sportsdataverse/cfbfastR-data
- Live odds: ESPN live (site.web.api.espn.com) + ESPN Core supplement
- Advanced feature coverage: **100.0%**
- Multi-book coverage: **98.2%**
- Current context sources: SportsDataverse ESPN injuries, SportsDataverse cfb_team_info, Open-Meteo forecast, Open-Meteo geocoding / team venue metadata

## Validation
- Method: nested whole-week chronology: core fit -> tune weight -> OOF calibration -> untouched evaluation; final refit keeps the tune-selected weight and uses OOF probability calibration
- Margin MAE: **12.657079769201356** vs baseline **13.22104338022683**
- Total MAE: **12.681388088386885** vs baseline **12.693581407531097**
- Win-probability Brier: **0.171944284202318**
- Calibration ECE: **0.0636411720732171**

## Operational controls
- Live monitoring score: **94.7/100**
- Portfolio mode: **PAPER**
- Approved units: **0.0**
- Proposed/paper units: **5.51**

## Leakage controls
Features are generated pregame from prior games only; model training is chronological; tuning/calibration/evaluation are separated; historical market evidence is walk-forward. Sportsbook prices are excluded from the score-generation model and are used only after fair scores/probabilities are produced.

## Known limitations
Injury designations and sportsbook feeds can be incomplete or change rapidly. Weather forecasts are uncertain. Historical betting edge may not persist. Public data cannot reproduce an unpublished third-party scoring formula. Do not infer profitability from MAE alone.
