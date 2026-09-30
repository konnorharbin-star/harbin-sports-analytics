# Harbin CFB Model Card

## Identity
- Core model version: **7.1.0**
- Repository revision: `26e0b86a757bb515cac7badaba241cbe18a922f7`
- Season / Week: **2026 / 5**
- Generated: **Sep 29, 2026 · 7:31 PM CT**

## Intended use
Independent CFB score, margin, total and win-probability estimation; sportsbook comparison; paper/production betting research with explicit uncertainty and risk controls. The Cooper-style table is a presentation layer, not the proprietary formula of any third party.

## Data
- Historical games: **4079**
- Schedule/results: sportsdataverse/cfbfastR-data
- Live odds: ESPN live (site.web.api.espn.com) + ESPN Core supplement
- Advanced feature coverage: **100.0%**
- Multi-book coverage: **0.0%**
- Current context sources: SportsDataverse ESPN injuries, SportsDataverse cfb_team_info, Open-Meteo forecast, Open-Meteo geocoding / team venue metadata

## Validation
- Method: nested chronological core/tune/calibration/release holdout + leak-free expanding-season walk-forward + fail-closed baseline fallback
- Margin MAE: **13.042876228378635** vs baseline **13.720203965537888**
- Total MAE: **12.778014944283827** vs baseline **12.778014944283827**
- Win-probability Brier: **0.15710434541934895**
- Calibration ECE: **0.038366813107869206**

## Operational controls
- Live monitoring score: **86.9/100**
- Portfolio mode: **PAPER**
- Approved units: **0.0**
- Proposed/paper units: **6.34**

## Leakage controls
Features are generated pregame from prior games only; model training is chronological; tuning/calibration/evaluation are separated; historical market evidence is walk-forward. Sportsbook prices are excluded from the score-generation model and are used only after fair scores/probabilities are produced.

## Known limitations
Injury designations and sportsbook feeds can be incomplete or change rapidly. Weather forecasts are uncertain. Historical betting edge may not persist. Public data cannot reproduce an unpublished third-party scoring formula. Do not infer profitability from MAE alone.
