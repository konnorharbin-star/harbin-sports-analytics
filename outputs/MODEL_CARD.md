# Harbin CFB Model Card

## Identity
- Core model version: **7.1.0**
- Repository revision: `902c403bbc3078bb937773e329403369c430afca`
- Season / Week: **2026 / 5**
- Generated: **Oct 2, 2026 · 12:43 PM CT**

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
- Margin MAE: **12.656183046421095** vs baseline **13.22104338022683**
- Total MAE: **12.668565921799578** vs baseline **12.693581407531097**
- Win-probability Brier: **0.1698076479580908**
- Calibration ECE: **0.04979509724742311**

## Operational controls
- Live monitoring score: **95.2/100**
- Portfolio mode: **PAPER**
- Approved units: **0.0**
- Proposed units before bankroll/concentration controls: **7.68**
- Paper/shadow allocated units after controls: **4.51**
- Unit-risk multiplier: **1.0**
- Current live/shadow drawdown: **0.0 units**
- Portfolio hard stop active: **False**
- Execution-blocked candidates: **43**

## Leakage controls
Features are generated pregame from prior games only; model training is chronological; tuning/calibration/evaluation are separated; historical market evidence is walk-forward. Sportsbook prices are excluded from the score-generation model and are used only after fair scores/probabilities are produced.

## Bankroll and execution interpretation
Risk is expressed in abstract betting units; the system does not invent a dollar bankroll. Production approval requires the hard release gate, a production policy, an independent live/shadow grading ledger, no active drawdown hard stop, and an executable sportsbook/price for each approved candidate. The software produces an approval plan; it does not place wagers.

## Known limitations
Injury designations and sportsbook feeds can be incomplete or change rapidly. Weather forecasts are uncertain. Historical betting edge may not persist. Public data cannot reproduce an unpublished third-party scoring formula. Do not infer profitability from MAE alone.
