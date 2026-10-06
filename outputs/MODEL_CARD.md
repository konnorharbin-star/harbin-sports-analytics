# Harbin CFB Model Card

## Identity
- Core model version: **7.1.0**
- Repository revision: `af133f5d3bd8c62f58b65b61aeebbf42eb159add`
- Season / Week: **2026 / 6**
- Generated: **Oct 6, 2026 · 12:15 PM CT**

## Intended use
Independent CFB score, margin, total and win-probability estimation; sportsbook comparison; paper/production betting research with explicit uncertainty and risk controls. The Cooper-style table is a presentation layer, not the proprietary formula of any third party.

## Data
- Historical games: **4135**
- Schedule/results: sportsdataverse/cfbfastR-data
- Live odds: ESPN live (site.web.api.espn.com) + ESPN Core supplement
- Advanced feature coverage: **100.0%**
- Multi-book coverage: **98.3%**
- Current context sources: SportsDataverse ESPN injuries, SportsDataverse ESPN rosters, SportsDataverse ESPN teams, Open-Meteo forecast / indoor venue suppression, Venue geocoding / team metadata

## Validation
- Method: nested whole-week chronology: core fit -> tune weight -> OOF calibration -> untouched evaluation; final refit keeps the tune-selected weight and uses OOF probability calibration
- Margin MAE: **12.95850296244939** vs baseline **13.50210922129303**
- Total MAE: **12.59160433487721** vs baseline **12.648615389536973**
- Win-probability Brier: **0.17476492013360306**
- Calibration ECE: **0.06218232013495895**

## Operational controls
- Live monitoring score: **94.0/100**
- Portfolio mode: **PAPER**
- Approved units: **0.0**
- Proposed units before bankroll/concentration controls: **6.08**
- Paper/shadow allocated units after controls: **3.82**
- Unit-risk multiplier: **1.0**
- Current live/shadow drawdown: **0.0 units**
- Portfolio hard stop active: **False**
- Execution-blocked candidates: **40**

## Leakage controls
Features are generated pregame from prior games only; model training is chronological; tuning/calibration/evaluation are separated; historical market evidence is walk-forward. Sportsbook prices are excluded from the score-generation model and are used only after fair scores/probabilities are produced.

## Bankroll and execution interpretation
Risk is expressed in abstract betting units; the system does not invent a dollar bankroll. Production approval requires the hard release gate, a production policy, an independent live/shadow grading ledger, no active drawdown hard stop, and an executable sportsbook/price for each approved candidate. The software produces an approval plan; it does not place wagers.

## Known limitations
Injury designations and sportsbook feeds can be incomplete or change rapidly. Weather forecasts are uncertain. Historical betting edge may not persist. Public data cannot reproduce an unpublished third-party scoring formula. Do not infer profitability from MAE alone.
