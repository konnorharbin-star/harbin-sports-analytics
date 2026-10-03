# Harbin CFB Run Report

**Platform / core:** 7.4.0 / 7.1.0  
**Season / Week:** 2026 / 5  
**Publication status:** WARN  
**Release state:** RESEARCH  
**Reconciliation:** PASS  

## Model validation
- Margin MAE: **12.688** vs baseline **13.221**; release weight **0.65**.
- Total MAE: **12.653** vs baseline **12.694**; release weight **0.25**.
- Brier / log loss / ECE: **0.1710 / 0.5160 / 0.0533**.
- Selection uses untouched evaluation outcomes: **False**.

## Monitoring and execution
- Live readiness: **94.7/100**; distribution stability **92.8/100**.
- Portfolio mode: **PAPER**; proposed **9.17u**; approved **0.00u**.
- Historical evidence: **0 bets**, ROI **—**, CLV **—**.
- Independent live evidence: **0 bets**, ROI **—**, CLV **—**.

## Current blockers
- probability_ece: ECE <= 0.05 on chronological release holdout
- historical_entry_integrity: historical promotion sample uses explicit verified opening-entry fields only
- historical_market_edge: ROBUST evidence: >=1000 verified opening-entry bets, ROI 95% CI lower bound >0, positive CLV, >=2 positive markets and >=2 positive seasons with minimum segment samples
- live_shadow_evidence: >=300 portfolio-verified graded live/shadow bets, non-negative ROI and positive pre-kickoff CLV proxy

## Operational alerts
- None.

## Interpretation
A green software run, a high readiness score, or good model error metrics do not establish a profitable betting edge. Historical and independent forward evidence remain separate release requirements.
