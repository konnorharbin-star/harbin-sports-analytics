# Harbin CFB Run Report

**Platform / core:** 7.4.0 / 7.1.0  
**Season / Week:** 2026 / 5  
**Publication status:** WARN  
**Release state:** PAPER  
**Reconciliation:** PASS  

## Model validation
- Margin MAE: **12.656** vs baseline **13.221**; release weight **0.70**.
- Total MAE: **12.669** vs baseline **12.694**; release weight **0.15**.
- Brier / log loss / ECE: **0.1698 / 0.5113 / 0.0498**.
- Selection uses untouched evaluation outcomes: **False**.

## Monitoring and execution
- Live readiness: **95.2/100**; distribution stability **93.1/100**.
- Portfolio mode: **PAPER**; proposed **7.77u**; approved **0.00u**.
- Historical evidence: **0 bets**, ROI **—**, CLV **—**.
- Independent live evidence: **0 bets**, ROI **—**, CLV **—**.

## Current blockers
- historical_entry_integrity: historical promotion sample uses explicit verified opening-entry fields only
- historical_market_edge: ROBUST evidence: >=1000 verified opening-entry bets, ROI 95% CI lower bound >0, positive CLV, >=2 positive markets and >=2 positive seasons with minimum segment samples
- live_shadow_evidence: >=300 portfolio-verified graded live/shadow bets, non-negative ROI and positive pre-kickoff CLV proxy

## Operational alerts
- None.

## Interpretation
A green software run, a high readiness score, or good model error metrics do not establish a profitable betting edge. Historical and independent forward evidence remain separate release requirements.
