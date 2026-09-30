# Harbin CFB Run Report

**Platform / core:** 7.3.0 / 7.1.0  
**Season / Week:** 2026 / 5  
**Publication status:** WARN  
**Release state:** RESEARCH  
**Reconciliation:** PASS  

## Model validation
- Margin MAE: **12.657** vs baseline **13.221**; release weight **0.65**.
- Total MAE: **12.681** vs baseline **12.694**; release weight **0.05**.
- Brier / log loss / ECE: **0.1719 / 0.5159 / 0.0636**.
- Selection uses untouched evaluation outcomes: **False**.

## Monitoring and execution
- Live readiness: **93.6/100**; distribution stability **93.6/100**.
- Portfolio mode: **PAPER**; proposed **4.57u**; approved **0.00u**.
- Historical evidence: **3436 bets**, ROI **-0.47%**, CLV **0.95%**.
- Independent live evidence: **0 bets**, ROI **—**, CLV **—**.

## Current blockers
- probability_ece: ECE <= 0.05 on chronological release holdout
- historical_market_edge: >=1000 bets, ROI 95% CI lower bound >0, positive CLV, >=2 positive markets and >=2 positive seasons
- live_shadow_evidence: >=300 graded live bets, non-negative ROI and positive pre-kickoff CLV proxy

## Operational alerts
- None.

## Interpretation
A green software run, a high readiness score, or good model error metrics do not establish a profitable betting edge. Historical and independent forward evidence remain separate release requirements.
