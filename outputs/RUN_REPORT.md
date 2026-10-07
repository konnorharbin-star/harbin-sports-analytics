# Harbin CFB Run Report

**Platform / core:** 7.4.0 / 7.1.0  
**Season / Week:** 2026 / 6  
**Publication status:** WARN  
**Release state:** RESEARCH  
**Reconciliation:** PASS  

## Model validation
- Margin MAE: **12.959** vs baseline **13.502**; release weight **0.75**.
- Total MAE: **12.592** vs baseline **12.649**; release weight **0.30**.
- Brier / log loss / ECE: **0.1748 / 0.5223 / 0.0622**.
- Selection uses untouched evaluation outcomes: **False**.

## Monitoring and execution
- Live readiness: **94.7/100**; distribution stability **96.5/100**.
- Portfolio mode: **PAPER**; proposed **4.90u**; approved **0.00u**.
- Historical evidence: **0 bets**, ROI **—**, CLV **—**.
- Independent live evidence: **0 bets**, ROI **—**, CLV **—**.

## Market × tier forward validation
- Display-tier ledger: **144 posted flat-1u decisions** · clean validation-eligible **0** · excluded legacy/unverified **144** · status **EARLY_SAMPLE** · validated cells **0**.

## Current blockers
- complete_market_coverage: >= 95% games with ML + spread + total
- probability_ece: ECE <= 0.05 on chronological release holdout
- historical_entry_integrity: historical promotion sample uses explicit verified opening-entry fields only
- historical_market_edge: ROBUST evidence: >=1000 verified opening-entry bets, ROI 95% CI lower bound >0, positive CLV, >=2 positive markets and >=2 positive seasons with minimum segment samples
- live_shadow_evidence: >=300 portfolio-verified graded live/shadow bets, non-negative ROI and positive pre-kickoff CLV proxy

## Operational alerts
- None.

## Interpretation
A green software run, a high readiness score, or good model error metrics do not establish a profitable betting edge. Historical and independent forward evidence remain separate release requirements.
