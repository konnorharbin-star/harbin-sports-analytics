# Latest CFB model output

**Model:** v7.1.0  
**Season / Week:** 2026 / 6  
**Updated:** Oct 8, 2026 · 5:58 PM CT  
**Market:** LIVE MARKET DATA — all 55 games have verified market data from ESPN live (site.web.api.espn.com) + ESPN Core supplement.  
**Dynamic advanced-feature live coverage:** 100%  
**System health:** 84.1/100 *(readiness, not predicted profitability)*

## Use these
- [Interactive Cooper-style table](cfb_model_2026_week6.html)
- [Quant card](quant_card.html)
- [Supported subgroup edge board](edge_card.html)
- [Priority edges with research-only BET_NOW / WAIT diagnostics](edge_priority.csv)
- [Observed timing/line survival report](edge_timing_report.json)
- [Forward timing validation (research-only)](edge_timing_forward_performance.json)
- [Forward timing baseline comparison (research-only)](edge_timing_baseline_performance.json)
- [Timing baseline paired audit rows](edge_timing_baseline_graded.csv)
- [Forward timing decision/quote ledger](edge_timing_forward_graded.csv)
- [Actionable price/line-cushion edges](edge_actionable.csv)
- [Core price-confirmed edges](edge_core.csv)
- [All holdout-supported subgroup edges](edge_supported.csv)
- [Parent-only persistent edges](edge_parent_only.csv)
- [Quant recommendations CSV](quant_recommendations.csv)
- [All persistent historical edge candidates](edge_candidates.csv)
- [Edge-regime watchlist](edge_watchlist.csv)
- [Historically contraindicated edge subgroups](edge_exclusions.csv)
- [Edge-regime evidence](edge_regimes.json)
- [Forward edge performance](edge_forward_performance.json)
- [Forward graded edge ledger](edge_forward_graded.csv)
- [Full model CSV](cfb_model_2026_week6.csv)
- [Metadata / diagnostics](cfb_model_2026_week6_metadata.json)
- [System health report](system_health.json)
- [Market × tier forward validation](tier_performance.json)
- [Posted market × tier matrix CSV](tier_performance.csv)
- [Clean validation-eligible tier matrix](tier_validation_clean.csv)
- [Posted flat-1u graded tier ledger](live_graded_tiers.csv)
- [Clean validation-entry ledger](live_graded_tiers_clean.csv)

## Fresh PNGs for mobile
- [Fresh page 1 — cache-safe](cfb_model_2026_week6_run_20261008_175825_CT_page1.png)
- [Fresh page 2 — cache-safe](cfb_model_2026_week6_run_20261008_175825_CT_page2.png)
- [Fresh page 3 — cache-safe](cfb_model_2026_week6_run_20261008_175825_CT_page3.png)
- [Fresh page 4 — cache-safe](cfb_model_2026_week6_run_20261008_175825_CT_page4.png)

These filenames change on every run so GitHub mobile cannot reuse an old image preview.

## Stable PNG names
- [Stable page 1](cfb_model_2026_week6_page1.png)
- [Stable page 2](cfb_model_2026_week6_page2.png)
- [Stable page 3](cfb_model_2026_week6_page3.png)
- [Stable page 4](cfb_model_2026_week6_page4.png)

The Cooper-style table is the reconstructed presentation layer. The Quant card is the independent EV/risk layer. Missing verified markets display **NO LINE**. Run the separate **CFB Backtest** workflow before treating signals as historically established.

## v7.4 final audit / evidence integrity layer
- [Portfolio card](portfolio_card.csv)
- [Portfolio summary](portfolio_summary.json)
- [Live monitoring](live_monitoring.json)
- [Canonical audit snapshot](audit_snapshot.json)
- [Run report](RUN_REPORT.md)
- [Publication validation](publication_validation.json)
- [Release gate](release_gate.json)
- [Data-quality contracts](data_quality.json)
- [Model card](MODEL_CARD.md)
- [System audit dashboard](../docs/audit.html)
- Forward evidence is now based on execution-ready cap-constrained portfolio decisions with strict pre-kickoff timestamps.
- Real approved stake remains **0** unless every hard PRODUCTION gate and audited execution check passes.
