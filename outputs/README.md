# Latest CFB model output

**Model:** v7.1.0  
**Season / Week:** 2026 / 6  
**Updated:** Oct 4, 2026 · 9:32 AM CT  
**Market:** PARTIAL MARKET DATA — 26/58 games have at least one verified market from ESPN live (site.web.api.espn.com) + ESPN Core supplement.  
**Dynamic advanced-feature live coverage:** 100%  
**System health:** 76.4/100 *(readiness, not predicted profitability)*

## Use these
- [Interactive Cooper-style table](cfb_model_2026_week6.html)
- [Quant card](quant_card.html)
- [Quant recommendations CSV](quant_recommendations.csv)
- [Full model CSV](cfb_model_2026_week6.csv)
- [Metadata / diagnostics](cfb_model_2026_week6_metadata.json)
- [System health report](system_health.json)

## Fresh PNGs for mobile
- [Fresh page 1 — cache-safe](cfb_model_2026_week6_run_20261004_093218_CT_page1.png)
- [Fresh page 2 — cache-safe](cfb_model_2026_week6_run_20261004_093218_CT_page2.png)
- [Fresh page 3 — cache-safe](cfb_model_2026_week6_run_20261004_093218_CT_page3.png)
- [Fresh page 4 — cache-safe](cfb_model_2026_week6_run_20261004_093218_CT_page4.png)
- [Fresh page 5 — cache-safe](cfb_model_2026_week6_run_20261004_093218_CT_page5.png)

These filenames change on every run so GitHub mobile cannot reuse an old image preview.

## Stable PNG names
- [Stable page 1](cfb_model_2026_week6_page1.png)
- [Stable page 2](cfb_model_2026_week6_page2.png)
- [Stable page 3](cfb_model_2026_week6_page3.png)
- [Stable page 4](cfb_model_2026_week6_page4.png)
- [Stable page 5](cfb_model_2026_week6_page5.png)

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
