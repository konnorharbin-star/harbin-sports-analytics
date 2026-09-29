# Harbin Sports Analytics — Agent Rules

## Non-negotiable modeling rules
- Never use future information in a historical feature. A game's features must be knowable before kickoff.
- Use chronological, season-based, or season/week walk-forward validation. Never use a random split for a time-series performance claim.
- Reserve an untouched final chronological test set after threshold/model selection whenever the data supports it.
- Sportsbook prices do not enter the core fair-score projection. Market prices may be used only after the fair margin/total is produced, or as an explicit risk/diagnostic layer.
- Current injuries, weather, depth-chart news, and other current-only context must not be backfilled into historical training unless a timestamp-correct historical source exists.
- Preserve unrounded model values internally. Round only in presentation.
- Missing data must remain missing or trigger a documented fallback. Never invent a sportsbook quote, player status, weather value, or feature.
- Keep Cooper Replica signals separate from Harbin Quant signals. Do not call replica thresholds validated unless historical evidence supports them.
- Do not claim Jason Cooper's private scoring formula is known.
- Do not claim profitability from MAE alone, a short record, a health score, or an in-sample fit.
- A backtest must state which line stage it uses and must not call a proxy an official closing line.

## Evidence and release gates
- Fail closed: an unvalidated market must return PASS, even if apparent EV/edge is large.
- Threshold research must use development data; threshold acceptance must use later tuning data; a final season/block must remain untouched for policy validation.
- Do not enable a market unless its tuning sample satisfies the configured minimum sample, positive ROI, positive week-cluster bootstrap 95% lower bound, and non-negative CLV when available.
- Historically destructive regimes may be blocked only from evidence generated without peeking at the live slate.
- `deployment_mode` remains `paper` unless a separate explicit release process proves forward-paper performance, positive CLV, stable calibration, independent price verification, and clean data quality.
- Never change real-money release gates simply to make a dashboard score look better.
- A software component can be production-grade while the betting edge remains UNPROVEN. Preserve that distinction in code, docs, and summaries.

## Model promotion rules
- Any learned residual layer must be eligible for weight 0 if it loses to the baseline on chronological tuning data.
- Keep an untouched chronological evaluation block after hyperparameter/weight selection.
- Report MAE/RMSE for margins/totals and Brier/log-loss/ECE for probabilities.
- Report market results by market, signal, edge bucket, season, and week with bets, win rate, ROI, units, max drawdown, confidence interval, and CLV when available.
- Prefer positive CLV over short-term win rate when judging whether a pricing edge is plausible.
- Stake sizing must be capped and risk-adjusted; no uncapped Kelly.
- Most games should be PASS when estimated EV is weak, evidence is weak, or data quality is poor.

## Data-quality rules
- Advanced features must report dynamic pregame coverage separately from static-prior coverage.
- Match upstream team identities by stable ID when available and canonical team name as a documented fallback.
- Multi-book claims require an independent second price source; a single sportsbook/provider cannot be labeled consensus.
- Current market, context, and injury sources must expose source failures and coverage percentages in metadata.

## Engineering rules
- Run `python -m pytest -q` before merging.
- Keep both the live-model and walk-forward-backtest workflows green.
- Do not commit secrets, virtualenvs, caches, or binary model artifacts.
- Data-source failures must appear in metadata/health output instead of being hidden.
- Research branches may upload artifacts but must not overwrite production outputs on `main`.
- Merge to `main` only after branch CI and evidence workflows complete successfully.
