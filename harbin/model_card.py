from __future__ import annotations

import os
from pathlib import Path


def write_model_card(meta: dict, monitoring: dict, portfolio: dict, output="outputs/MODEL_CARD.md"):
    m = meta.get("metrics") or {}
    adv = meta.get("advanced_features") or {}
    ctx = meta.get("current_context") or {}
    intel = meta.get("market_intelligence") or {}
    bankroll = portfolio.get("bankroll_risk") or {}
    sha = os.environ.get("GITHUB_SHA", "local")
    text = f"""# Harbin CFB Model Card

## Identity
- Core model version: **{meta.get('model_version','unknown')}**
- Repository revision: `{sha}`
- Season / Week: **{meta.get('season')} / {meta.get('week')}**
- Generated: **{meta.get('updated_at_ct',meta.get('generated_at'))}**

## Intended use
Independent CFB score, margin, total and win-probability estimation; sportsbook comparison; paper/production betting research with explicit uncertainty and risk controls. The Cooper-style table is a presentation layer, not the proprietary formula of any third party.

## Data
- Historical games: **{meta.get('historical_games')}**
- Schedule/results: {meta.get('schedule_source','SportsDataverse')}
- Live odds: {meta.get('odds_source','unknown')}
- Advanced feature coverage: **{float(adv.get('live_coverage',adv.get('coverage',0)) or 0):.1%}**
- Multi-book coverage: **{float(intel.get('multi_book_coverage',0) or 0):.1%}**
- Current context sources: {', '.join(ctx.get('sources') or []) or 'none'}

## Validation
- Method: {meta.get('validation')}
- Margin MAE: **{m.get('margin_mae')}** vs baseline **{m.get('margin_baseline_mae')}**
- Total MAE: **{m.get('total_mae')}** vs baseline **{m.get('total_baseline_mae')}**
- Win-probability Brier: **{m.get('win_brier')}**
- Calibration ECE: **{m.get('win_ece')}**

## Operational controls
- Live monitoring score: **{monitoring.get('live_readiness_score')}/100**
- Portfolio mode: **{portfolio.get('mode','paper').upper()}**
- Approved units: **{portfolio.get('approved_units',0)}**
- Proposed units before bankroll/concentration controls: **{portfolio.get('proposed_units',0)}**
- Paper/shadow allocated units after controls: **{portfolio.get('paper_allocated_units',0)}**
- Unit-risk multiplier: **{bankroll.get('risk_multiplier',1.0)}**
- Current live/shadow drawdown: **{bankroll.get('current_drawdown_units',0)} units**
- Portfolio hard stop active: **{bool(bankroll.get('hard_stop',False))}**
- Execution-blocked candidates: **{portfolio.get('execution_blocked_bets',0)}**

## Leakage controls
Features are generated pregame from prior games only; model training is chronological; tuning/calibration/evaluation are separated; historical market evidence is walk-forward. Sportsbook prices are excluded from the score-generation model and are used only after fair scores/probabilities are produced.

## Bankroll and execution interpretation
Risk is expressed in abstract betting units; the system does not invent a dollar bankroll. Production approval requires the hard release gate, a production policy, an independent live/shadow grading ledger, no active drawdown hard stop, and an executable sportsbook/price for each approved candidate. The software produces an approval plan; it does not place wagers.

## Known limitations
Injury designations and sportsbook feeds can be incomplete or change rapidly. Weather forecasts are uncertain. Historical betting edge may not persist. Public data cannot reproduce an unpublished third-party scoring formula. Do not infer profitability from MAE alone.
"""
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
