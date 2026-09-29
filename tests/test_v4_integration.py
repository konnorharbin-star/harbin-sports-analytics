import numpy as np
import pandas as pd

from harbin.backtest import _grade_spread, _grade_total, _group_summary
from harbin.health import build_health_report
from harbin.market_intel import MarketIntelligence
from harbin.line_history import attach_line_movement

class G:
    game_id="1"; provider="DraftKings"; home_ml=-150; away_ml=130; home_spread=-3.0; market_total=52.5

def test_market_consensus_summary():
    q=[{"provider":"DraftKings","home_ml":-150,"away_ml":130,"home_spread":-3.0,"market_total":52.5},{"provider":"FanDuel","home_ml":-145,"away_ml":125,"home_spread":-2.5,"market_total":53.0}]
    s=MarketIntelligence._summary(G(),q); assert s["market_book_count"]>=2; assert s["consensus_home_spread"]==-2.75; assert s["best_home_ml"]==-145; assert 0<s["consensus_home_novig_probability"]<1

def test_backtest_grading_math():
    assert _grade_spread(7,"Home","Home",-3.5)==1; assert _grade_spread(2,"Away","Home",3.5)==1; assert _grade_total(60,"O",55.5)==1; assert _grade_total(50,"U",55.5)==1

def test_group_summary_drawdown_and_roi():
    df=pd.DataFrame({"result":[1,-1,1],"profit":[.91,-1,.91],"clv":[1.,2.,-.5]}); s=_group_summary(df); assert s["bets"]==3; assert s["units"]>0; assert s["max_drawdown"]>=1; assert s["clv_samples"]==3

def test_health_report_is_readiness_not_profit_claim(tmp_path):
    meta={"market_coverage":{"games":10,"moneyline":10,"spread":10,"total":10},"advanced_features":{"live_coverage":1.0},"metrics":{"margin_baseline_mae":13,"margin_mae":12.5,"margin_walkforward_folds":[{"mae":12,"baseline_mae":13}],"win_brier":.22,"win_ece":.05},"market_intelligence":{"multi_book_coverage":.5},"current_context":{"sources":["injuries"],"source_available":True}}; h=build_health_report(meta,reports_dir=tmp_path); assert 0<=h["system_health_score"]<=100; assert "not a profitability" in h["meaning"]

def test_line_history_no_file_is_safe(tmp_path):
    p=pd.DataFrame([{"game_id":"1","home_ml":-150,"away_ml":130,"market_spread_home":-3,"market_total":52}]); out,m=attach_line_movement(p,tmp_path); assert m["coverage"]==0; assert "opening_home_spread" in out.columns
