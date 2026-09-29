import pandas as pd
from harbin.grading import _grade_row,_clv_from_market_snapshot,_summary


def test_grade_spread_total_moneyline():
    s=pd.Series({"quant_market":"spread","quant_side":"Home","home_team":"Home","away_team":"Away","quant_price":-3.0})
    assert _grade_row(s,7,50)==1
    t=pd.Series({"quant_market":"total","quant_side":"U","quant_price":52.5})
    assert _grade_row(t,0,49)==1
    m=pd.Series({"quant_market":"moneyline","quant_side":"Away","home_team":"Home","away_team":"Away","quant_price":140})
    assert _grade_row(m,-1,50)==1


def test_clv_uses_latest_market_snapshot_and_direction():
    e=pd.Series({"quant_market":"spread","quant_side":"Home","home_team":"Home","quant_price":-3.0})
    c=pd.Series({"home_spread":-4.5})
    v,src=_clv_from_market_snapshot(e,c)
    assert v==1.5 and src=="latest_verified_pre_kickoff_snapshot"
    e2=pd.Series({"quant_market":"total","quant_side":"O","quant_price":51.5})
    c2=pd.Series({"market_total":53.0})
    v2,_=_clv_from_market_snapshot(e2,c2)
    assert v2==1.5


def test_live_summary_has_uncertainty_and_drawdown():
    b=pd.DataFrame({"result":[1,-1,1,0]*10,"profit":[.9091,-1,.9091,0]*10,"clv_proxy":[.5,.25,-.1,.1]*10})
    x=_summary(b)
    assert x["graded_bets"]==40
    assert x["roi_ci_95"][0] is not None
    assert x["max_drawdown"]>=0
