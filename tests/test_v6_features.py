import pandas as pd
from harbin.advanced import canon_id,AdvancedFeatureStore
from harbin.monitoring import build_live_monitoring

def test_canon_id():
    assert canon_id(123.0)=="123" and canon_id("123.0")=="123"
def test_monitoring_does_not_credit_source_names_as_context():
    pred=pd.DataFrame({"market_available":[True]*10,"model_margin_home":[1]*10,"model_total":[55]*10,"calibrated_home_probability":[.55]*10})
    meta={"market_coverage":{"games":10,"moneyline":10,"spread":10,"total":10},"advanced_features":{"dynamic_coverage":0,"dynamic_feature_count":0,"sources":["x"]},"market_intelligence":{"multi_book_coverage":0},"current_context":{"coverage":0,"weather_coverage":0,"sources":["x"]},"metrics":{"win_brier":.20,"win_ece":.05},"generated_at":"2099-01-01T00:00:00+00:00"}
    r=build_live_monitoring(pred,meta,reports_dir="/tmp/nope")
    assert r["scores"]["advanced_coverage"]==0 and r["scores"]["context"]==0
