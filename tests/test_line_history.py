import pandas as pd
from harbin.line_history import attach_line_movement


def test_hourly_market_snapshots_take_priority(tmp_path):
    hist=tmp_path/"history"; hist.mkdir()
    pd.DataFrame([
        {"captured_at":"2026-09-29T12:00:00Z","game_id":"1","kickoff":"2026-10-01T00:00:00Z","home_spread":-3.0,"market_total":51.5,"home_ml":-150,"away_ml":130},
        {"captured_at":"2026-09-30T23:00:00Z","game_id":"1","kickoff":"2026-10-01T00:00:00Z","home_spread":-4.0,"market_total":52.5,"home_ml":-180,"away_ml":155},
    ]).to_csv(hist/"market_snapshots.csv",index=False)
    pred=pd.DataFrame([{"game_id":"1","date":"2026-10-01T00:00:00Z","market_spread_home":-4.5,"market_total":53.0,"home_ml":-190,"away_ml":165}])
    out,meta=attach_line_movement(pred,hist)
    r=out.iloc[0]
    assert r.opening_home_spread==-3.0
    assert r.latest_pre_kickoff_home_spread==-4.0
    assert r.market_snapshot_count==2
    assert r.line_history_source=="hourly_market_snapshots"
    assert meta["hourly_snapshot_games"]==1
