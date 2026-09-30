import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from capture_lines import build_snapshot_rows
from harbin.data import Game
from harbin.grading import _clv_from_market_snapshot, _execution_clv_from_market_snapshot, _latest_pre_kickoff_market, _profit
from harbin.line_history import attach_line_movement
from harbin.market_intel import MarketIntelligence, parse_action_network_game
import harbin.pro_market as pro_market


def _game():
    return Game(game_id="1",season=2026,week=5,date="2026-10-01T00:00:00Z",away_id="a",away_team="Away",home_id="h",home_team="Home",away_score=None,home_score=None,completed=False,neutral_site=False,provider="Primary",away_ml=130,home_ml=-150,home_spread=-3.0,market_total=52.0)


def test_action_network_parser_never_merges_different_books():
    raw={
        "home_team_id":1,"away_team_id":2,
        "teams":[{"id":1,"display_name":"Home"},{"id":2,"display_name":"Away"}],
        "markets":{"event":{"event":{
            "moneyline":[
                {"book_id":10,"side":"home","odds":-150},{"book_id":10,"side":"away","odds":130},
                {"book_id":20,"side":"home","odds":-140},{"book_id":20,"side":"away","odds":120},
            ],
            "spread":[
                {"book_id":10,"side":"home","value":-3.5,"odds":-110},{"book_id":10,"side":"away","value":3.5,"odds":-110},
                {"book_id":20,"side":"home","value":-2.5,"odds":-105},{"book_id":20,"side":"away","value":2.5,"odds":-115},
            ],
            "total":[
                {"book_id":10,"side":"over","value":52.5,"odds":-110},{"book_id":10,"side":"under","value":52.5,"odds":-110},
                {"book_id":20,"side":"over","value":51.5,"odds":-105},{"book_id":20,"side":"under","value":51.5,"odds":-115},
            ],
        }}},
    }
    out=parse_action_network_game(raw)
    assert len(out["quotes"])==2
    by={q["provider"]:q for q in out["quotes"]}
    assert by["ActionNetwork book 10"]["home_ml"]==-150
    assert by["ActionNetwork book 10"]["home_spread"]==-3.5
    assert by["ActionNetwork book 20"]["home_ml"]==-140
    assert by["ActionNetwork book 20"]["home_spread"]==-2.5


def test_multi_book_summary_line_shops_with_provenance_and_price_tiebreaks():
    g=_game()
    quotes=[
        {"provider":"BookA","source":"test","home_ml":-150,"away_ml":130,"home_spread":-3.0,"home_spread_price":-110,"away_spread_price":-110,"market_total":52.5,"over_price":-110,"under_price":-105},
        {"provider":"BookB","source":"test","home_ml":-145,"away_ml":125,"home_spread":-2.5,"home_spread_price":-115,"away_spread_price":-105,"market_total":51.5,"over_price":-110,"under_price":-115},
        {"provider":"BookC","source":"test","home_ml":-148,"away_ml":135,"home_spread":-2.5,"home_spread_price":-105,"away_spread_price":-110,"market_total":51.5,"over_price":-105,"under_price":-110},
    ]
    x=MarketIntelligence._summary(g,quotes)
    assert x["market_book_count"]==4  # 3 external books + independent Primary quote
    assert x["best_home_ml"]==-145 and x["best_home_ml_book"]=="BookB"
    assert x["best_away_ml"]==135 and x["best_away_ml_book"]=="BookC"
    assert x["best_home_spread"]==-2.5 and x["best_home_spread_odds"]==-105 and x["best_home_spread_book"]=="BookC"
    assert x["best_away_spread"]==3.0 and x["best_away_spread_book"]=="BookA"
    assert x["best_over_total"]==51.5 and x["best_over_odds"]==-105 and x["best_over_book"]=="BookC"
    assert x["best_under_total"]==52.5 and x["best_under_book"]=="BookA"
    audit=json.loads(x["market_quotes_json"])
    assert {q["provider"] for q in audit}=={"BookA","BookB","BookC","Primary"}


def test_market_selection_carries_execution_book(monkeypatch):
    monkeypatch.setattr(pro_market,"quant_signal",lambda *a,**k:"BET")
    monkeypatch.setattr(pro_market,"fractional_kelly_units",lambda *a,**k:.25)
    row={
        "week":5,"home_team":"Home","away_team":"Away","provider":"Primary",
        "calibrated_home_probability":.65,"quant_best_ml_roi":.10,"quant_best_ml_side":"Home","quant_best_ml_edge_pp":5.0,"best_home_ml":-130,"best_home_ml_book":"BookX",
        "cover_probability":np.nan,"spread_edge_pts":np.nan,"total_probability":np.nan,"total_edge_pts":np.nan,
    }
    pick=pro_market.select_best_market(row)
    assert pick["quant_market"]=="moneyline"
    assert pick["quant_book"]=="BookX"
    assert pick["quant_odds"]==-130


def test_capture_snapshot_persists_consensus_and_best_book_audit_fields():
    g=_game()
    intel=pd.DataFrame([{
        "game_id":"1","consensus_home_spread":-2.5,"consensus_total":52.5,"consensus_home_novig_probability":.61,
        "best_home_ml":-140,"best_away_ml":135,"best_home_ml_book":"BookM","best_away_ml_book":"BookN",
        "best_home_spread":-2.0,"best_away_spread":3.0,"best_home_spread_odds":-105,"best_away_spread_odds":-110,"best_home_spread_book":"BookS1","best_away_spread_book":"BookS2",
        "best_over_total":51.5,"best_under_total":53.0,"best_over_odds":-105,"best_under_odds":-102,"best_over_book":"BookT1","best_under_book":"BookT2",
        "market_book_count":4,"market_books":"A | B | C | D","market_quote_sources":"test","market_quotes_json":"[]",
    }])
    rows=build_snapshot_rows([g],intel,datetime(2026,9,29,tzinfo=timezone.utc))
    r=rows[0]
    assert r["consensus_home_novig_probability"]==.61
    assert r["best_home_ml_book"]=="BookM"
    assert r["best_home_spread_book"]=="BookS1"
    assert r["best_under_book"]=="BookT2"
    assert r["market_book_count"]==4


def test_clv_prefers_consensus_close_and_uses_actual_execution_odds():
    entry=pd.Series({"quant_market":"spread","quant_side":"Home","home_team":"Home","quant_price":-3.0,"quant_odds":-105})
    close=pd.Series({"home_spread":-3.5,"consensus_home_spread":-4.5,"market_book_count":4})
    clv,source=_clv_from_market_snapshot(entry,close)
    assert clv==1.5
    assert source=="consensus_latest_pre_kickoff_snapshot"
    assert abs(_profit(1,-105,"spread")-(100/105))<1e-12

    ml_entry=pd.Series({"quant_market":"moneyline","quant_side":"Home","home_team":"Home","quant_price":140,"quant_odds":140,"consensus_home_novig_probability":.40})
    ml_close=pd.Series({"consensus_home_novig_probability":.45,"home_ml":-115,"away_ml":105})
    market_clv,source=_clv_from_market_snapshot(ml_entry,ml_close)
    execution_clv=_execution_clv_from_market_snapshot(ml_entry,ml_close)
    assert abs(market_clv-.05)<1e-12
    assert abs(execution_clv-(.45-(100/240)))<1e-12
    assert source=="consensus_latest_pre_kickoff_snapshot"


def test_latest_close_is_strictly_pre_kickoff_and_line_history_tracks_consensus(tmp_path):
    snapshots=pd.DataFrame([
        {"captured_at":"2026-09-30T22:00:00Z","game_id":"1","home_spread":-3.0,"market_total":52.0,"consensus_home_spread":-2.5,"consensus_total":52.5,"consensus_home_novig_probability":.58,"market_book_count":3,"market_books":"A | B | C"},
        {"captured_at":"2026-09-30T23:50:00Z","game_id":"1","home_spread":-4.0,"market_total":53.0,"consensus_home_spread":-3.5,"consensus_total":53.0,"consensus_home_novig_probability":.62,"market_book_count":4,"market_books":"A | B | C | D"},
        {"captured_at":"2026-10-01T00:01:00Z","game_id":"1","home_spread":-6.0,"market_total":55.0,"consensus_home_spread":-5.5,"consensus_total":55.0,"consensus_home_novig_probability":.70,"market_book_count":5,"market_books":"future"},
    ])
    close=_latest_pre_kickoff_market(snapshots,"1",pd.Timestamp("2026-10-01T00:00:00Z"))
    assert close["consensus_home_spread"]==-3.5

    hist=tmp_path/"history"; hist.mkdir(); snapshots.assign(kickoff="2026-10-01T00:00:00Z",home_ml=-150,away_ml=130).to_csv(hist/"market_snapshots.csv",index=False)
    pred=pd.DataFrame([{"game_id":"1","date":"2026-10-01T00:00:00Z","market_spread_home":-4.0,"market_total":53.0,"home_ml":-160,"away_ml":140}])
    out,meta=attach_line_movement(pred,hist); r=out.iloc[0]
    assert r.opening_consensus_home_spread==-2.5
    assert r.latest_pre_kickoff_consensus_home_spread==-3.5
    assert abs(r.consensus_home_novig_move-.04)<1e-12
    assert r.latest_pre_kickoff_market_book_count==4
    assert meta["consensus_close_games"]==1
