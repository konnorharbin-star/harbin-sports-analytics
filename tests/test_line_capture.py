from datetime import datetime, timezone
import pandas as pd
from harbin.data import Game
from capture_lines import build_snapshot_rows


def test_build_snapshot_rows_preserves_primary_and_consensus():
    g=Game(game_id="1",season=2026,week=5,date="2026-10-01T00:00:00Z",away_id="a",away_team="Away",home_id="h",home_team="Home",away_score=None,home_score=None,completed=False,neutral_site=False,provider="Book",away_ml=130,home_ml=-150,home_spread=-3.0,market_total=51.5)
    intel=pd.DataFrame([{"game_id":"1","consensus_home_spread":-2.5,"consensus_total":52.0,"best_home_ml":-145,"best_away_ml":135,"market_book_count":3,"market_books":"A | B | C"}])
    rows=build_snapshot_rows([g],intel,datetime(2026,9,29,tzinfo=timezone.utc))
    assert rows[0]["home_spread"]==-3.0
    assert rows[0]["consensus_home_spread"]==-2.5
    assert rows[0]["market_book_count"]==3
