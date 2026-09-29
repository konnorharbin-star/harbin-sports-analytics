import numpy as np
import pandas as pd

from harbin.data import Game, SportsDataVerseClient, parse_espn_odds
from harbin.models import choose_blend_weight


def test_espn_odds_details_to_home_spread():
    raw = {
        "provider": {"name": "ESPN BET"},
        "details": "ALA -6.5",
        "overUnder": 58.5,
        "homeTeamOdds": {"moneyLine": -240, "favorite": True},
        "awayTeamOdds": {"moneyLine": 195, "favorite": False},
        "spread": 6.5,
    }
    x = parse_espn_odds(raw, {"ala", "alabama"}, {"uga", "georgia"})
    assert x["home_spread"] == -6.5
    assert x["market_total"] == 58.5
    assert x["home_ml"] == -240
    assert x["away_ml"] == 195


def test_espn_away_favorite_flips_home_spread():
    raw = {
        "details": "UGA -3",
        "homeTeamOdds": {"moneyLine": 130},
        "awayTeamOdds": {"moneyLine": -155},
    }
    x = parse_espn_odds(raw, {"ala"}, {"uga"})
    assert x["home_spread"] == 3.0


def test_model_blend_guard_can_reject_harmful_layer():
    target = np.array([1.0, -1.0, 0.5, -0.5])
    baseline = target.copy()
    residual = np.array([10.0, 10.0, -10.0, -10.0])
    w, mae = choose_blend_weight(target, baseline, residual)
    assert w == 0.0
    assert mae == 0.0


def test_long_archive_parser_attaches_all_markets_without_network():
    g = Game(
        game_id="123", season=2025, week=5, date="2025-10-01",
        away_id="2", away_team="Georgia", home_id="1", home_team="Alabama",
        away_score=None, home_score=None, completed=False, neutral_site=False,
    )
    rows = [
        {"game_id":"123","market_type":"money_line","abbr":"Alabama","lines":np.nan,"odds":-240,"book":"DraftKings"},
        {"game_id":"123","market_type":"money_line","abbr":"Georgia","lines":np.nan,"odds":195,"book":"DraftKings"},
        {"game_id":"123","market_type":"spread","abbr":"Alabama","lines":-6.5,"odds":-110,"book":"DraftKings"},
        {"game_id":"123","market_type":"spread","abbr":"Georgia","lines":6.5,"odds":-110,"book":"DraftKings"},
        {"game_id":"123","market_type":"total","abbr":"over","lines":58.5,"odds":-110,"book":"DraftKings"},
        {"game_id":"123","market_type":"total","abbr":"under","lines":58.5,"odds":-110,"book":"DraftKings"},
    ]
    c = SportsDataVerseClient()
    c._archive_odds = pd.DataFrame(rows)
    out = c._attach_archive_odds([g])[0]
    assert out.home_ml == -240
    assert out.away_ml == 195
    assert out.home_spread == -6.5
    assert out.market_total == 58.5
