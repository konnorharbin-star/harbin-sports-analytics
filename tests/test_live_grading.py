import pandas as pd
from harbin.grading import _grade_row,_clv_from_market_snapshot,_summary,_display_market_entry,_grade_display_tiers


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


def test_display_market_entry_normalizes_each_published_market():
    base = pd.Series(
        {
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.60,
            "snapshot_at": "2026-10-06T18:00:00Z",
            "ml_badge": "STRONG",
            "ml_team": "Home",
            "ml_odds": -110,
            "ml_edge_pp": 7.6,
            "ml_est_roi": 0.14,
            "spread_badge": "BET",
            "spread_team": "Away",
            "spread_line": 3.5,
            "spread_odds": -105,
            "spread_edge_pts": 4.0,
            "cover_probability": 0.58,
            "total_badge": "LEAN",
            "total_dir": "U",
            "market_total": 52.5,
            "total_odds": -108,
            "total_edge_pts": 3.0,
            "total_probability": 0.55,
        }
    )

    moneyline = _display_market_entry(base, "moneyline")
    spread = _display_market_entry(base, "spread")
    total = _display_market_entry(base, "total")

    assert moneyline["tier"] == "STRONG"
    assert moneyline["quant_side"] == "Home"
    assert moneyline["model_probability"] == 0.60
    assert spread["tier"] == "BET"
    assert spread["quant_side"] == "Away"
    assert spread["quant_price"] == 3.5
    assert total["tier"] == "LEAN"
    assert total["quant_side"] == "U"
    assert total["quant_price"] == 52.5


def test_display_market_entry_requires_real_book_and_quote_timestamp():
    missing = pd.Series(
        {
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.60,
            "snapshot_at": "2026-10-06T18:00:00Z",
            "ml_badge": "STRONG",
            "ml_team": "Home",
            "ml_odds": 120,
            "ml_edge_pp": 14.5,
            "ml_est_roi": 0.32,
            "ml_book": float("nan"),
            "ml_quote_at": float("nan"),
        }
    )
    verified = missing.copy()
    verified["ml_book"] = "Book A"
    verified["ml_quote_at"] = "2026-10-06T17:59:00Z"

    assert _display_market_entry(missing, "moneyline")["price_verified"] is False
    assert _display_market_entry(verified, "moneyline")["price_verified"] is True


def test_display_market_entry_rejects_unparseable_quote_timestamp():
    row = pd.Series(
        {
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.60,
            "ml_badge": "STRONG",
            "ml_team": "Home",
            "ml_odds": 120,
            "ml_edge_pp": 14.5,
            "ml_est_roi": 0.32,
            "ml_book": "Book A",
            "ml_quote_at": "not-a-timestamp",
            "ml_quote_time_source": "captured_at",
        }
    )

    entry = _display_market_entry(row, "moneyline")

    assert entry["price_verified"] is False
    assert entry["quote_time_source"] == "captured_at"


def test_clean_tier_grading_uses_first_eligible_pre_kickoff_snapshot():
    kickoff = "2026-10-10T18:00:00Z"
    hist = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2026,
                "week": 6,
                "date": kickoff,
                "snapshot_at": "2026-10-10T15:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "calibrated_home_probability": 0.60,
                "model_margin_home": 4.0,
                "model_total": 50.0,
                "ml_badge": "STRONG",
                "ml_team": "Home",
                "ml_odds": 120,
                "ml_edge_pp": 14.5,
                "ml_est_roi": 0.32,
                "ml_book": float("nan"),
                "ml_quote_at": float("nan"),
            },
            {
                "game_id": "g1",
                "season": 2026,
                "week": 6,
                "date": kickoff,
                "snapshot_at": "2026-10-10T16:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "calibrated_home_probability": 0.60,
                "model_margin_home": 6.0,
                "model_total": 52.0,
                "ml_badge": "STRONG",
                "ml_team": "Home",
                "ml_odds": 120,
                "ml_edge_pp": 14.5,
                "ml_est_roi": 0.32,
                "ml_book": "Book A",
                "ml_quote_at": "2026-10-10T15:59:00Z",
                "ml_quote_time_source": "captured_at",
            },
            {
                "game_id": "g1",
                "season": 2026,
                "week": 6,
                "date": kickoff,
                "snapshot_at": "2026-10-10T17:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "calibrated_home_probability": 0.60,
                "model_margin_home": 8.0,
                "model_total": 54.0,
                "ml_badge": "STRONG",
                "ml_team": "Home",
                "ml_odds": 125,
                "ml_edge_pp": 15.6,
                "ml_est_roi": 0.35,
                "ml_book": "Book B",
                "ml_quote_at": "2026-10-10T16:59:00Z",
                "ml_quote_time_source": "captured_at",
            },
        ]
    )
    hist["_ts"] = pd.to_datetime(hist["snapshot_at"], utc=True)
    finals = {"g1": (27.0, 20.0)}

    posted = _grade_display_tiers(hist, finals, pd.DataFrame(), clean_only=False)
    clean = _grade_display_tiers(hist, finals, pd.DataFrame(), clean_only=True)

    posted_ml = posted[posted["market"] == "moneyline"].iloc[0]
    clean_ml = clean[clean["market"] == "moneyline"].iloc[0]

    assert posted_ml["entry_snapshot"] == "2026-10-10T15:00:00Z"
    assert bool(posted_ml["validation_eligible"]) is False
    assert clean_ml["entry_snapshot"] == "2026-10-10T16:00:00Z"
    assert bool(clean_ml["validation_eligible"]) is True
    assert clean_ml["book"] == "Book A"
    assert clean_ml["projected_margin_home"] == 6.0
    assert clean_ml["projected_total"] == 52.0


def test_display_market_entry_rejects_quote_after_prediction_snapshot():
    row = pd.Series(
        {
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.60,
            "snapshot_at": "2026-10-06T18:00:00Z",
            "ml_badge": "STRONG",
            "ml_team": "Home",
            "ml_odds": 120,
            "ml_edge_pp": 14.5,
            "ml_est_roi": 0.32,
            "ml_book": "Book A",
            "ml_quote_at": "2026-10-06T18:01:00Z",
            "ml_quote_time_source": "source_last_update",
        }
    )

    entry = _display_market_entry(row, "moneyline")

    assert entry["price_verified"] is False
