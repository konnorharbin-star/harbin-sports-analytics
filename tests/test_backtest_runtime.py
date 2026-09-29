from harbin.backtest_runtime import canonical_game_id


def test_canonical_game_id_normalizes_nullable_csv_ids():
    assert canonical_game_id(401520123.0) == "401520123"
    assert canonical_game_id("401520123.0") == "401520123"
    assert canonical_game_id("401520123") == "401520123"
    assert canonical_game_id(None) == ""
