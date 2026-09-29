from harbin.market import replica_ml_label, replica_spread_label, replica_total_label, replica_win_probability, no_vig


def test_replica_thresholds():
    assert replica_ml_label(.61,-142)[0] == 'LEAN'
    assert replica_spread_label(17,-10)[0] == 'STRONG'
    assert replica_spread_label(18,-12.5)[0] == 'BET'
    assert replica_total_label(63,46.5)[0] == 'STRONG'
    assert replica_total_label(39,45.5)[0] == 'BET'


def test_probability_transform():
    p=replica_win_probability(17)
    assert .84 < p < .86


def test_no_vig_sums_to_one():
    a,h=no_vig(130,-155)
    assert abs(a+h-1)<1e-12
