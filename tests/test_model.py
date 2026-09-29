from harbin_model import ml_badge, spread_badge, total_badge, cdf, REPLICA_SIGMA

def test_moneyline_thresholds():
    assert ml_badge(.61,-142)[0] == 'LEAN'
    assert ml_badge(.71,-170)[0] == 'STRONG'

def test_spread_thresholds():
    assert spread_badge(17,-10)[0] == 'STRONG'
    assert spread_badge(18,-12.5)[0] == 'BET'

def test_total_thresholds():
    assert total_badge(63,46.5)[0] == 'STRONG'
    assert total_badge(39,45.5)[0] == 'BET'

def test_probability_transform():
    p=cdf(17/REPLICA_SIGMA)
    assert .84 < p < .86
