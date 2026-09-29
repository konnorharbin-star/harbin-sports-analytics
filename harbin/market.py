from __future__ import annotations

import math

REPLICA_MARGIN_SIGMA = 16.41


def norm_cdf(x):
    return .5*(1+math.erf(x/math.sqrt(2)))


def american_implied(odds):
    o = float(odds)
    return abs(o)/(abs(o)+100) if o < 0 else 100/(o+100)


def no_vig(away_odds, home_odds):
    pa, ph = american_implied(away_odds), american_implied(home_odds)
    s = pa+ph
    return pa/s, ph/s


def fair_american(p):
    p = min(.999, max(.001, float(p)))
    return int(round(-100*p/(1-p))) if p >= .5 else int(round(100*(1-p)/p))


def roi(p, odds):
    o = float(odds)
    profit = 100/abs(o) if o < 0 else o/100
    return p*profit-(1-p)


def replica_win_probability(margin):
    return norm_cdf(abs(float(margin))/REPLICA_MARGIN_SIGMA)


def label(value, lean, bet, strong):
    x = abs(float(value))
    return "STRONG" if x >= strong else "BET" if x >= bet else "LEAN" if x >= lean else ""


def replica_ml_label(model_p, odds):
    edge_pp = 100*(float(model_p)-american_implied(odds))
    return label(edge_pp, 2, 3, 6), edge_pp


def replica_spread_label(model_margin_home, home_spread):
    # Sportsbook home spread is negative when home is favorite. Fair market threshold is -home_spread.
    edge_home = float(model_margin_home)+float(home_spread)
    return label(edge_home, 2, 4, 6), edge_home


def replica_total_label(model_total, market_total):
    edge = float(model_total)-float(market_total)
    return label(edge, 2.5, 4.5, 7.5), edge
