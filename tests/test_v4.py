import numpy as np
import pandas as pd

from harbin.advanced import canon_team
from harbin.calibration import brier_score, expected_calibration_error
from harbin.pro_market import kelly_fraction, quant_signal, risk_multiplier
from harbin.models import choose_blend_weight


def test_team_canonicalization_aliases():
    assert canon_team("North Carolina State") == "ncstate"
    assert canon_team("UCF") == "ucf"


def test_calibration_metrics_are_sane():
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
    assert brier_score(y,p) < .05
    assert expected_calibration_error(y,p,2) < .2


def test_quant_gate_and_risk():
    assert quant_signal(.08,6,.60,"spread") == "STRONG"
    assert quant_signal(-.01,9,.70,"spread") == "PASS"
    assert 0 < risk_multiplier(.8,.7,20) < risk_multiplier(0,1,0) <= 1


def test_kelly_nonnegative():
    assert kelly_fraction(.60,-110) > 0
    assert kelly_fraction(.40,-110) == 0


def test_blend_guard_still_rejects_bad_layer():
    y=np.array([1.,-1.,0.]); b=y.copy(); r=np.array([9.,9.,9.]); w,_=choose_blend_weight(y,b,r); assert w==0
