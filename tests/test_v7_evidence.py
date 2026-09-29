import json
from pathlib import Path

import pandas as pd

from harbin.advanced import _looks_numeric_id, canon_team
from harbin.policy import signal_from_policy, _derive_blocked_weeks, load_policy
from harbin.evidence_v7 import validate_policy_against_backtest


def _write_policy(tmp_path, *, enabled=True, blocked=None):
    p={
        "version":2,
        "deployment_mode":"paper",
        "markets":{
            "spread":{"enabled":enabled,"evidence_tier":"VALIDATED" if enabled else "UNVALIDATED","lean":{"min_ev":.02,"min_edge":2,"min_prob":.52},"bet":{"min_ev":.04,"min_edge":3,"min_prob":.54},"strong":{"min_ev":.07,"min_edge":5,"min_prob":.57}},
            "moneyline":{"enabled":False,"evidence_tier":"UNVALIDATED","lean":{"min_ev":.02,"min_edge":1.5,"min_prob":.52},"bet":{"min_ev":.04,"min_edge":2.5,"min_prob":.54},"strong":{"min_ev":.07,"min_edge":4,"min_prob":.56}},
            "total":{"enabled":False,"evidence_tier":"UNVALIDATED","lean":{"min_ev":.02,"min_edge":2.5,"min_prob":.52},"bet":{"min_ev":.04,"min_edge":4,"min_prob":.54},"strong":{"min_ev":.07,"min_edge":6,"min_prob":.57}},
        },
        "regime_filters":{"blocked_weeks":blocked or [],"reason":{}},
        "portfolio":{"kelly_fraction":.2},
    }
    f=tmp_path/"policy.json"; f.write_text(json.dumps(p)); return f


def test_unvalidated_market_never_signals(tmp_path):
    f=_write_policy(tmp_path,enabled=False)
    assert signal_from_policy(.20,8,.70,"spread",week=5,path=f)=="PASS"


def test_blocked_week_never_signals(tmp_path):
    f=_write_policy(tmp_path,enabled=True,blocked=[1])
    assert signal_from_policy(.20,8,.70,"spread",week=1,path=f)=="PASS"
    assert signal_from_policy(.20,8,.70,"spread",week=5,path=f)=="STRONG"


def test_bad_regime_is_identified():
    summary={"by_week":{"1":{"bets":200,"roi":-.10,"roi_ci_95":[-.16,-.03]},"5":{"bets":220,"roi":.04,"roi_ci_95":[-.02,.10]}}}
    blocked,reason=_derive_blocked_weeks(summary)
    assert blocked==[1]
    assert "1" in reason


def test_dynamic_identity_type_detection():
    assert _looks_numeric_id(pd.Series([1,2,333,55]))
    assert not _looks_numeric_id(pd.Series(["Alabama","Georgia","Ohio State"]))
    assert canon_team("North Carolina State")=="ncstate"


def test_policy_validation_uses_only_enabled_market(tmp_path):
    policy=_write_policy(tmp_path,enabled=True)
    bets=pd.DataFrame([
        {"season":2024,"week":5,"market":"spread","ev":.10,"edge":6,"probability":.62,"profit":1.0,"clv":.02},
        {"season":2025,"week":5,"market":"spread","ev":.10,"edge":6,"probability":.62,"profit":1.0,"clv":.02},
        {"season":2025,"week":5,"market":"moneyline","ev":.50,"edge":20,"probability":.80,"profit":-1.0,"clv":-.1},
    ])
    bp=tmp_path/"bets.csv"; bets.to_csv(bp,index=False)
    out=tmp_path/"validation.json"
    r=validate_policy_against_backtest(bp,policy,out)
    assert r["markets"]["spread"]["bets"]==1
    assert r["markets"]["moneyline"]["bets"]==0
    assert r["overall"]["roi"]==1.0
    assert out.exists()
