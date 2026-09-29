import json

import pandas as pd

from harbin.advanced import _looks_numeric_id, canon_team
from harbin.policy import signal_from_policy, _derive_blocked_weeks
from harbin.evidence_v7 import validate_policy_against_backtest


def _write_policy(tmp_path, *, enabled=False, candidate=False, blocked=None, test_season=2025):
    p={
        "version":4,
        "deployment_mode":"paper",
        "markets":{
            "spread":{"enabled":enabled,"candidate":candidate,"evidence_tier":"FINAL_TEST_VALIDATED" if enabled else ("TUNING_VALIDATED_AWAITING_FINAL_TEST" if candidate else "UNVALIDATED"),"lean":{"min_ev":.02,"min_edge":2,"min_prob":.52},"bet":{"min_ev":.04,"min_edge":3,"min_prob":.54},"strong":{"min_ev":.07,"min_edge":5,"min_prob":.57}},
            "moneyline":{"enabled":False,"candidate":False,"evidence_tier":"UNVALIDATED","lean":{"min_ev":.02,"min_edge":1.5,"min_prob":.52},"bet":{"min_ev":.04,"min_edge":2.5,"min_prob":.54},"strong":{"min_ev":.07,"min_edge":4,"min_prob":.56}},
            "total":{"enabled":False,"candidate":False,"evidence_tier":"UNVALIDATED","lean":{"min_ev":.02,"min_edge":2.5,"min_prob":.52},"bet":{"min_ev":.04,"min_edge":4,"min_prob":.54},"strong":{"min_ev":.07,"min_edge":6,"min_prob":.57}},
        },
        "regime_filters":{"blocked_weeks":blocked or [],"reason":{}},
        "portfolio":{"kelly_fraction":.2},
        "candidate_markets":["spread"] if candidate else [],
        "validated_markets":["spread"] if enabled else [],
        "untouched_test_seasons":[test_season],
    }
    f=tmp_path/"policy.json"; f.write_text(json.dumps(p)); return f


def test_unvalidated_market_never_signals(tmp_path):
    f=_write_policy(tmp_path,enabled=False,candidate=True)
    assert signal_from_policy(.20,8,.70,"spread",week=5,path=f)=="PASS"


def test_blocked_week_never_signals(tmp_path):
    f=_write_policy(tmp_path,enabled=True,candidate=True,blocked=[1])
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


def _winning_final_test_rows(n=120):
    rows=[]
    # 12 independent week clusters, each profitable, so the week-block bootstrap
    # lower bound is positive rather than relying on an IID illusion.
    for i in range(n):
        rows.append({"season":2025,"week":2+(i%12),"market":"spread","ev":.10,"edge":6,"probability":.62,"profit":1.0,"clv":.02})
    return rows


def test_candidate_promotes_only_after_positive_untouched_test(tmp_path):
    policy=_write_policy(tmp_path,candidate=True)
    bets=pd.DataFrame([
        {"season":2024,"week":5,"market":"spread","ev":.10,"edge":6,"probability":.62,"profit":-1.0,"clv":-.02},
        *_winning_final_test_rows(),
        {"season":2025,"week":5,"market":"moneyline","ev":.50,"edge":20,"probability":.80,"profit":-1.0,"clv":-.1},
    ])
    bp=tmp_path/"bets.csv"; bets.to_csv(bp,index=False)
    out=tmp_path/"validation.json"
    r=validate_policy_against_backtest(bp,policy,out)
    saved=json.loads(policy.read_text())
    assert r["markets"]["spread"]["enabled"] is True
    assert r["markets"]["spread"]["bets"]==120
    assert r["markets"]["moneyline"]["enabled"] is False
    assert r["status"]=="FORWARD_PAPER_READY"
    assert saved["markets"]["spread"]["enabled"] is True
    assert saved["markets"]["spread"]["evidence_tier"]=="FINAL_TEST_VALIDATED"
    assert saved["deployment_mode"]=="paper"


def test_candidate_fails_closed_on_negative_final_test(tmp_path):
    policy=_write_policy(tmp_path,candidate=True)
    rows=[]
    for i in range(120):
        rows.append({"season":2025,"week":2+(i%12),"market":"spread","ev":.10,"edge":6,"probability":.62,"profit":-1.0,"clv":.02})
    bp=tmp_path/"bets.csv"; pd.DataFrame(rows).to_csv(bp,index=False)
    out=tmp_path/"validation.json"
    r=validate_policy_against_backtest(bp,policy,out)
    saved=json.loads(policy.read_text())
    assert r["markets"]["spread"]["enabled"] is False
    assert r["status"]=="RESEARCH_ONLY"
    assert saved["markets"]["spread"]["enabled"] is False
    assert saved["markets"]["spread"]["evidence_tier"]=="FINAL_TEST_FAILED"


def test_empty_backtest_file_fails_closed_without_crashing(tmp_path):
    policy=_write_policy(tmp_path,candidate=True)
    bp=tmp_path/"bets.csv"; bp.write_text("")
    out=tmp_path/"validation.json"
    r=validate_policy_against_backtest(bp,policy,out)
    saved=json.loads(policy.read_text())
    assert r["status"]=="UNPROVEN"
    assert r["deployment_mode"]=="paper"
    assert r["markets"]=={}
    assert saved["markets"]["spread"]["enabled"] is False
    assert json.loads(out.read_text())["status"]=="UNPROVEN"
