import pandas as pd

from harbin.forward_proof import build_forward_evidence


def _rows(n=360, profit=1.0, clv=.02):
    rows=[]
    for i in range(n):
        week=1+(i%15)
        # Keep probabilities calibrated around realized home outcomes.
        home_win=(i%5)!=0
        rows.append({
            "season":2026,"week":week,"quant_signal":"BET","quant_market":"spread",
            "profit":profit,"clv_proxy":clv,"calibrated_home_probability":.80 if home_win else .20,
            "actual_margin_home":7 if home_win else -7,
        })
    return rows


def test_missing_or_empty_forward_ledger_is_safe(tmp_path):
    out=tmp_path/"forward.json"
    r=build_forward_evidence(tmp_path/"missing.csv",out)
    assert r["status"]=="INSUFFICIENT_SAMPLE"
    assert r["real_money_allowed"] is False
    (tmp_path/"empty.csv").write_text("")
    r=build_forward_evidence(tmp_path/"empty.csv",out)
    assert r["status"]=="INSUFFICIENT_SAMPLE"


def test_positive_forward_sample_only_reaches_human_review(tmp_path):
    p=tmp_path/"graded.csv";pd.DataFrame(_rows()).to_csv(p,index=False)
    r=build_forward_evidence(p,tmp_path/"out.json",min_bets=300,min_week_blocks=12)
    assert r["status"]=="ELIGIBLE_FOR_HUMAN_RELEASE_REVIEW"
    assert r["real_money_allowed"] is False
    assert all(r["checks"].values())


def test_negative_clv_blocks_forward_release(tmp_path):
    p=tmp_path/"graded.csv";pd.DataFrame(_rows(clv=-.02)).to_csv(p,index=False)
    r=build_forward_evidence(p,tmp_path/"out.json",min_bets=300,min_week_blocks=12)
    assert r["status"]!="ELIGIBLE_FOR_HUMAN_RELEASE_REVIEW"
    assert r["checks"]["positive_clv"] is False


def test_negative_roi_blocks_forward_release(tmp_path):
    p=tmp_path/"graded.csv";pd.DataFrame(_rows(profit=-1.0)).to_csv(p,index=False)
    r=build_forward_evidence(p,tmp_path/"out.json",min_bets=300,min_week_blocks=12)
    assert r["status"]!="ELIGIBLE_FOR_HUMAN_RELEASE_REVIEW"
    assert r["checks"]["positive_roi"] is False
