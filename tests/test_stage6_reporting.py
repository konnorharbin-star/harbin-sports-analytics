import json

import pandas as pd

from harbin.monitoring import build_live_monitoring
from harbin.reporting import build_audit_snapshot, validate_publication_files, write_reporting_bundle


def _base_state():
    pred = pd.DataFrame([
        {"season": 2026, "week": 5, "model_margin_home": 3.0, "model_total": 55.0, "calibrated_home_probability": .58},
        {"season": 2026, "week": 5, "model_margin_home": -2.0, "model_total": 49.0, "calibrated_home_probability": .44},
    ])
    meta = {
        "platform_version": "7.3.0",
        "model_version": "7.1.0",
        "season": 2026,
        "week": 5,
        "upcoming_games": 2,
        "generated_at": "2026-09-30T02:00:00+00:00",
        "market_coverage": {"games": 2, "moneyline": 2, "spread": 2, "total": 2},
        "advanced_features": {"dynamic_coverage": 1.0, "dynamic_feature_count": 20},
        "current_context": {"coverage": 1.0, "weather_coverage": 1.0},
        "market_intelligence": {"multi_book_coverage": 1.0},
        "metrics": {"win_brier": .17, "win_ece": .04, "selection_uses_evaluation": False},
    }
    monitor = {"status": "OK", "live_readiness_score": 95.0, "scores": {"distribution_stability": 90.0}, "drift_details": {}, "alerts": []}
    gate = {"release_state": "PAPER", "production_eligible": False, "checks": [], "blockers": ["historical evidence incomplete"]}
    dq = {"status": "PASS", "errors": 0, "warnings": 0}
    portfolio = {"mode": "paper", "policy_mode": "paper", "proposed_units": 1.0, "paper_allocated_units": 1.0, "approved_units": 0.0}
    meta["live_monitoring"] = monitor
    meta["release_gate"] = gate
    meta["data_quality"] = dq
    meta["portfolio"] = portfolio
    return pred, meta, monitor, gate, dq, portfolio


def test_audit_snapshot_reconciles_fail_closed_paper_state():
    pred, meta, monitor, gate, dq, portfolio = _base_state()
    snap = build_audit_snapshot(pred, meta, monitor, gate, dq, portfolio, policy={"deployment_mode": "paper"})
    assert snap["reconciliation"]["status"] == "PASS"
    assert snap["release"]["state"] == "PAPER"
    assert snap["portfolio"]["approved_units"] == 0


def test_audit_snapshot_rejects_impossible_real_stake():
    pred, meta, monitor, gate, dq, portfolio = _base_state()
    portfolio = dict(portfolio, approved_units=.5, mode="production")
    snap = build_audit_snapshot(pred, meta, monitor, gate, dq, portfolio, policy={"deployment_mode": "production"})
    failed = {x["name"] for x in snap["reconciliation"]["errors"]}
    assert "approved_stake_fail_closed" in failed
    assert snap["status"] == "FAIL"


def test_reporting_bundle_writes_one_canonical_public_snapshot(tmp_path):
    pred, meta, monitor, gate, dq, portfolio = _base_state()
    out, docs, reports, history = [tmp_path / x for x in ("outputs", "docs", "reports", "history")]
    for p in (out, docs, reports, history):
        p.mkdir()
    for name, obj in (("live_monitoring.json", monitor), ("release_gate.json", gate), ("data_quality.json", dq), ("portfolio_summary.json", portfolio)):
        (out / name).write_text(json.dumps(obj))
        (docs / name).write_text(json.dumps(obj))
    (docs / "metadata.json").write_text(json.dumps(meta))
    (reports / "production_policy.json").write_text(json.dumps({"deployment_mode": "paper"}))

    snap, validation = write_reporting_bundle(pred, meta, out, docs, reports, history)
    assert validation["status"] == "PASS"
    assert json.loads((out / "audit_snapshot.json").read_text()) == json.loads((docs / "audit_snapshot.json").read_text())
    assert "Harbin CFB Run Report" in (out / "RUN_REPORT.md").read_text()
    assert "audit_snapshot.json" in (docs / "audit.html").read_text()
    assert len((history / "audit_snapshots_v1.jsonl").read_text().splitlines()) == 1
    assert snap["identity"]["season"] == 2026


def test_publication_validator_detects_cross_file_identity_mismatch(tmp_path):
    pred, meta, monitor, gate, dq, portfolio = _base_state()
    out, docs = tmp_path / "outputs", tmp_path / "docs"
    out.mkdir(); docs.mkdir()
    snap = build_audit_snapshot(pred, meta, monitor, gate, dq, portfolio, policy={"deployment_mode": "paper"})
    (out / "audit_snapshot.json").write_text(json.dumps(snap))
    (docs / "audit_snapshot.json").write_text(json.dumps(snap))
    bad_meta = dict(meta, week=6)
    (docs / "metadata.json").write_text(json.dumps(bad_meta))
    (docs / "release_gate.json").write_text(json.dumps(gate))
    (docs / "portfolio_summary.json").write_text(json.dumps(portfolio))
    (docs / "live_monitoring.json").write_text(json.dumps(monitor))
    (docs / "data_quality.json").write_text(json.dumps(dq))
    report = validate_publication_files(out, docs)
    assert report["status"] == "FAIL"
    assert "metadata_week" in {x["name"] for x in report["errors"]}


def test_monitoring_emits_drift_diagnostics_and_alert(tmp_path):
    hist = pd.DataFrame({
        "pred_margin_home": [0.0] * 120,
        "pred_total": [50.0] * 120,
        "home_win_probability": [.50] * 120,
    })
    hist.to_csv(tmp_path / "backtest_predictions.csv", index=False)
    pred = pd.DataFrame({
        "model_margin_home": [20, 21, 22, 23, 24, 25],
        "model_total": [75, 76, 77, 78, 79, 80],
        "calibrated_home_probability": [.80, .81, .82, .83, .84, .85],
    })
    meta = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "market_coverage": {"games": 6, "moneyline": 6, "spread": 6, "total": 6},
        "advanced_features": {"dynamic_coverage": 1, "dynamic_feature_count": 20},
        "market_intelligence": {"multi_book_coverage": 1},
        "current_context": {"coverage": 1, "weather_coverage": 1},
        "metrics": {"win_brier": .17, "win_ece": .03},
    }
    report = build_live_monitoring(pred, meta, reports_dir=tmp_path)
    assert set(report["drift_details"]) == {"margin", "total", "probability"}
    assert report["scores"]["distribution_stability"] < 55
    assert any("shifted materially" in x for x in report["alerts"])
