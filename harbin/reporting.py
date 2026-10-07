from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

AUDIT_SCHEMA_VERSION = 1


def _read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _num(v: Any, default=None):
    return float(v) if _finite(v) else default


def _check(name: str, passed: bool, detail: str, severity: str = "ERROR") -> dict:
    return {"name": name, "passed": bool(passed), "severity": severity, "detail": detail}


def _json_equal(a: Any, b: Any) -> bool:
    try:
        return json.dumps(a, sort_keys=True, separators=(",", ":"), default=str) == json.dumps(
            b, sort_keys=True, separators=(",", ":"), default=str
        )
    except Exception:
        return a == b


def _compact_evidence(report: dict, live: bool = False) -> dict:
    if not report:
        return {}
    if live:
        return {
            "graded_bets": int(report.get("graded_bets", report.get("bets", 0)) or 0),
            "roi": _num(report.get("roi")),
            "units": _num(report.get("units"), 0.0),
            "win_rate": _num(report.get("win_rate")),
            "avg_clv": _num(report.get("avg_clv_proxy", report.get("avg_clv"))),
            "clv_samples": int(report.get("verified_close_clv_samples", report.get("clv_samples", 0)) or 0),
            "roi_ci_95": report.get("roi_ci_95") or [None, None],
            "status": report.get("status"),
            "clv_method": report.get("clv_method"),
        }
    overall = report.get("overall") or {}
    return {
        "bets": int(overall.get("bets", overall.get("graded_bets", 0)) or 0),
        "roi": _num(overall.get("roi")),
        "units": _num(overall.get("units"), 0.0),
        "win_rate": _num(overall.get("win_rate")),
        "avg_clv": _num(overall.get("avg_clv", overall.get("avg_clv_proxy"))),
        "roi_ci_95": overall.get("roi_ci_95") or [None, None],
        "max_drawdown": _num(overall.get("max_drawdown"), 0.0),
        "status": report.get("status"),
        "note": report.get("note"),
    }


def _load_trend(path: str | Path, limit: int = 12) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    try:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return []
    return rows[-max(1, int(limit)) :]


def build_audit_snapshot(
    pred: pd.DataFrame,
    meta: dict,
    monitor: dict,
    release_gate: dict,
    data_quality: dict,
    portfolio: dict,
    evidence: dict | None = None,
    live: dict | None = None,
    policy: dict | None = None,
    prior_trend: list[dict] | None = None,
) -> dict:
    """Build one canonical, reconciled view of a published model run.

    The dashboard consumes this single snapshot instead of independently joining several
    files in the browser. Reconciliation failures are explicit and production stake can
    never be reported as approved when the release/policy path is not production-safe.
    """
    evidence = evidence or {}
    live = live or {}
    policy = policy or {}
    prior_trend = prior_trend or []
    metrics = meta.get("metrics") or {}
    market = meta.get("market_coverage") or {}
    adv = meta.get("advanced_features") or {}
    ctx = meta.get("current_context") or {}
    intel = meta.get("market_intelligence") or {}

    checks = []
    expected_rows = int(meta.get("upcoming_games", len(pred)) or 0)
    checks.append(_check("row_count", len(pred) == expected_rows, f"predictions={len(pred)} metadata={expected_rows}"))

    for key in ("season", "week"):
        if key in pred.columns and len(pred):
            vals = pd.to_numeric(pred[key], errors="coerce").dropna().astype(int).unique().tolist()
            target = int(meta.get(key)) if meta.get(key) is not None else None
            checks.append(_check(f"identity_{key}", target is not None and vals == [target], f"values={vals} metadata={target}"))

    embedded_gate = meta.get("release_gate") or {}
    if embedded_gate:
        checks.append(
            _check(
                "release_gate_reconciled",
                embedded_gate.get("release_state") == release_gate.get("release_state"),
                f"metadata={embedded_gate.get('release_state')} output={release_gate.get('release_state')}",
            )
        )
    embedded_monitor = meta.get("live_monitoring") or {}
    if embedded_monitor and _finite(embedded_monitor.get("live_readiness_score")):
        checks.append(
            _check(
                "monitor_reconciled",
                abs(float(embedded_monitor["live_readiness_score"]) - float(monitor.get("live_readiness_score", -999))) < 1e-9,
                f"metadata={embedded_monitor.get('live_readiness_score')} output={monitor.get('live_readiness_score')}",
            )
        )

    checks.append(
        _check(
            "data_contracts_publishable",
            str(data_quality.get("status", "FAIL")).upper() != "FAIL",
            f"status={data_quality.get('status', 'UNKNOWN')}",
        )
    )

    approved = float(portfolio.get("approved_units", 0) or 0)
    p_mode = str(portfolio.get("mode", "paper") or "paper").lower()
    policy_mode = str(policy.get("deployment_mode", portfolio.get("policy_mode", "paper")) or "paper").lower()
    production_safe = bool(release_gate.get("production_eligible")) and str(release_gate.get("release_state", "")).upper() == "PRODUCTION" and policy_mode == "production" and p_mode == "production"
    checks.append(
        _check(
            "approved_stake_fail_closed",
            approved <= 1e-9 or production_safe,
            f"approved={approved:.2f} release={release_gate.get('release_state')} policy={policy_mode} portfolio={p_mode}",
        )
    )
    allocated = portfolio.get("candidate_allocated_units", portfolio.get("paper_or_shadow_allocated_units", portfolio.get("paper_allocated_units")))
    if _finite(allocated):
        checks.append(
            _check(
                "approved_not_above_allocation",
                approved <= float(allocated) + 1e-9,
                f"approved={approved:.2f} allocated={float(allocated):.2f}",
            )
        )

    drift_details = monitor.get("drift_details") or {}
    drift_scores = {k: _num(v.get("stability_score")) for k, v in drift_details.items() if isinstance(v, dict)}
    trend_rows = prior_trend[-12:]
    prior_scores = [float(x["live_readiness_score"]) for x in trend_rows if _finite(x.get("live_readiness_score"))]
    trend_delta = None
    if prior_scores and _finite(monitor.get("live_readiness_score")):
        baseline = float(pd.Series(prior_scores[-5:]).median())
        trend_delta = float(monitor["live_readiness_score"]) - baseline

    critical_failures = [c for c in checks if c["severity"] == "ERROR" and not c["passed"]]
    warnings = list(dict.fromkeys((monitor.get("alerts") or []) + (release_gate.get("blockers") or [])))
    if trend_delta is not None and trend_delta <= -15:
        warnings.append(f"live readiness dropped {abs(trend_delta):.1f} points versus the recent median")
    status = "FAIL" if critical_failures else "WARN" if warnings or str(data_quality.get("status", "")).upper() == "WARN" else "PASS"

    snapshot = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "identity": {
            "platform_version": meta.get("platform_version"),
            "model_version": meta.get("model_version"),
            "season": meta.get("season"),
            "week": meta.get("week"),
            "model_generated_at": meta.get("generated_at"),
            "updated_at_ct": meta.get("updated_at_ct"),
            "prediction_rows": int(len(pred)),
        },
        "release": {
            "state": release_gate.get("release_state", "UNKNOWN"),
            "production_eligible": bool(release_gate.get("production_eligible")),
            "engineering_ready": bool(release_gate.get("engineering_ready")),
            "historical_edge_ready": bool(release_gate.get("historical_edge_ready")),
            "live_evidence_ready": bool(release_gate.get("live_evidence_ready")),
            "checks": release_gate.get("checks") or [],
            "blockers": release_gate.get("blockers") or [],
            "next_requirements": release_gate.get("next_requirements") or [],
        },
        "model": {
            "validation": meta.get("validation"),
            "margin_mae": _num(metrics.get("margin_mae")),
            "margin_baseline_mae": _num(metrics.get("margin_baseline_mae")),
            "margin_release_weight": _num(metrics.get("margin_release_weight")),
            "total_mae": _num(metrics.get("total_mae")),
            "total_baseline_mae": _num(metrics.get("total_baseline_mae")),
            "total_release_weight": _num(metrics.get("total_release_weight")),
            "win_brier": _num(metrics.get("win_brier")),
            "win_log_loss": _num(metrics.get("win_log_loss")),
            "win_ece": _num(metrics.get("win_ece")),
            "selection_uses_evaluation": bool(metrics.get("selection_uses_evaluation", False)),
        },
        "coverage": {
            "games": int(market.get("games", len(pred)) or 0),
            "moneyline": int(market.get("moneyline", 0) or 0),
            "spread": int(market.get("spread", 0) or 0),
            "total": int(market.get("total", 0) or 0),
            "dynamic_advanced": _num(adv.get("dynamic_coverage", adv.get("live_coverage", 0)), 0.0),
            "context": _num(ctx.get("coverage"), 0.0),
            "weather": _num(ctx.get("weather_coverage"), 0.0),
            "multi_book": _num(intel.get("multi_book_coverage"), 0.0),
        },
        "monitoring": {
            "status": monitor.get("status", "UNKNOWN"),
            "live_readiness_score": _num(monitor.get("live_readiness_score"), 0.0),
            "scores": monitor.get("scores") or {},
            "drift": drift_details,
            "drift_scores": drift_scores,
            "alerts": monitor.get("alerts") or [],
            "recent_readiness_delta": trend_delta,
        },
        "data_quality": data_quality,
        "portfolio": portfolio,
        "historical_evidence": _compact_evidence(evidence, live=False),
        "live_evidence": _compact_evidence(live, live=True),
        "tier_validation": live.get("tier_validation") or meta.get("tier_validation") or {},
        "reconciliation": {
            "status": "FAIL" if critical_failures else "PASS",
            "checks": checks,
            "errors": [c for c in checks if not c["passed"] and c["severity"] == "ERROR"],
        },
        "trend": trend_rows,
        "warnings": warnings,
        "meaning": "Publication/audit snapshot. Readiness and evidence are reported separately; this is not a guarantee of future profitability.",
    }
    return snapshot


def _trend_record(snapshot: dict) -> dict:
    ident = snapshot.get("identity") or {}
    monitoring = snapshot.get("monitoring") or {}
    portfolio = snapshot.get("portfolio") or {}
    return {
        "generated_at": snapshot.get("generated_at"),
        "model_generated_at": ident.get("model_generated_at"),
        "season": ident.get("season"),
        "week": ident.get("week"),
        "platform_version": ident.get("platform_version"),
        "release_state": (snapshot.get("release") or {}).get("state"),
        "publication_status": snapshot.get("status"),
        "live_readiness_score": monitoring.get("live_readiness_score"),
        "distribution_stability": (monitoring.get("scores") or {}).get("distribution_stability"),
        "approved_units": portfolio.get("approved_units", 0),
        "data_quality_status": (snapshot.get("data_quality") or {}).get("status"),
    }


def _append_trend(snapshot: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = _trend_record(snapshot)
    existing = _load_trend(p, limit=200)
    signature = (record.get("model_generated_at"), record.get("season"), record.get("week"))
    if existing:
        last = existing[-1]
        last_sig = (last.get("model_generated_at"), last.get("season"), last.get("week"))
        if last_sig == signature:
            return
    with p.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _fmt(v, digits=3):
    return "—" if not _finite(v) else f"{float(v):.{digits}f}"


def _pct(v):
    return "—" if not _finite(v) else f"{100 * float(v):.2f}%"


def render_run_report(snapshot: dict) -> str:
    i = snapshot.get("identity") or {}
    r = snapshot.get("release") or {}
    m = snapshot.get("model") or {}
    mon = snapshot.get("monitoring") or {}
    p = snapshot.get("portfolio") or {}
    h = snapshot.get("historical_evidence") or {}
    live = snapshot.get("live_evidence") or {}
    tiers = snapshot.get("tier_validation") or {}
    rec = snapshot.get("reconciliation") or {}
    blockers = r.get("blockers") or []
    alerts = mon.get("alerts") or []
    lines = [
        "# Harbin CFB Run Report",
        "",
        f"**Platform / core:** {i.get('platform_version','?')} / {i.get('model_version','?')}  ",
        f"**Season / Week:** {i.get('season')} / {i.get('week')}  ",
        f"**Publication status:** {snapshot.get('status')}  ",
        f"**Release state:** {r.get('state')}  ",
        f"**Reconciliation:** {rec.get('status')}  ",
        "",
        "## Model validation",
        f"- Margin MAE: **{_fmt(m.get('margin_mae'))}** vs baseline **{_fmt(m.get('margin_baseline_mae'))}**; release weight **{_fmt(m.get('margin_release_weight'),2)}**.",
        f"- Total MAE: **{_fmt(m.get('total_mae'))}** vs baseline **{_fmt(m.get('total_baseline_mae'))}**; release weight **{_fmt(m.get('total_release_weight'),2)}**.",
        f"- Brier / log loss / ECE: **{_fmt(m.get('win_brier'),4)} / {_fmt(m.get('win_log_loss'),4)} / {_fmt(m.get('win_ece'),4)}**.",
        f"- Selection uses untouched evaluation outcomes: **{m.get('selection_uses_evaluation')}**.",
        "",
        "## Monitoring and execution",
        f"- Live readiness: **{_fmt(mon.get('live_readiness_score'),1)}/100**; distribution stability **{_fmt((mon.get('scores') or {}).get('distribution_stability'),1)}/100**.",
        f"- Portfolio mode: **{str(p.get('mode','paper')).upper()}**; proposed **{_fmt(p.get('proposed_units'),2)}u**; approved **{_fmt(p.get('approved_units'),2)}u**.",
        f"- Historical evidence: **{h.get('bets',0)} bets**, ROI **{_pct(h.get('roi'))}**, CLV **{_pct(h.get('avg_clv'))}**.",
        f"- Independent live evidence: **{live.get('graded_bets',0)} bets**, ROI **{_pct(live.get('roi'))}**, CLV **{_pct(live.get('avg_clv'))}**.",
        "",
        "## Market × tier forward validation",
        f"- Display-tier ledger: **{tiers.get('graded_tier_bets',0)} flat-1u decisions** · status **{tiers.get('status','EARLY_SAMPLE')}** · validated cells **{tiers.get('validated_cells',0)}**.",
    ]
    for row in (tiers.get("matrix") or []):
        if int(row.get("graded_bets",0) or 0) <= 0:
            continue
        lines.append(
            f"- {str(row.get('market','')).upper()} {row.get('tier')}: "
            f"{row.get('wins',0)}-{row.get('losses',0)}-{row.get('pushes',0)} · "
            f"{_pct(row.get('hit_rate'))} hit · {_fmt(row.get('flat_units'),2)}u · "
            f"{_pct(row.get('flat_roi'))} ROI · CLV {_fmt(row.get('avg_execution_clv'),3)} · "
            f"{row.get('status')}"
        )
    lines += [
        "",
        "## Current blockers",
    ]
    lines.extend([f"- {x}" for x in blockers] or ["- None."])
    lines += ["", "## Operational alerts"]
    lines.extend([f"- {x}" for x in alerts] or ["- None."])
    lines += [
        "",
        "## Interpretation",
        "A green software run, a high readiness score, or good model error metrics do not establish a profitable betting edge. Historical and independent forward evidence remain separate release requirements.",
        "",
    ]
    return "\n".join(lines)


AUDIT_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Harbin Model Audit</title><style>*{box-sizing:border-box}body{margin:0;background:#0f1113;color:#f3f4f5;font-family:Inter,Arial,sans-serif}.wrap{max-width:1260px;margin:28px auto;padding:0 18px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.muted{color:#969ba1}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{background:#171a1e;border:1px solid #292d32;border-radius:12px;padding:16px}.k{color:#91979e;font-size:11px;text-transform:uppercase;letter-spacing:.08em}.v{font-size:25px;font-weight:800;margin-top:7px}.good{color:#6fd17a}.warn{color:#e0b34f}.bad{color:#ef7777}table{width:100%;border-collapse:collapse;background:#15181b;border-radius:10px;overflow:hidden}th,td{text-align:left;padding:10px;border-bottom:1px solid #292d32;vertical-align:top}th{color:#92979d;font-size:11px}a{color:#74a7ff}.section{margin-top:24px}.check{display:flex;justify-content:space-between;gap:14px;padding:8px 0;border-bottom:1px solid #292d32}.check:last-child{border-bottom:0}.yes{color:#6fd17a}.no{color:#ef7777}.small{font-size:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}.bar{height:8px;background:#272b30;border-radius:8px;overflow:hidden;margin-top:5px}.fill{height:100%;background:#6f9cff}@media(max-width:780px){.cols{grid-template-columns:1fr}.top{display:block}}</style></head><body><div class="wrap"><div class="top"><div><h1>Harbin CFB · System Audit</h1><div class="muted">One reconciled publication snapshot: validation, monitoring, evidence, release gates and portfolio execution. Readiness is not a profitability guarantee.</div></div><div><a href="./">← Picks dashboard</a> · <a href="run_report.md">Run report</a></div></div><div id="status" class="section muted">Loading audit snapshot…</div><div id="cards" class="grid"></div><div class="cols"><div class="section"><h2>Release gates</h2><div id="gates" class="card"></div></div><div class="section"><h2>Publication reconciliation</h2><div id="recon" class="card"></div></div></div><div class="section"><h2>Monitoring & drift</h2><div id="drift" class="card"></div></div><div class="section"><h2>Model validation</h2><table id="validation"></table></div><div class="cols"><div class="section"><h2>Historical evidence</h2><div id="hist" class="card"></div></div><div class="section"><h2>Independent live evidence</h2><div id="live" class="card"></div></div></div><div class="section"><h2>Portfolio execution</h2><div id="portfolio" class="card"></div></div><div class="section"><h2>Alerts & blockers</h2><div id="alerts" class="card"></div></div></div><script>const get=async f=>{try{let r=await fetch(f+'?t='+Date.now());return r.ok?await r.json():null}catch(e){return null}};const fmt=(x,d=2)=>x==null||Number.isNaN(Number(x))?'—':Number(x).toFixed(d);const pct=x=>x==null||Number.isNaN(Number(x))?'—':(100*Number(x)).toFixed(2)+'%';const yn=x=>x?'yes':'no';get('audit_snapshot.json').then(s=>{if(!s){document.getElementById('status').textContent='No audit snapshot has been published.';return}let i=s.identity||{},r=s.release||{},m=s.model||{},mon=s.monitoring||{},cov=s.coverage||{},p=s.portfolio||{},dq=s.data_quality||{},h=s.historical_evidence||{},l=s.live_evidence||{},rec=s.reconciliation||{};document.getElementById('status').innerHTML=`Updated <b>${i.updated_at_ct||i.model_generated_at||s.generated_at}</b> · Season ${i.season} Week ${i.week} · Platform ${i.platform_version||'?' } · publication <b>${s.status}</b>`;let cards=[['Release state',r.state,''],['Approved stake',p.approved_units??0,'u'],['Data contracts',dq.status||'UNKNOWN',''],['Live readiness',mon.live_readiness_score,'/100'],['Distribution stability',(mon.scores||{}).distribution_stability,'/100'],['Margin MAE',m.margin_mae,' pts'],['Total MAE',m.total_mae,' pts'],['Win Brier',m.win_brier,''],['Calibration ECE',m.win_ece,''],['Dynamic features',100*(cov.dynamic_advanced||0),'%'],['Multi-book',100*(cov.multi_book||0),'%'],['Live graded bets',l.graded_bets??0,'']];document.getElementById('cards').innerHTML=cards.map(c=>`<div class=card><div class=k>${c[0]}</div><div class=v>${typeof c[1]==='number'?fmt(c[1],c[0].includes('MAE')?2:c[0].includes('Brier')||c[0].includes('ECE')?3:1):c[1]}${c[2]}</div></div>`).join('');let checks=r.checks||[];document.getElementById('gates').innerHTML=checks.length?checks.map(x=>`<div class=check><div><b>${String(x.name).replaceAll('_',' ')}</b><div class='muted small'>${x.requirement||''}</div></div><div class=${yn(x.passed)}>${x.passed?'PASS':'BLOCK'}</div></div>`).join(''):'No release-gate checks.';let rc=rec.checks||[];document.getElementById('recon').innerHTML=rc.map(x=>`<div class=check><div><b>${String(x.name).replaceAll('_',' ')}</b><div class='muted small'>${x.detail||''}</div></div><div class=${yn(x.passed)}>${x.passed?'PASS':'FAIL'}</div></div>`).join('')||'No reconciliation checks.';let drift=mon.drift||{},dk=Object.keys(drift);document.getElementById('drift').innerHTML=dk.length?dk.map(k=>{let d=drift[k]||{},v=Number(d.stability_score||0);return `<div><b>${k}</b> · stability ${fmt(v,1)}/100 · mean shift ${fmt(d.mean_z,2)}σ · volatility ratio ${fmt(d.std_ratio,2)}<div class=bar><div class=fill style="width:${Math.max(0,Math.min(100,v))}%"></div></div></div><br>`}).join(''):'No historical distribution reference was available; monitoring remains fail-visible.';document.getElementById('validation').innerHTML=`<tr><th>Metric</th><th>Deployed</th><th>Baseline / guard</th></tr><tr><td>Margin MAE</td><td>${fmt(m.margin_mae)}</td><td>${fmt(m.margin_baseline_mae)} · weight ${fmt(m.margin_release_weight,2)}</td></tr><tr><td>Total MAE</td><td>${fmt(m.total_mae)}</td><td>${fmt(m.total_baseline_mae)} · weight ${fmt(m.total_release_weight,2)}</td></tr><tr><td>Brier</td><td>${fmt(m.win_brier,4)}</td><td>chronological untouched evaluation</td></tr><tr><td>Log loss</td><td>${fmt(m.win_log_loss,4)}</td><td>chronological untouched evaluation</td></tr><tr><td>ECE</td><td>${fmt(m.win_ece,4)}</td><td>chronological untouched evaluation</td></tr>`;document.getElementById('hist').innerHTML=`Bets: <b>${h.bets??0}</b><br>ROI: <b>${pct(h.roi)}</b> · Units: ${fmt(h.units)} · CLV: ${pct(h.avg_clv)}<br>ROI 95% CI: ${JSON.stringify(h.roi_ci_95||[])}<br><span class=muted>${h.status||''}</span>`;document.getElementById('live').innerHTML=`Graded bets: <b>${l.graded_bets??0}</b><br>ROI: <b>${pct(l.roi)}</b> · Units: ${fmt(l.units)} · CLV: ${pct(l.avg_clv)}<br>Verified CLV samples: ${l.clv_samples??0}<br><span class=muted>${l.clv_method||l.status||''}</span>`;let br=p.bankroll_risk||{};document.getElementById('portfolio').innerHTML=`Mode: <b>${String(p.mode||'paper').toUpperCase()}</b> · proposed ${fmt(p.proposed_units)}u · allocated ${fmt(p.candidate_allocated_units??p.paper_or_shadow_allocated_units??p.paper_allocated_units)}u · approved <b>${fmt(p.approved_units)}u</b><br>Bankroll risk multiplier: ${fmt(br.risk_multiplier,2)} · current drawdown: ${fmt(br.current_drawdown,2)}u<br><span class=muted>Actual approved stake remains zero unless release, policy, live-ledger, execution and portfolio controls all pass.</span>`;let alerts=[...(s.warnings||[])];document.getElementById('alerts').innerHTML=alerts.length?'<ul>'+alerts.map(x=>`<li>${x}</li>`).join('')+'</ul>':'No current alerts or blockers.'})</script></body></html>'''


def write_reporting_bundle(
    pred: pd.DataFrame,
    meta: dict,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
    reports_dir: str | Path = "reports",
    history_dir: str | Path = "history",
) -> tuple[dict, dict]:
    out = Path(output_dir)
    docs = Path(docs_dir)
    reports = Path(reports_dir)
    history = Path(history_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)

    monitor = _read_json(out / "live_monitoring.json") or meta.get("live_monitoring") or {}
    gate = _read_json(out / "release_gate.json") or meta.get("release_gate") or {}
    dq = _read_json(out / "data_quality.json") or meta.get("data_quality") or {}
    portfolio = _read_json(out / "portfolio_summary.json") or meta.get("portfolio") or {}
    evidence = _read_json(reports / "evidence_report.json")
    live = _read_json(reports / "live_performance.json")
    policy = _read_json(reports / "production_policy.json")
    trend_path = history / "audit_snapshots_v1.jsonl"
    prior = _load_trend(trend_path, limit=12)

    snapshot = build_audit_snapshot(pred, meta, monitor, gate, dq, portfolio, evidence, live, policy, prior)
    report = render_run_report(snapshot)
    for p in (out / "audit_snapshot.json", docs / "audit_snapshot.json"):
        p.write_text(json.dumps(snapshot, indent=2))
    (out / "RUN_REPORT.md").write_text(report)
    (docs / "run_report.md").write_text(report)
    (docs / "audit.html").write_text(AUDIT_HTML)
    _append_trend(snapshot, trend_path)

    validation = validate_publication_files(output_dir=out, docs_dir=docs)
    (out / "publication_validation.json").write_text(json.dumps(validation, indent=2))
    (docs / "publication_validation.json").write_text(json.dumps(validation, indent=2))
    if validation["status"] == "FAIL":
        raise RuntimeError("Publication reconciliation failed: " + "; ".join(x["detail"] for x in validation["errors"]))
    return snapshot, validation


def validate_publication_files(output_dir: str | Path = "outputs", docs_dir: str | Path = "docs") -> dict:
    out = Path(output_dir)
    docs = Path(docs_dir)
    snapshot = _read_json(out / "audit_snapshot.json")
    public_snapshot = _read_json(docs / "audit_snapshot.json")
    metadata = _read_json(docs / "metadata.json")
    gate = _read_json(docs / "release_gate.json")
    portfolio = _read_json(docs / "portfolio_summary.json")
    monitor = _read_json(docs / "live_monitoring.json")
    dq = _read_json(docs / "data_quality.json")
    checks = []
    checks.append(_check("audit_snapshot_present", bool(snapshot), "outputs/audit_snapshot.json exists and parses"))
    checks.append(_check("public_snapshot_present", bool(public_snapshot), "docs/audit_snapshot.json exists and parses"))
    if snapshot and public_snapshot:
        checks.append(_check("snapshot_copy_exact", _json_equal(snapshot, public_snapshot), "outputs and docs audit snapshots are identical"))
    ident = snapshot.get("identity") or {}
    if metadata:
        checks.append(_check("metadata_season", metadata.get("season") == ident.get("season"), f"docs={metadata.get('season')} snapshot={ident.get('season')}"))
        checks.append(_check("metadata_week", metadata.get("week") == ident.get("week"), f"docs={metadata.get('week')} snapshot={ident.get('week')}"))
        checks.append(_check("metadata_platform", metadata.get("platform_version") == ident.get("platform_version"), f"docs={metadata.get('platform_version')} snapshot={ident.get('platform_version')}"))
    else:
        checks.append(_check("metadata_present", False, "docs/metadata.json missing or invalid"))
    if gate:
        checks.append(_check("release_copy", gate.get("release_state") == (snapshot.get("release") or {}).get("state"), f"docs={gate.get('release_state')} snapshot={(snapshot.get('release') or {}).get('state')}"))
    else:
        checks.append(_check("release_present", False, "docs/release_gate.json missing or invalid"))
    if portfolio:
        checks.append(_check("portfolio_copy", _num(portfolio.get("approved_units"), 0.0) == _num((snapshot.get("portfolio") or {}).get("approved_units"), 0.0), f"docs={portfolio.get('approved_units')} snapshot={(snapshot.get('portfolio') or {}).get('approved_units')}"))
    else:
        checks.append(_check("portfolio_present", False, "docs/portfolio_summary.json missing or invalid"))
    if monitor:
        checks.append(_check("monitor_copy", _num(monitor.get("live_readiness_score"), -999) == _num((snapshot.get("monitoring") or {}).get("live_readiness_score"), -998), f"docs={monitor.get('live_readiness_score')} snapshot={(snapshot.get('monitoring') or {}).get('live_readiness_score')}"))
    else:
        checks.append(_check("monitor_present", False, "docs/live_monitoring.json missing or invalid"))
    checks.append(_check("data_quality_present", bool(dq), "docs/data_quality.json exists and parses"))
    errors = [c for c in checks if not c["passed"] and c["severity"] == "ERROR"]
    return {"status": "FAIL" if errors else "PASS", "checks": checks, "errors": errors}
