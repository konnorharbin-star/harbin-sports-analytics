"""Policy and static workflow tests: no paid dependencies or wager execution."""
from pathlib import Path


def test_free_workflow_never_calls_bookmakers() -> None:
    flow = Path(".github/workflows/free-research-observer.yml").read_text()
    assert "workflow_dispatch:" in flow
    assert "branches: [main]" in flow
    assert "paths:" in flow
    assert "'.github/workflows/free-research-observer.yml'" in flow
    assert 'cron: "17 */6 * * *"' in flow
    assert "pull_request:" not in flow
    assert "raw.githubusercontent" not in flow  # Sources defined in vetted Python module.
    assert "actions/upload-artifact@v4" in flow
    assert "ops-evidence" in flow
    assert "refs/heads/ops-evidence" in flow
    assert "platform_ops.free_observer capture" in flow
    assert "platform_ops.free_observer archive" in flow
    assert "pip install" not in flow
    for bad in ("BETFAIR", "place_bet", "THE_ODDS_API_KEY", "stripe", "betting_api"):
        assert bad not in flow


def test_documented_storage_and_no_wager_guarantee() -> None:
    contents = Path("docs/FREE_RESEARCH_OBSERVER.md").read_text()
    assert "cannot place" in contents
    assert "RESEARCH_ONLY" in contents
    assert "Python 3.12" in contents
    assert "SQLite" in contents
    assert "ops-evidence" in contents
