"""Safety and deployability checks for free postgame research grading."""
from pathlib import Path


def test_grade_workflow_read_only_and_free():
    flow = Path(".github/workflows/free-forward-grading.yml").read_text()
    assert "contents: read" in flow
    assert "contents: write" not in flow
    assert "workflow_dispatch:" in flow
    assert "branches: [main]" in flow
    assert "schedule:" in flow
    assert "cron:" in flow
    assert "platform_ops.grade_observations" in flow
    assert "refs/heads/ops-evidence" in flow
    assert "upload-artifact@v4" in flow
    assert "pip install" not in flow
    for prohibited in ("place_bet(", "sportsbook_api", "THE_ODDS_API_KEY", "Stripe", "deposit("):
        assert prohibited not in flow


def test_grade_docs_state_research_only():
    content = Path("docs/FREE_FORWARD_GRADING.md").read_text()
    assert "cannot place wagers" in content
    assert "hypothetical one-unit ROI" in content
    assert "not independently certified" in content
    assert "no paid data subscriptions" in content


def test_grade_module_only_fetches_existing_public_results():
    module = Path("platform_ops/grade_observations.py").read_text()
    assert 'RESULTS_PATH = "reports/live_graded_bets.csv"' in module
    assert "automatic_betting_enabled" in module
    assert "verified_executable_entry" in module
    assert "bet_was_placed" in module
    assert "grade_candidate" in module
    assert "get_public_bytes" in module
    assert "requests.post" not in module
