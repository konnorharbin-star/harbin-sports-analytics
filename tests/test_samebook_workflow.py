"""Static guardrails: free data, read-only GitHub, manual wagering only."""
from pathlib import Path


def test_research_clv_workflow_is_free_and_read_only():
    flow=Path(".github/workflows/free-samebook-movement.yml").read_text()
    assert "workflow_dispatch:" in flow
    assert "schedule:" in flow
    assert "branches: [main]" in flow
    assert "contents: read" in flow
    assert "contents: write" not in flow
    assert "ops-evidence" in flow
    assert "actions/upload-artifact@v4" in flow
    assert "platform_ops.samebook_movement" in flow
    assert "pip install" not in flow
    for prohibited in ("place_bet", "submit_bet", "THE_ODDS_API_KEY",
                       "Stripe", "billing", "deposit"):
        assert prohibited not in flow


def test_research_clv_does_not_claim_official_close():
    py=Path("platform_ops/samebook_movement.py").read_text()
    doc=Path("docs/FREE_SAMEBOOK_MOVEMENT.md").read_text()
    assert "official_closing_prices_verified" in py
    assert "price_executability_verified" in py
    assert "source_reconciliation_status" in py
    assert "MAX_CLOSE_AGE" in py
    assert "official bookmaker closing price" in doc
    assert "Never automatically bet" in doc
    assert "Never introduce paid dependencies" in doc
    assert "raw.githubusercontent.com" in py
    assert "api.the-odds-api.com" not in py


def test_collector_is_nonexecution_and_has_no_money_parameters():
    source=Path("platform_ops/samebook_movement.py").read_text()
    assert "wager_executed" in source
    assert "automatic_betting_enabled" in source
    assert "paid_sources_used" in source
    assert "requests.post" not in source
    assert "urlopen(request" in source
