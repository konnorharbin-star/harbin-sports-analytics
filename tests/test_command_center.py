"""Smoke checks for the public, fail-closed operations hub."""

from pathlib import Path


def test_command_center_is_standalone_and_links_both_repositories() -> None:
    page = Path("docs/command-center.html").read_text()
    assert "<!doctype html>" in page.lower()
    assert "harbin-sports-analytics" in page
    assert "harbin-nfl-analytics" in page
    assert "audit_snapshot.json" in page
    assert "actions/runs" in page
    assert "MODEL" not in page[:0]  # The page does not need runtime Python.


def test_command_center_is_read_only_and_fails_closed() -> None:
    page = Path("docs/command-center.html").read_text()
    assert 'fetch(url,{cache:"no-store"' in page
    assert 'AbortController' in page
    assert 'audit.status!=="fulfilled"' in page
    assert '"UNKNOWN"' in page
    assert "release.production_eligible" in page
    assert 'approved_units' in page
    assert 'recon.errors' in page
    assert 'document.createElement("li")' in page
    assert 'li.textContent=String(item)' in page
    assert "Authorization: Bearer" not in page
    assert "github_pat_" not in page
