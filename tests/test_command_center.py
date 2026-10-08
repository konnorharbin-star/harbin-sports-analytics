"""Smoke checks for the public, fail-closed operations hub."""

from pathlib import Path


def test_command_center_is_standalone_and_links_both_repositories() -> None:
    page = Path("docs/command-center.html").read_text()
    assert "<!doctype html>" in page.lower()
    assert "harbin-sports-analytics" in page
    assert "harbin-nfl-analytics" in page
    assert "audit_snapshot.json" in page
    assert "actions/runs" in page
    assert 'id="cfb-card"' in page
    assert 'id="nfl-card"' in page


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

def test_command_center_embedded_javascript_syntax(tmp_path) -> None:
    """Check the script parses when Node is available on the CI runner."""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        return
    page = Path("docs/command-center.html").read_text()
    scripts = re.findall(r"<script>(.*?)</script>", page, flags=re.DOTALL)
    assert len(scripts) == 1
    script_path = tmp_path / "command-center.js"
    script_path.write_text(scripts[0])
    result = subprocess.run(
        [node, "--check", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
