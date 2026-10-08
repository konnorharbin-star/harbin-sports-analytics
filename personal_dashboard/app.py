"""Personal read-only dashboard for the Harbin NFL and CFB models."""
from __future__ import annotations

import hmac
import io
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Harbin Sports | Private", page_icon="🏈", layout="wide")

SOURCES = {
    "CFB": {
        "repo": "harbin-sports-analytics",
        "edges": "outputs/edge_actionable.csv",
        "bets": "outputs/quant_recommendations.csv",
        "board": "outputs/README.md",
    },
    "NFL": {
        "repo": "harbin-nfl-analytics",
        "edges": "outputs/edge_watchlist.csv",
        "bets": "outputs/quant_recommendations.csv",
        "board": "outputs/README.md",
    },
}
OWNER = "konnorharbin-star"


def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return str(os.environ.get(name, "")).strip()


def gate() -> None:
    expected = secret("DASHBOARD_PASSWORD")
    if not expected:
        st.error("Access locked: configure DASHBOARD_PASSWORD before running this dashboard.")
        st.stop()
    if not st.session_state.get("authenticated", False):
        st.title("🔒 Harbin Sports Analytics")
        with st.form("login"):
            supplied = st.text_input("Private dashboard password", type="password")
            submitted = st.form_submit_button("Unlock")
        if submitted:
            if hmac.compare_digest(supplied.encode(), expected.encode()):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def fetch(repo: str, path: str) -> tuple[bytes | None, str]:
    # Only public, already-published model outputs are requested; no credentials.
    url = f"https://raw.githubusercontent.com/{OWNER}/{repo}/main/{path}"
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        return response.content, ""
    except requests.RequestException as exc:
        return None, str(exc)


def table(repo: str, path: str) -> tuple[pd.DataFrame, str]:
    content, error = fetch(repo, path)
    if content is None:
        return pd.DataFrame(), error
    try:
        return pd.read_csv(io.BytesIO(content)), ""
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return pd.DataFrame(), str(exc)


def metadata(repo: str) -> str:
    content, _ = fetch(repo, "outputs/README.md")
    if not content:
        return "Last run unavailable"
    raw = content.decode("utf-8", errors="replace")
    stamp = re.search(r"\*\*Updated:\*\*\s*([^\n]+)", raw)
    return stamp.group(1).strip() if stamp else "Run time not published"


def clean_view(frame: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    columns = [
        "date", "away_team", "home_team", "market", "side", "line", "odds",
        "probability", "edge", "ev", "edge_priority", "badge",
        "book", "price_evidence_status", "timing_action",
        "release_state", "status", "reason",
    ]
    if frame.empty:
        return frame
    keep = [col for col in columns if col in frame.columns]
    view = frame[keep].copy() if keep else frame.copy()
    for col in ("probability", "ev"):
        if col in view:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
    return view.head(limit)


gate()
with st.sidebar:
    st.title("🏈 HARBIN")
    st.caption("PERSONAL SPORTS ANALYTICS")
    sport = st.radio("League", ["CFB", "NFL"], horizontal=True)
    if st.button("Refresh published outputs"):
        fetch.clear()
        st.rerun()
    if st.button("Lock dashboard"):
        st.session_state["authenticated"] = False
        st.rerun()
    st.caption("Read-only. No wagers placed or stakes authorized.")

cfg = SOURCES[sport]
repo = cfg["repo"]
st.title(f"{sport} betting intelligence")
st.caption(f"Model update: {metadata(repo)}")
st.info("Research candidates are NOT approved bets. The model's own release and portfolio gates remain authoritative.")

tabs = st.tabs(["Edge research", "Approved portfolio", "Every game", "Data quality"])

with tabs[0]:
    st.subheader("Evidence-ranked edge candidates")
    candidates, issue = table(repo, cfg["edges"])
    if issue:
        st.warning(f"Research file currently unavailable: {issue}")
    elif candidates.empty:
        st.info("No published research edges in this run.")
    else:
        if "edge_priority" in candidates:
            levels = sorted(candidates["edge_priority"].dropna().astype(str).unique())
            selected = st.multiselect("Edge categories", levels, default=levels)
            candidates = candidates[candidates["edge_priority"].astype(str).isin(selected)]
        st.metric("Research candidates", len(candidates))
        st.dataframe(clean_view(candidates), hide_index=True, use_container_width=True)
        st.caption("Prices can go stale. These signals do not override the production release gate.")

with tabs[1]:
    st.subheader("Model portfolio recommendations")
    portfolio, issue = table(repo, cfg["bets"])
    if issue:
        st.warning(f"Portfolio file currently unavailable: {issue}")
    elif portfolio.empty:
        st.info("No portfolio records published. No betting approval is implied.")
    else:
        st.dataframe(clean_view(portfolio), hide_index=True, use_container_width=True)
        st.caption("Check approval status and stake in the source report before betting.")

with tabs[2]:
    st.subheader("Complete weekly game projections")
    st.write("Open the canonical prediction board, which includes games even when no bet qualifies.")
    board = f"https://github.com/{OWNER}/{repo}/blob/main/outputs/README.md"
    st.link_button(f"Open {sport} latest prediction board", board)
    st.caption("The model's own live board remains the source of truth for all games and projected scores.")

with tabs[3]:
    st.subheader("Publication diagnostics")
    st.write(f"Repository: [{OWNER}/{repo}](https://github.com/{OWNER}/{repo})")
    st.write(f"Retrieved (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("Published source files are refreshed at most every five minutes, or manually.")
    st.write("This dashboard reads existing public model outputs. Its password protects the interface, not the source repositories.")
    st.write("Never place tokens, passwords, bankroll records, or personal bets into a public repository.")
