"""Personal read-only dashboard for the Harbin NFL and CFB models."""
from __future__ import annotations

import hmac
import io
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup
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



def latest_board_path(repo: str) -> tuple[str | None, str]:
    """Resolve the weekly HTML from the model's published README, never hardcode week."""
    content, error = fetch(repo, "outputs/README.md")
    if content is None:
        return None, error
    raw = content.decode("utf-8", errors="replace")
    # Markdown links in published output; resolve only HTML filenames under outputs/.
    links = re.findall(r"\[[^]]+\]\(([^)]+\.html)\)", raw)
    for link in links:
        if re.fullmatch(r"[A-Za-z0-9_.-]+\.html", link) and (
            link.startswith("nfl_week_") or link.startswith("cfb_model_")
        ):
            return f"outputs/{link}", ""
    return None, "No current weekly HTML board published in outputs/README.md"


def game_board(repo: str) -> tuple[pd.DataFrame, str]:
    path, problem = latest_board_path(repo)
    if not path:
        return pd.DataFrame(), problem
    raw, error = fetch(repo, path)
    if raw is None:
        return pd.DataFrame(), error
    soup = BeautifulSoup(raw, "html.parser")
    records = []
    for tr in soup.select("table tbody tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 6:
            continue
        matchup = cells[0].get_text(" ", strip=True)
        if " @ " not in matchup:
            continue
        away, home = matchup.split(" @ ", 1)
        # HTML rows contain the model's published percentages and picks.
        records.append({
            "Away": away.strip(),
            "Home": home.strip(),
            "Projected score (away–home)": cells[1].get_text(" ", strip=True),
            "Win probability (projected favorite)": cells[2].get_text(" ", strip=True),
            "Moneyline": cells[3].get_text(" ", strip=True),
            "Spread": cells[4].get_text(" ", strip=True),
            "Total": cells[5].get_text(" ", strip=True),
        })
    if not records:
        return pd.DataFrame(), "Weekly HTML board contained no parseable game rows"
    return pd.DataFrame(records), ""


def portfolio_classification(frame: pd.DataFrame, sport: str) -> pd.Series:
    """Approval is NEVER inferred from a STRONG/BET/LEAN research label."""
    if frame.empty:
        return pd.Series(dtype=str)
    stake = pd.to_numeric(frame.get("portfolio_stake_units", pd.Series(0, index=frame.index)) if sport == "NFL"
        else frame.get("stake_units", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    if sport == "NFL":
        production = frame.get("production_signal", pd.Series("", index=frame.index)).astype(str).str.upper()
        action = frame.get("portfolio_action", pd.Series("", index=frame.index)).astype(str).str.upper()
        # Release may separately block staking. Never label approved without verified release state.
        return pd.Series(["No approved wager" for _ in frame.index], index=frame.index) if not (
            (production.isin(["BET", "STRONG"]) & action.isin(["BET", "STRONG"]) & (stake > 0)).any()
        ) else pd.Series(["Portfolio stake present — verify release gate" if s > 0 and p in ("BET", "STRONG") and a in ("BET", "STRONG") else "No approved wager"
                          for s, p, a in zip(stake, production, action)], index=frame.index)
    return pd.Series(["Research allocation only — verify release gate" if s > 0 else "No approved wager"
                      for s in stake], index=frame.index)


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
    st.subheader("Portfolio status — never inferred from edge labels")
    portfolio, issue = table(repo, cfg["bets"])
    if issue:
        st.warning(f"Portfolio source unavailable: {issue}")
    elif portfolio.empty:
        st.info("No published portfolio records.")
    else:
        portfolio = portfolio.copy()
        portfolio.insert(0, "Dashboard interpretation", portfolio_classification(portfolio, sport))
        st.dataframe(portfolio[[c for c in [
            "away_team", "home_team", "Dashboard interpretation",
            "quant_signal", "production_signal", "research_signal",
            "quant_market", "quant_side", "quant_price", "quant_odds",
            "quant_probability", "quant_ev", "stake_units", "portfolio_stake_units",
            "portfolio_action", "portfolio_limit_reason",
        ] if c in portfolio.columns]], hide_index=True, use_container_width=True)
        st.caption("A model allocation or label is not proof of production approval. Confirm release state in source reports.")

with tabs[2]:
    st.subheader("Every published game — full weekly board")
    games, issue = game_board(repo)
    if issue:
        st.warning(f"Could not load full weekly board: {issue}")
    else:
        st.metric("Games on board", len(games))
        keyword = st.text_input("Search team", key=f"team_{sport}")
        if keyword:
            games = games[
                games["Away"].str.contains(keyword, case=False, regex=False) |
                games["Home"].str.contains(keyword, case=False, regex=False)
            ]
        st.dataframe(games, use_container_width=True, hide_index=True)
        st.caption("Winner probability and market tags are copied from the canonical HTML board, not recalculated here.")
    board_path, _ = latest_board_path(repo)
    if board_path:
        st.link_button("Open original published board",
                       f"https://github.com/{OWNER}/{repo}/blob/main/{board_path}")

with tabs[3]:
    st.subheader("Publication diagnostics")
    st.write(f"Repository: [{OWNER}/{repo}](https://github.com/{OWNER}/{repo})")
    st.write(f"Retrieved (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("Published source files are refreshed at most every five minutes, or manually.")
    st.write("This dashboard reads existing public model outputs. Its password protects the interface, not the source repositories.")
    st.write("Never place tokens, passwords, bankroll records, or personal bets into a public repository.")
