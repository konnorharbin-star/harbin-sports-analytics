# Harbin Private Betting Dashboard — Phase 1

A **personal, password-gated, read-only Streamlit dashboard** for the published CFB and NFL GitHub models. This is an initial foundation, not a deployed website.

## Local launch (free)

1. Install Python 3.11+.
2. From repository root: `python -m pip install -r personal_dashboard/requirements.txt`
3. Set a strong unique password in your terminal: `export DASHBOARD_PASSWORD='your-long-unique-password'` (PowerShell: `$env:DASHBOARD_PASSWORD='your-long-unique-password'`).
4. Run `streamlit run personal_dashboard/app.py`.
5. Open the local URL Streamlit prints and enter that password.

## Security & costs

- **Do not commit a password or a Streamlit secrets file.** Set `DASHBOARD_PASSWORD` using private runtime secrets on any host.
- Existing source repositories are **public**. A dashboard password does not make their underlying model outputs private.
- An ordinary public Streamlit deployment is **not** private without an actual access control layer. Do not deploy this to a public URL without configuring secret handling and authenticated access. For single-user remote access, a private network tunnel or host-level access control can be used, subject to its free-tier limits and terms.
- App reads public GitHub Raw files; it does not need a GitHub API token, paid AI API, or odds subscription.
- Odds and published prices may be stale. The display is research-only and does not place bets.
- All-game views parse the latest linked native model HTML boards into a searchable weekly table. The data is still read-only and retains native market tags.

## Private phone setup

Follow [PRIVATE_PHONE_SETUP.md](PRIVATE_PHONE_SETUP.md) for local Streamlit plus Tailscale Serve (private HTTPS tailnet access, no public hosting). Your computer must remain powered on to provide access. Do not use Tailscale Funnel.

## Sources

- CFB research candidates: `harbin-sports-analytics/outputs/edge_actionable.csv`
- CFB approved portfolio: `harbin-sports-analytics/outputs/quant_recommendations.csv` (shown as unavailable if absent)
- NFL research watchlist: `harbin-nfl-analytics/outputs/edge_watchlist.csv`
- NFL approved portfolio: `harbin-nfl-analytics/outputs/quant_recommendations.csv`
- Run freshness and full projection links: respective `outputs/README.md`

The research tables are intentionally separated from portfolio approval. The app never converts edge candidates into approved wagers.
