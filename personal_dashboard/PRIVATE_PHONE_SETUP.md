# Private phone access — $0 Tailscale Serve setup

This path requires **your own computer to stay switched on** while you use the dashboard. It does **not** use a public hosting service, paid server, or public internet exposure.

## 1. On the computer running the dashboard

Install [Python 3.11+](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads), and [Tailscale](https://tailscale.com/download). Sign in to Tailscale with your own account.

Clone the dashboard branch:

```bash
git clone --branch feature/private-betting-dashboard https://github.com/konnorharbin-star/harbin-sports-analytics.git
cd harbin-sports-analytics
python -m pip install -r personal_dashboard/requirements.txt
```

Set a **long unique password** as a private environment variable. On macOS/Linux:

```bash
export DASHBOARD_PASSWORD='choose-a-long-unique-password'
```

On Windows PowerShell:

```powershell
$env:DASHBOARD_PASSWORD = 'choose-a-long-unique-password'
```

Run the dashboard **bound to loopback**, not all local-network interfaces:

```bash
python -m streamlit run personal_dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Keep this terminal running. In a second terminal, once Tailscale is signed in:

```bash
tailscale serve --bg 8501
tailscale serve status
```

Tailscale may ask you to authorize HTTPS/Serve in the admin panel. The status output gives the private `https://...ts.net` address.

**Use `tailscale serve`, not `tailscale funnel`. Funnel publishes to the open internet.**

## 2. On your phone

1. Install the Tailscale mobile app and sign into **the same tailnet** as your computer.
2. Turn on Tailscale.
3. Open the private HTTPS address reported by `tailscale serve status` in your phone's browser.
4. Enter the dashboard password. Optionally use your browser's **Add to Home Screen** option.

No public website is created. The address is reachable through your tailnet while your computer and Tailscale service are running.

## 3. Shut it down

```bash
tailscale serve reset
```

Then stop the Streamlit process (Ctrl+C). Disconnect Tailscale on your phone if desired.

## Safety checks

- Do not set `--server.address 0.0.0.0`, open router ports, or enable Funnel.
- Do not commit credentials to GitHub. The existing GitHub model outputs are **public**, even though the dashboard front door is private.
- Streamlit's local password gate is an extra check, **not a production-grade authentication provider**. Tailscale device/user access is the primary private network boundary.
- Anyone admitted to the same tailnet who is permitted by your ACLs may reach the Serve endpoint; use a single-user tailnet or configure restrictive access controls.
- Keep Tailscale and Python libraries updated. Avoid recording private bankroll information in GitHub.
- Refreshes are constrained by the source model runs, a five-minute dashboard cache, and your computer being online.
- The Tailscale Personal plan is currently $0 for non-commercial use, subject to the provider's current terms; no guarantee it will remain unchanged.
- This dashboard does not place wagers, access sportsbook accounts, or imply that research signals are profitable.

## Deployment status

This document supplies **instructions**, not a hosted instance. We cannot log into your phone, sign up for your Tailscale account, or keep your computer running on your behalf.
