"""Conservative official CFB announcement discovery; snippets are never facts.

Only vetted conference publishers are automated initially. Unsupported school
sites, PDF-only reports and undated pages remain explicit coverage gaps.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse
from xml.etree import ElementTree

from .public_inputs import utc, canon, sid

OFFICIAL_DOMAINS = frozenset({"secsports.com", "theacc.com", "bigten.org", "big12sports.com",
    "theamerican.org", "getsomemaction.com", "sunbeltsports.org", "conferenceusa.com", "themw.com", "pac-12.com"})


def publisher(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port or parsed.fragment:
        return None
    host = (parsed.hostname or "").lower()
    return next((domain for domain in OFFICIAL_DOMAINS if host == domain or host.endswith("." + domain)), None)


class ArticleMetadata(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.data, self.buffer, self.inside = [], [], False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type", "").lower() == "application/ld+json":
            self.inside, self.buffer = True, []

    def handle_data(self, data):
        if self.inside:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.inside:
            self.inside = False
            try:
                data = json.loads("".join(self.buffer))
                pending = data if isinstance(data, list) else [data]
                while pending:
                    row = pending.pop()
                    if not isinstance(row, dict):
                        continue
                    pending.extend(row.get("@graph") or [])
                    kinds = row.get("@type")
                    if any(k in {"Article", "NewsArticle"} for k in (kinds if isinstance(kinds, list) else [kinds])):
                        self.data.append(row)
            except (TypeError, ValueError):
                pass


def evaluate_article(article, game, players, *, as_of):
    """Exact roster IDs and explicit declarations; no depth-chart inference."""
    result = {"starters": {}, "player_designations": [], "rejections": []}
    try:
        if not publisher(article["source_url"]):
            raise ValueError("PUBLISHER_NOT_VETTED_OFFICIAL_CONFERENCE")
        now, capture, kickoff = utc(as_of), utc(article["captured_utc"]), utc(game["date"])
        if not capture <= now < kickoff or now - capture > timedelta(hours=6):
            raise ValueError("OFFICIAL_CAPTURE_NOT_FRESH_PREGAME")
        html = article["html"]
        if hashlib.sha256(html.encode()).hexdigest() != article["snapshot_sha256"]:
            raise ValueError("OFFICIAL_ARTICLE_HASH_MISMATCH")
        metadata = ArticleMetadata(html).data
        if len(metadata) != 1:
            raise ValueError("NO_UNIQUE_OFFICIAL_ARTICLE")
        data = metadata[0]
        published = utc(data["datePublished"])
        modified = utc(data.get("dateModified", data["datePublished"]))
        if not published <= modified <= capture or capture - published > timedelta(days=7):
            raise ValueError("OFFICIAL_ARTICLE_CLOCK_INVALID_OR_STALE")
        body, title = data.get("articleBody"), data.get("headline")
        if not isinstance(body, str) or not body.strip() or not isinstance(title, str):
            raise ValueError("NO_EXPLICIT_ARTICLE_BODY")
        if "<" in body or "<" in title:
            raise ValueError("ARTICLE_BODY_MARKUP_REQUIRES_SEPARATE_PARSER")
        # Conference-wide roundups cannot bind a sentence to this matchup.
        home, away = (re.escape(game[k]) for k in ("home_team", "away_team"))
        matchup = rf"\b(?:{home}\s+(?:vs\.?|at|versus|@)\s+{away}|{away}\s+(?:vs\.?|at|versus|@)\s+{home})(?=\s*(?:[|:;,\-–—(]|$|\bWeek\b))"
        if not re.search(matchup, title, re.I):
            raise ValueError("OFFICIAL_HEADLINE_MATCHUP_NOT_EXPLICIT")
        weeks = {int(w) for w in re.findall(r"\bweek\s+(\d+)\b", title + " " + body, re.I)}
        if weeks != {int(game["week"])}:
            raise ValueError("OFFICIAL_ARTICLE_WEEK_MISMATCH_OR_AMBIGUOUS")
        provenance = {k: article[k] for k in ("source_url", "captured_utc", "snapshot_sha256")}
        provenance["source_publication_utc"] = published.isoformat()
        candidates = {}
        for side in ("home", "away"):
            tid = sid(game.get(side + "_id"))
            roster = [p for p in players if sid(p.get("team_id")) == tid]
            by_name = {}
            for player in roster:
                name = str(player.get("full_name") or player.get("display_name") or "").strip()
                if name and sid(player.get("athlete_id")):
                    by_name.setdefault(name, []).append(player)
            for name, matches in by_name.items():
                same_name = [p for p in players if str(p.get("full_name") or p.get("display_name") or "").strip().lower() == name.lower()
                             and sid(p.get("team_id")) in {sid(game.get("home_id")), sid(game.get("away_id"))}]
                if len({(sid(p.get("team_id")), sid(p.get("athlete_id"))) for p in same_name}) != 1:
                    continue
                if len({sid(p["athlete_id"]) for p in matches}) != 1:
                    continue
                player = matches[0]
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
                    if not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", sentence, re.I):
                        continue
                    if re.search(r"\b(if|unless|expected|likely|could|may|might|would|should|projected|next|last)\b|\?", sentence, re.I):
                        continue
                    suffix = re.split(re.escape(name), sentence, flags=re.I)[-1]
                    source = {"team_id": tid, "athlete_id": sid(player["athlete_id"]), "player_name": name, "evidence": sentence, **provenance}
                    qb = str(player.get("position_abbreviation") or player.get("position") or "").upper() in {"QB", "QUARTERBACK"}
                    if qb and re.match(r"\s+(?:will start|is the starting quarterback|has been named the starter)\b", suffix, re.I) and not re.search(r"\b(not|won't|backup|reserve)\b", sentence, re.I):
                        candidates.setdefault(tid, {})[source["athlete_id"]] = source
                    status = None
                    if re.match(r"\s+(?:has been ruled out|will not play|is out)\b", suffix, re.I):
                        status = "OUT"
                    elif re.match(r"\s+(?:is active|will play)\b", suffix, re.I) and published.date() == kickoff.date():
                        status = "EXPLICIT_AVAILABLE"
                    if status:
                        result["player_designations"].append({**source, "designation": status})
        for tid, values in candidates.items():
            if len(values) == 1:
                player = next(iter(values.values()))
                if not any(p["team_id"] == tid and p["athlete_id"] == player["athlete_id"] and p["designation"] == "OUT" for p in result["player_designations"]):
                    result["starters"][tid] = player
            else:
                result["rejections"].append("CONFLICTING_STARTER_DECLARATIONS")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        result["rejections"].append(str(exc))
    return result


class OfficialResearch:
    def __init__(self, session, max_pages=48):
        self.session, self.remaining = session, max_pages
        self.cache = {}

    def collect(self, game, players):
        query = f'college football "{game["away_team"]}" "{game["home_team"]}" {utc(game["date"]).date()} quarterback starter availability injury (' + " OR ".join("site:" + d for d in sorted(OFFICIAL_DOMAINS)) + ")"
        url = "https://www.bing.com/search?" + urlencode({"format": "rss", "q": query})
        result = {"query": query, "search_url": url, "search_results_are_facts": False,
                  "articles": [], "errors": [], "starters": {}, "player_designations": []}
        try:
            response = self.session.get(url, timeout=8, allow_redirects=False)
            response.raise_for_status()
            if response.status_code != 200 or response.url != url:
                raise ValueError("SEARCH_REDIRECT_OR_URL_MISMATCH")
            result["search_response_sha256"] = hashlib.sha256(response.content).hexdigest()
            leads = [item.findtext("link") or "" for item in ElementTree.fromstring(response.content).findall("./channel/item")]
            result["official_leads"] = list(dict.fromkeys(link for link in leads if publisher(link)))[:3]
            for link in result["official_leads"]:
                if link not in self.cache:
                    if self.remaining <= 0:
                        result["errors"].append("OFFICIAL_PAGE_BUDGET_REACHED")
                        continue
                    self.remaining -= 1
                    page = self.session.get(link, timeout=8, allow_redirects=False)
                    page.raise_for_status()
                    if page.status_code != 200 or page.url != link or "text/html" not in page.headers.get("Content-Type", ""):
                        result["errors"].append("OFFICIAL_PAGE_REDIRECT_OR_UNSUPPORTED_FORMAT")
                        continue
                    html = page.content.decode("utf-8")
                    self.cache[link] = {"source_url": link, "html": html,
                        "snapshot_sha256": hashlib.sha256(html.encode()).hexdigest(),
                        "captured_utc": datetime.now(timezone.utc).isoformat()}
                article = self.cache[link]
                evaluated = evaluate_article(article, game, players, as_of=datetime.now(timezone.utc).isoformat())
                result["articles"].append({**{k: v for k, v in article.items() if k != "html"}, **evaluated})
            for item in result["articles"]:
                result["player_designations"].extend(item["player_designations"])
            teams = {tid for item in result["articles"] for tid in item["starters"]}
            for tid in teams:
                records = [item["starters"][tid] for item in result["articles"] if tid in item["starters"]]
                if len({r["athlete_id"] for r in records}) == 1:
                    player = records[-1]
                    if not any(p["team_id"] == tid and p["athlete_id"] == player["athlete_id"] and p["designation"] == "OUT" for p in result["player_designations"]):
                        result["starters"][tid] = player
                else:
                    result["errors"].append("CONFLICTING_OFFICIAL_STARTER_ARTICLES")
        except Exception as exc:
            result["errors"].append(f"{type(exc).__name__}: {exc}")
        return result
