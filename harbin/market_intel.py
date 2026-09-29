from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import numpy as np
import requests

from .advanced import canon_team
from .data import ESPN_CORE_ODDS, parse_espn_odds
from .market import no_vig


def _best_american(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return max(vals) if vals else np.nan


class MarketIntelligence:
    """Build a multi-book market view without feeding prices into the score model."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; HarbinSportsAnalytics/4.0)",
            "Accept": "application/json,text/plain,*/*",
        })
        self.errors: list[str] = []

    def _event_quotes(self, game):
        url = ESPN_CORE_ODDS.format(game_id=game.game_id)
        try:
            r = self.session.get(url, params={"limit": 100}, timeout=9)
            r.raise_for_status()
            items = r.json().get("items") or []
            resolved = []
            for item in items:
                if isinstance(item, dict) and item.get("$ref"):
                    rr = self.session.get(item["$ref"], timeout=7)
                    rr.raise_for_status()
                    item = rr.json()
                if isinstance(item, dict):
                    q = parse_espn_odds(item, {canon_team(game.home_team)}, {canon_team(game.away_team)})
                    if any(q.get(k) is not None for k in ("home_ml","away_ml","home_spread","market_total")):
                        resolved.append(q)
            return str(game.game_id), resolved
        except Exception as exc:
            return str(game.game_id), exc

    @staticmethod
    def _summary(game, quotes):
        base = {
            "provider": game.provider or "primary",
            "home_ml": game.home_ml,
            "away_ml": game.away_ml,
            "home_spread": game.home_spread,
            "market_total": game.market_total,
        }
        # One sportsbook gets one vote in the consensus. The primary live quote
        # can also be returned by ESPN Core, so merge by provider instead of
        # appending it twice and accidentally overweighting that book.
        all_quotes = [dict(q) for q in (quotes or [])]
        if any(base[k] is not None for k in ("home_ml","away_ml","home_spread","market_total")):
            bp = str(base.get("provider") or "primary").strip().lower()
            existing = next(
                (q for q in all_quotes if str(q.get("provider") or "unknown").strip().lower() == bp),
                None,
            )
            if existing is None:
                all_quotes.append(dict(base))
            else:
                for k in ("home_ml","away_ml","home_spread","market_total"):
                    if existing.get(k) is None and base.get(k) is not None:
                        existing[k] = base[k]

        providers=[]
        for q in all_quotes:
            p=str(q.get("provider") or "unknown")
            if p not in providers: providers.append(p)
        spreads=[float(q["home_spread"]) for q in all_quotes if q.get("home_spread") is not None]
        totals=[float(q["market_total"]) for q in all_quotes if q.get("market_total") is not None]
        home_ml=[float(q["home_ml"]) for q in all_quotes if q.get("home_ml") is not None]
        away_ml=[float(q["away_ml"]) for q in all_quotes if q.get("away_ml") is not None]
        novig_h=[]
        for q in all_quotes:
            if q.get("home_ml") is not None and q.get("away_ml") is not None:
                try:
                    _,ph=no_vig(q["away_ml"],q["home_ml"]); novig_h.append(float(ph))
                except Exception: pass
        return {
            "market_book_count":len(providers),"market_books":" | ".join(providers[:12]),
            "consensus_home_spread":float(np.median(spreads)) if spreads else np.nan,
            "consensus_total":float(np.median(totals)) if totals else np.nan,
            "spread_market_std":float(np.std(spreads)) if len(spreads)>1 else 0.0 if spreads else np.nan,
            "total_market_std":float(np.std(totals)) if len(totals)>1 else 0.0 if totals else np.nan,
            "best_home_ml":_best_american(home_ml),"best_away_ml":_best_american(away_ml),
            "consensus_home_novig_probability":float(np.mean(novig_h)) if novig_h else np.nan,
            "market_consensus_quality":min(1.0,len(providers)/4.0),
        }

    def attach(self, games, pred):
        if pred.empty: return pred.copy(), {"coverage":0.0,"multi_book_coverage":0.0,"errors":[]}
        by_id={str(g.game_id):g for g in games}; quotes={}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs=[ex.submit(self._event_quotes,g) for g in games]
            for fut in as_completed(futs):
                gid,result=fut.result()
                if isinstance(result,Exception):
                    self.errors.append(f"{gid}: {type(result).__name__}: {result}"); quotes[gid]=[]
                else: quotes[gid]=result
        out=pred.copy(); covered=multi=0
        for i,row in out.iterrows():
            gid=str(row.game_id); s=self._summary(by_id[gid],quotes.get(gid,[]))
            for k,v in s.items(): out.at[i,k]=v
            covered += int(s["market_book_count"]>0); multi += int(s["market_book_count"]>=2)
        return out,{"coverage":covered/len(out),"multi_book_coverage":multi/len(out),"errors":self.errors[-20:]}
