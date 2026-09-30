from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os

import numpy as np
import requests

from .advanced import canon_team
from .data import ESPN_CORE_ODDS, parse_espn_odds
from .market import no_vig

ACTION_NETWORK_URL="https://api.actionnetwork.com/web/v2/scoreboard/ncaaf"


def _finite_values(vals):
    out=[]
    for v in vals:
        try:
            x=float(v)
            if math.isfinite(x): out.append(x)
        except Exception:
            pass
    return out


def _median(vals):
    x=_finite_values(vals)
    return float(np.median(x)) if x else np.nan


def _num(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _provider(q):
    return str((q or {}).get("provider") or "unknown").strip() or "unknown"


def _provider_key(q):
    return " ".join(_provider(q).lower().split())


def _quote_timestamp(q):
    raw=(q or {}).get("last_update") or (q or {}).get("updated_at") or (q or {}).get("timestamp")
    if not raw:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        x=datetime.fromisoformat(str(raw).replace("Z","+00:00"))
        if x.tzinfo is None: x=x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _quote_completeness(q):
    fields=("home_ml","away_ml","home_spread","market_total","home_spread_price","away_spread_price","over_price","under_price")
    return sum(math.isfinite(_num((q or {}).get(k))) for k in fields)


def _clean_quote(q):
    q=dict(q or {})
    out={
        "provider":_provider(q),
        "source":str(q.get("source") or "unknown"),
        "last_update":q.get("last_update") or q.get("updated_at") or q.get("timestamp"),
    }
    for k in ("home_ml","away_ml","home_spread","market_total","home_spread_price","away_spread_price","over_price","under_price"):
        x=_num(q.get(k)); out[k]=float(x) if math.isfinite(x) else None
    return out


def _dedupe_quotes(quotes):
    """Keep one coherent quote per sportsbook, preferring freshness then completeness."""
    chosen={}
    for raw in quotes or []:
        q=_clean_quote(raw); key=_provider_key(q)
        prev=chosen.get(key)
        if prev is None or (_quote_timestamp(q),_quote_completeness(q))>(_quote_timestamp(prev),_quote_completeness(prev)):
            chosen[key]=q
    return list(chosen.values())


def _best_quote(quotes,line_field=None,price_field=None,line_direction=1,odds_field=None):
    candidates=[]
    for q in quotes or []:
        if odds_field is not None:
            odds=_num(q.get(odds_field))
            if math.isfinite(odds): candidates.append((float(odds),q))
            continue
        line=_num(q.get(line_field));
        if not math.isfinite(line): continue
        price=_num(q.get(price_field)) if price_field else np.nan
        price_score=float(price) if math.isfinite(price) else -1e12
        candidates.append(((float(line_direction)*float(line),price_score),q))
    if not candidates: return None
    return max(candidates,key=lambda x:x[0])[1]


def _team_candidates(team: dict) -> set[str]:
    vals=set()
    for k in ("display_name","full_name","short_name","name","location","abbr","abbreviation"):
        v=team.get(k)
        if v: vals.add(canon_team(v))
    return {x for x in vals if x}


def _fuzzy_team_match(target: str, candidates: set[str]) -> bool:
    t=canon_team(target)
    if not t: return False
    for c in candidates:
        if t==c: return True
        if len(t)>=4 and (c.startswith(t) or t.startswith(c)): return True
    return False


def parse_action_network_game(game: dict) -> dict:
    """Normalize one Action Network game into independent, per-book quotes.

    Rows are grouped by `book_id`; data from two books is never combined into one
    synthetic quote. The endpoint is unofficial, so malformed or ambiguous rows fail
    closed instead of fabricating prices.
    """
    teams=game.get("teams") or []
    hid=str(game.get("home_team_id")); aid=str(game.get("away_team_id"))
    home=next((x for x in teams if str(x.get("id"))==hid),{})
    away=next((x for x in teams if str(x.get("id"))==aid),{})
    home_names=_team_candidates(home); away_names=_team_candidates(away)
    markets=game.get("markets") or {}
    if isinstance(markets,list):
        markets={str(i):m for i,m in enumerate(markets) if isinstance(m,dict)}
    if not isinstance(markets,dict): markets={}
    grouped={}

    for market_id,market in markets.items():
        if not isinstance(market,dict): continue
        event=market.get("event") or {}
        if not isinstance(event,dict): continue
        fallback_ts=market.get("last_update") or market.get("updated_at") or event.get("last_update") or event.get("updated_at")
        for kind in ("moneyline","spread","total"):
            rows=event.get(kind) or []
            if not isinstance(rows,list): continue
            for x in rows:
                if not isinstance(x,dict): continue
                book_id=str(x.get("book_id") if x.get("book_id") is not None else market_id)
                q=grouped.setdefault(book_id,{"provider":f"ActionNetwork book {book_id}","source":"action_network","home_ml":None,"away_ml":None,"home_spread":None,"market_total":None,"home_spread_price":None,"away_spread_price":None,"over_price":None,"under_price":None,"last_update":None,"_away_spread":None})
                q["last_update"]=x.get("last_update") or x.get("updated_at") or x.get("timestamp") or q.get("last_update") or fallback_ts
                side=str(x.get("side") or "").lower(); value=x.get("value"); odds=x.get("odds")
                if kind=="moneyline":
                    if side=="home": q["home_ml"]=odds
                    elif side=="away": q["away_ml"]=odds
                elif kind=="spread":
                    if side=="home": q["home_spread"]=value; q["home_spread_price"]=odds
                    elif side=="away": q["_away_spread"]=value; q["away_spread_price"]=odds
                elif kind=="total":
                    if side=="over": q["market_total"]=value; q["over_price"]=odds
                    elif side=="under":
                        if q["market_total"] is None: q["market_total"]=value
                        q["under_price"]=odds

    quotes=[]
    for q in grouped.values():
        if q.get("home_spread") is None and math.isfinite(_num(q.get("_away_spread"))):
            q["home_spread"]=-float(q["_away_spread"])
        q.pop("_away_spread",None)
        if any(q.get(k) is not None for k in ("home_ml","away_ml","home_spread","market_total")):
            quotes.append(_clean_quote(q))
    return {"home_names":home_names,"away_names":away_names,"quotes":quotes,"start_time":game.get("start_time")}


class MarketIntelligence:
    """Consensus, line-shopping, and auditable quote-provenance layer."""

    def __init__(self,max_workers=10):
        self.max_workers=max_workers
        self.s=requests.Session()
        self.s.headers.update({"User-Agent":"Mozilla/5.0 (compatible; HarbinSportsAnalytics/7.3)","Accept":"application/json,text/plain,*/*"})
        self.errors=[]
        self.odds_key=os.getenv("THE_ODDS_API_KEY","").strip()
        self.odds_api_used=False
        self.action_network_enabled=os.getenv("HARBIN_ACTION_NETWORK","1").strip().lower() not in {"0","false","off","no"}
        self.action_network_used=False
        self.external={}
        self.action_games=[]

    def _espn(self,g):
        try:
            r=self.s.get(ESPN_CORE_ODDS.format(game_id=g.game_id),params={"limit":100},timeout=9)
            r.raise_for_status(); items=r.json().get("items") or []; out=[]
            for it in items:
                if isinstance(it,dict) and it.get("$ref"):
                    rr=self.s.get(it["$ref"],timeout=7); rr.raise_for_status(); it=rr.json()
                if isinstance(it,dict):
                    q=parse_espn_odds(it,{canon_team(g.home_team)},{canon_team(g.away_team)})
                    q["source"]="espn_core"; q["last_update"]=it.get("lastUpdated") or it.get("last_update") or it.get("date")
                    if any(q.get(k) is not None for k in ("home_ml","away_ml","home_spread","market_total")): out.append(q)
            return str(g.game_id),out
        except Exception as e: return str(g.game_id),e

    def _load_action_network(self,games):
        if not self.action_network_enabled or not games: return
        try:
            season=int(games[0].season); week=int(games[0].week)
            r=self.s.get(ACTION_NETWORK_URL,params={"season":season,"division":"FBS","week":week,"seasonType":"reg","periods":"event"},timeout=20)
            r.raise_for_status(); data=r.json(); parsed=[]
            for game in data.get("games") or []:
                if isinstance(game,dict):
                    x=parse_action_network_game(game)
                    if x["quotes"]: parsed.append(x)
            self.action_games=parsed; self.action_network_used=bool(parsed)
        except Exception as e:
            self.errors.append(f"Action Network public scoreboard: {type(e).__name__}: {e}")

    def _action_quotes_for(self,g):
        matches=[]
        for x in self.action_games:
            if _fuzzy_team_match(g.home_team,x["home_names"]) and _fuzzy_team_match(g.away_team,x["away_names"]):
                matches.append(x)
        if len(matches)==1: return matches[0]["quotes"]
        return []

    def _load_odds_api(self):
        if not self.odds_key: return
        try:
            r=self.s.get("https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds",params={"apiKey":self.odds_key,"regions":"us,us2","markets":"h2h,spreads,totals","oddsFormat":"american","dateFormat":"iso"},timeout=20)
            r.raise_for_status(); self.odds_api_used=True
            for ev in r.json():
                key=(canon_team(ev.get("away_team")),canon_team(ev.get("home_team"))); qs=[]
                for book in ev.get("bookmakers") or []:
                    q={"provider":book.get("title") or book.get("key"),"source":"odds_api","home_ml":None,"away_ml":None,"home_spread":None,"market_total":None,"home_spread_price":None,"away_spread_price":None,"over_price":None,"under_price":None,"last_update":book.get("last_update")}
                    for m in book.get("markets") or []:
                        mk=m.get("key"); outs=m.get("outcomes") or []
                        if mk=="h2h":
                            for x in outs:
                                if canon_team(x.get("name"))==key[1]: q["home_ml"]=x.get("price")
                                elif canon_team(x.get("name"))==key[0]: q["away_ml"]=x.get("price")
                        elif mk=="spreads":
                            for x in outs:
                                if canon_team(x.get("name"))==key[1]: q["home_spread"]=x.get("point"); q["home_spread_price"]=x.get("price")
                                elif canon_team(x.get("name"))==key[0]: q["away_spread_price"]=x.get("price")
                        elif mk=="totals":
                            for x in outs:
                                side=str(x.get("name") or "").lower()
                                if side=="over": q["market_total"]=x.get("point"); q["over_price"]=x.get("price")
                                elif side=="under": q["under_price"]=x.get("price")
                    if any(q[k] is not None for k in ("home_ml","away_ml","home_spread","market_total")): qs.append(q)
                self.external[key]=qs
        except Exception as e: self.errors.append(f"The Odds API: {type(e).__name__}: {e}")

    @staticmethod
    def _summary(g,quotes):
        base={"provider":g.provider or "primary","source":"primary","home_ml":g.home_ml,"away_ml":g.away_ml,"home_spread":g.home_spread,"market_total":g.market_total,"home_spread_price":None,"away_spread_price":None,"over_price":None,"under_price":None,"last_update":None}
        qs=_dedupe_quotes([dict(x) for x in quotes or []])
        ex=next((q for q in qs if _provider_key(q)==_provider_key(base)),None)
        if ex is None:
            qs.append(_clean_quote(base))
        else:
            # Preserve the freshest provider quote while filling any missing primary fields.
            for k in ("home_ml","away_ml","home_spread","market_total"):
                if ex.get(k) is None and base.get(k) is not None: ex[k]=float(base[k])
        providers=[_provider(q) for q in qs]
        sp=_finite_values(q.get("home_spread") for q in qs); to=_finite_values(q.get("market_total") for q in qs)
        ph=[]
        for q in qs:
            if math.isfinite(_num(q.get("home_ml"))) and math.isfinite(_num(q.get("away_ml"))):
                try: _,x=no_vig(q["away_ml"],q["home_ml"]); ph.append(float(x))
                except Exception: pass

        hb=_best_quote(qs,"home_spread","home_spread_price",1); ab=_best_quote(qs,"home_spread","away_spread_price",-1)
        ob=_best_quote(qs,"market_total","over_price",-1); ub=_best_quote(qs,"market_total","under_price",1)
        hml=_best_quote(qs,odds_field="home_ml"); aml=_best_quote(qs,odds_field="away_ml")
        bhr=_num(hb.get("home_spread")) if hb else np.nan; bar=-_num(ab.get("home_spread")) if ab else np.nan
        bot=_num(ob.get("market_total")) if ob else np.nan; but=_num(ub.get("market_total")) if ub else np.nan
        sr=(max(sp)-min(sp)) if len(sp)>1 else (0. if sp else np.nan); tr=(max(to)-min(to)) if len(to)>1 else (0. if to else np.nan)
        cq=min(1.,len(providers)/4.)
        if len(sp)>1: cq*=max(.55,1-min(1.,float(np.std(sp))/2.5)*.35)
        if len(to)>1: cq*=max(.55,1-min(1.,float(np.std(to))/3.5)*.35)
        if len(ph)>1: cq*=max(.70,1-min(1.,float(np.std(ph))/.08)*.20)
        serial=[_clean_quote(q) for q in sorted(qs,key=lambda q:_provider(q).lower())]
        sources=sorted({str(q.get("source") or "unknown") for q in qs})
        return {
            "market_book_count":len(providers),"market_books":" | ".join(providers[:20]),"market_quote_sources":" | ".join(sources),"market_quotes_json":json.dumps(serial,sort_keys=True,separators=(",",":")),
            "consensus_home_spread":_median(sp),"consensus_total":_median(to),"spread_market_std":float(np.std(sp)) if len(sp)>1 else 0. if sp else np.nan,"total_market_std":float(np.std(to)) if len(to)>1 else 0. if to else np.nan,"spread_market_range":sr,"total_market_range":tr,
            "best_home_spread":bhr,"best_away_spread":bar,"best_home_spread_odds":_num(hb.get("home_spread_price")) if hb else np.nan,"best_away_spread_odds":_num(ab.get("away_spread_price")) if ab else np.nan,"best_home_spread_book":_provider(hb) if hb else None,"best_away_spread_book":_provider(ab) if ab else None,
            "best_over_total":bot,"best_under_total":but,"best_over_odds":_num(ob.get("over_price")) if ob else np.nan,"best_under_odds":_num(ub.get("under_price")) if ub else np.nan,"best_over_book":_provider(ob) if ob else None,"best_under_book":_provider(ub) if ub else None,
            "best_home_ml":_num(hml.get("home_ml")) if hml else np.nan,"best_away_ml":_num(aml.get("away_ml")) if aml else np.nan,"best_home_ml_book":_provider(hml) if hml else None,"best_away_ml_book":_provider(aml) if aml else None,
            "consensus_home_novig_probability":float(np.mean(ph)) if ph else np.nan,"consensus_probability_std":float(np.std(ph)) if len(ph)>1 else 0. if ph else np.nan,"moneyline_pair_book_count":int(len(ph)),"market_consensus_quality":float(cq),
        }

    def attach(self,games,pred):
        if pred.empty: return pred.copy(),{"coverage":0.,"multi_book_coverage":0.,"errors":[]}
        self._load_action_network(games); self._load_odds_api(); by={str(g.game_id):g for g in games}; quotes={}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fs=[ex.submit(self._espn,g) for g in games]
            for f in as_completed(fs):
                gid,x=f.result(); quotes[gid]=[] if isinstance(x,Exception) else x
        out=pred.copy(); covered=multi=action_matched=0
        for i,r in out.iterrows():
            g=by[str(r.game_id)]; aq=self._action_quotes_for(g); action_matched+=int(bool(aq)); q=list(quotes.get(str(r.game_id),[])); q.extend(aq); q.extend(self.external.get((canon_team(g.away_team),canon_team(g.home_team)),[])); z=self._summary(g,q)
            for k,v in z.items(): out.at[i,k]=v
            covered+=int(z["market_book_count"]>0); multi+=int(z["market_book_count"]>=2)
        return out,{"coverage":covered/len(out),"multi_book_coverage":multi/len(out),"odds_api_configured":bool(self.odds_key),"odds_api_used":self.odds_api_used,"action_network_enabled":self.action_network_enabled,"action_network_used":self.action_network_used,"action_network_match_coverage":action_matched/len(out),"errors":self.errors[-20:],"best_line_fields":["best_home_ml","best_away_ml","best_home_spread","best_away_spread","best_over_total","best_under_total"],"best_line_provenance_fields":["best_home_ml_book","best_away_ml_book","best_home_spread_book","best_away_spread_book","best_over_book","best_under_book"]}
