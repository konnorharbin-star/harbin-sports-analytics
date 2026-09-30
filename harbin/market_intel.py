from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _best(vals):
    x=_finite_values(vals)
    return max(x) if x else np.nan


def _median(vals):
    x=_finite_values(vals)
    return float(np.median(x)) if x else np.nan


def _num(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


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
        # Covers API display names such as "Alabama Crimson Tide" vs schedule
        # school name "Alabama" without introducing global fuzzy matching.
        if len(t)>=4 and (c.startswith(t) or t.startswith(c)): return True
    return False


def parse_action_network_game(game: dict) -> dict:
    """Normalize one Action Network scoreboard game into independent book quotes.

    The public web endpoint is unofficial/undocumented, so this parser is defensive and
    is treated only as an opportunistic no-key supplement. A schema change yields no
    quotes rather than fabricated prices.
    """
    teams=game.get("teams") or []
    hid=game.get("home_team_id"); aid=game.get("away_team_id")
    home=next((x for x in teams if x.get("id")==hid),{})
    away=next((x for x in teams if x.get("id")==aid),{})
    home_names=_team_candidates(home); away_names=_team_candidates(away)
    markets=game.get("markets") or {}
    if isinstance(markets,list):
        markets={str(i):m for i,m in enumerate(markets) if isinstance(m,dict)}
    quotes=[]
    if not isinstance(markets,dict): markets={}
    for market_id,market in markets.items():
        if not isinstance(market,dict): continue
        event=market.get("event") or {}
        if not isinstance(event,dict): continue
        ml=event.get("moneyline") or []; sp=event.get("spread") or []; tot=event.get("total") or []
        rows=[]
        for x in list(ml)+list(sp)+list(tot):
            if isinstance(x,dict): rows.append(x)
        book_ids=[]
        for x in rows:
            b=x.get("book_id")
            if b is not None: book_ids.append(str(b))
        book_id=book_ids[0] if book_ids else str(market_id)
        provider=f"ActionNetwork book {book_id}"
        q={"provider":provider,"home_ml":None,"away_ml":None,"home_spread":None,"market_total":None,
           "home_spread_price":None,"away_spread_price":None,"over_price":None,"under_price":None}
        for x in ml:
            if not isinstance(x,dict): continue
            side=str(x.get("side") or "").lower()
            if side=="home": q["home_ml"]=x.get("odds")
            elif side=="away": q["away_ml"]=x.get("odds")
        for x in sp:
            if not isinstance(x,dict): continue
            side=str(x.get("side") or "").lower(); val=x.get("value"); odds=x.get("odds")
            if side=="home": q["home_spread"]=val; q["home_spread_price"]=odds
            elif side=="away": q["away_spread_price"]=odds
        for x in tot:
            if not isinstance(x,dict): continue
            side=str(x.get("side") or "").lower(); val=x.get("value"); odds=x.get("odds")
            if side=="over": q["market_total"]=val; q["over_price"]=odds
            elif side=="under":
                if q["market_total"] is None: q["market_total"]=val
                q["under_price"]=odds
        if any(q[k] is not None for k in ("home_ml","away_ml","home_spread","market_total")):
            quotes.append(q)
    return {"home_names":home_names,"away_names":away_names,"quotes":quotes,"start_time":game.get("start_time")}


class MarketIntelligence:
    """Consensus and line-shopping layer. Fair-score projections never consume market prices."""

    def __init__(self,max_workers=10):
        self.max_workers=max_workers
        self.s=requests.Session()
        self.s.headers.update({"User-Agent":"Mozilla/5.0 (compatible; HarbinSportsAnalytics/7.2)","Accept":"application/json,text/plain,*/*"})
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
        # Ambiguous name matches (e.g. Miami) fail closed.
        return []

    def _load_odds_api(self):
        if not self.odds_key: return
        try:
            r=self.s.get("https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds",params={"apiKey":self.odds_key,"regions":"us,us2","markets":"h2h,spreads,totals","oddsFormat":"american","dateFormat":"iso"},timeout=20)
            r.raise_for_status(); self.odds_api_used=True
            for ev in r.json():
                key=(canon_team(ev.get("away_team")),canon_team(ev.get("home_team"))); qs=[]
                for book in ev.get("bookmakers") or []:
                    q={"provider":book.get("title") or book.get("key"),"home_ml":None,"away_ml":None,"home_spread":None,"market_total":None,"home_spread_price":None,"away_spread_price":None,"over_price":None,"under_price":None,"last_update":book.get("last_update")}
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
        base={"provider":g.provider or "primary","home_ml":g.home_ml,"away_ml":g.away_ml,"home_spread":g.home_spread,"market_total":g.market_total}
        qs=[dict(x) for x in quotes or []]; bp=str(base["provider"]).lower(); ex=next((q for q in qs if str(q.get("provider") or "").lower()==bp),None)
        if ex is None: qs.append(base)
        else:
            for k in ("home_ml","away_ml","home_spread","market_total"):
                if ex.get(k) is None: ex[k]=base.get(k)
        unique={str(q.get("provider") or "unknown").lower():q for q in qs}; qs=list(unique.values()); providers=[str(q.get("provider") or "unknown") for q in qs]
        sp=_finite_values(q.get("home_spread") for q in qs); to=_finite_values(q.get("market_total") for q in qs); hm=_finite_values(q.get("home_ml") for q in qs); am=_finite_values(q.get("away_ml") for q in qs); ph=[]
        for q in qs:
            if q.get("home_ml") is not None and q.get("away_ml") is not None:
                try: _,x=no_vig(q["away_ml"],q["home_ml"]); ph.append(float(x))
                except Exception: pass
        hq=[q for q in qs if math.isfinite(_num(q.get("home_spread")))]; hb=max(hq,key=lambda q:_num(q.get("home_spread"))) if hq else None; ab=min(hq,key=lambda q:_num(q.get("home_spread"))) if hq else None; tq=[q for q in qs if math.isfinite(_num(q.get("market_total")))]; ob=min(tq,key=lambda q:_num(q.get("market_total"))) if tq else None; ub=max(tq,key=lambda q:_num(q.get("market_total"))) if tq else None
        bhr=_num(hb.get("home_spread")) if hb else np.nan; bar=-_num(ab.get("home_spread")) if ab else np.nan; bot=_num(ob.get("market_total")) if ob else np.nan; but=_num(ub.get("market_total")) if ub else np.nan; sr=(max(sp)-min(sp)) if len(sp)>1 else (0. if sp else np.nan); tr=(max(to)-min(to)) if len(to)>1 else (0. if to else np.nan); cq=min(1.,len(providers)/4.)
        if len(sp)>1: cq*=max(.55,1-min(1.,float(np.std(sp))/2.5)*.35)
        if len(to)>1: cq*=max(.55,1-min(1.,float(np.std(to))/3.5)*.35)
        return {"market_book_count":len(providers),"market_books":" | ".join(providers[:15]),"consensus_home_spread":_median(sp),"consensus_total":_median(to),"spread_market_std":float(np.std(sp)) if len(sp)>1 else 0. if sp else np.nan,"total_market_std":float(np.std(to)) if len(to)>1 else 0. if to else np.nan,"spread_market_range":sr,"total_market_range":tr,"best_home_spread":bhr,"best_away_spread":bar,"best_home_spread_odds":_num(hb.get("home_spread_price")) if hb else np.nan,"best_away_spread_odds":_num(ab.get("away_spread_price")) if ab else np.nan,"best_over_total":bot,"best_under_total":but,"best_over_odds":_num(ob.get("over_price")) if ob else np.nan,"best_under_odds":_num(ub.get("under_price")) if ub else np.nan,"best_home_ml":_best(hm),"best_away_ml":_best(am),"consensus_home_novig_probability":float(np.mean(ph)) if ph else np.nan,"consensus_probability_std":float(np.std(ph)) if len(ph)>1 else 0. if ph else np.nan,"market_consensus_quality":float(cq)}

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
        return out,{"coverage":covered/len(out),"multi_book_coverage":multi/len(out),"odds_api_configured":bool(self.odds_key),"odds_api_used":self.odds_api_used,"action_network_enabled":self.action_network_enabled,"action_network_used":self.action_network_used,"action_network_match_coverage":action_matched/len(out),"errors":self.errors[-20:],"best_line_fields":["best_home_ml","best_away_ml","best_home_spread","best_away_spread","best_over_total","best_under_total"]}
