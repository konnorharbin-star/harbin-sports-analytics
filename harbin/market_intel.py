from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os

import numpy as np
import requests

from .advanced import canon_team
from .data import ESPN_CORE_ODDS, parse_espn_odds
from .market import no_vig


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


class MarketIntelligence:
    """Consensus and line-shopping layer. Fair-score projections never consume market prices."""

    def __init__(self,max_workers=10):
        self.max_workers=max_workers
        self.s=requests.Session()
        self.s.headers.update({"User-Agent":"HarbinSportsAnalytics/7.1","Accept":"application/json,text/plain,*/*"})
        self.errors=[]
        self.odds_key=os.getenv("THE_ODDS_API_KEY","").strip()
        self.odds_api_used=False
        self.external={}

    def _espn(self,g):
        try:
            r=self.s.get(ESPN_CORE_ODDS.format(game_id=g.game_id),params={"limit":100},timeout=9)
            r.raise_for_status()
            items=r.json().get("items") or []
            out=[]
            for it in items:
                if isinstance(it,dict) and it.get("$ref"):
                    rr=self.s.get(it["$ref"],timeout=7)
                    rr.raise_for_status()
                    it=rr.json()
                if isinstance(it,dict):
                    q=parse_espn_odds(it,{canon_team(g.home_team)},{canon_team(g.away_team)})
                    if any(q.get(k) is not None for k in ("home_ml","away_ml","home_spread","market_total")):
                        out.append(q)
            return str(g.game_id),out
        except Exception as e:
            return str(g.game_id),e

    def _load_odds_api(self):
        if not self.odds_key:
            return
        try:
            r=self.s.get(
                "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds",
                params={"apiKey":self.odds_key,"regions":"us,us2","markets":"h2h,spreads,totals","oddsFormat":"american","dateFormat":"iso"},
                timeout=20,
            )
            r.raise_for_status()
            self.odds_api_used=True
            for ev in r.json():
                key=(canon_team(ev.get("away_team")),canon_team(ev.get("home_team")))
                qs=[]
                for book in ev.get("bookmakers") or []:
                    q={
                        "provider":book.get("title") or book.get("key"),
                        "home_ml":None,"away_ml":None,"home_spread":None,"market_total":None,
                        "home_spread_price":None,"away_spread_price":None,
                        "over_price":None,"under_price":None,"last_update":book.get("last_update"),
                    }
                    for m in book.get("markets") or []:
                        mk=m.get("key"); outs=m.get("outcomes") or []
                        if mk=="h2h":
                            for x in outs:
                                if canon_team(x.get("name"))==key[1]: q["home_ml"]=x.get("price")
                                elif canon_team(x.get("name"))==key[0]: q["away_ml"]=x.get("price")
                        elif mk=="spreads":
                            for x in outs:
                                if canon_team(x.get("name"))==key[1]:
                                    q["home_spread"]=x.get("point"); q["home_spread_price"]=x.get("price")
                                elif canon_team(x.get("name"))==key[0]:
                                    q["away_spread_price"]=x.get("price")
                        elif mk=="totals":
                            for x in outs:
                                side=str(x.get("name") or "").lower()
                                if side=="over": q["market_total"]=x.get("point"); q["over_price"]=x.get("price")
                                elif side=="under": q["under_price"]=x.get("price")
                    if any(q[k] is not None for k in ("home_ml","away_ml","home_spread","market_total")):
                        qs.append(q)
                self.external[key]=qs
        except Exception as e:
            self.errors.append(f"The Odds API: {type(e).__name__}: {e}")

    @staticmethod
    def _summary(g,quotes):
        base={"provider":g.provider or "primary","home_ml":g.home_ml,"away_ml":g.away_ml,"home_spread":g.home_spread,"market_total":g.market_total}
        qs=[dict(x) for x in quotes or []]
        bp=str(base["provider"]).lower()
        ex=next((q for q in qs if str(q.get("provider") or "").lower()==bp),None)
        if ex is None:
            qs.append(base)
        else:
            for k in ("home_ml","away_ml","home_spread","market_total"):
                if ex.get(k) is None: ex[k]=base.get(k)
        unique={str(q.get("provider") or "unknown").lower():q for q in qs}
        qs=list(unique.values())
        providers=[str(q.get("provider") or "unknown") for q in qs]
        sp=_finite_values(q.get("home_spread") for q in qs)
        to=_finite_values(q.get("market_total") for q in qs)
        hm=_finite_values(q.get("home_ml") for q in qs)
        am=_finite_values(q.get("away_ml") for q in qs)
        ph=[]
        for q in qs:
            if q.get("home_ml") is not None and q.get("away_ml") is not None:
                try:
                    _,x=no_vig(q["away_ml"],q["home_ml"]); ph.append(float(x))
                except Exception:
                    pass

        home_spread_quotes=[q for q in qs if math.isfinite(_num(q.get("home_spread")))]
        home_best_q=max(home_spread_quotes,key=lambda q:_num(q.get("home_spread"))) if home_spread_quotes else None
        away_best_q=min(home_spread_quotes,key=lambda q:_num(q.get("home_spread"))) if home_spread_quotes else None
        total_quotes=[q for q in qs if math.isfinite(_num(q.get("market_total")))]
        over_best_q=min(total_quotes,key=lambda q:_num(q.get("market_total"))) if total_quotes else None
        under_best_q=max(total_quotes,key=lambda q:_num(q.get("market_total"))) if total_quotes else None

        best_home_spread=_num(home_best_q.get("home_spread")) if home_best_q else np.nan
        best_away_spread=-_num(away_best_q.get("home_spread")) if away_best_q else np.nan
        best_over_total=_num(over_best_q.get("market_total")) if over_best_q else np.nan
        best_under_total=_num(under_best_q.get("market_total")) if under_best_q else np.nan
        spread_range=(max(sp)-min(sp)) if len(sp)>1 else (0. if sp else np.nan)
        total_range=(max(to)-min(to)) if len(to)>1 else (0. if to else np.nan)
        consensus_quality=min(1.,len(providers)/4.)
        if len(sp)>1: consensus_quality*=max(.55,1-min(1.,float(np.std(sp))/2.5)*.35)
        if len(to)>1: consensus_quality*=max(.55,1-min(1.,float(np.std(to))/3.5)*.35)

        return {
            "market_book_count":len(providers),"market_books":" | ".join(providers[:15]),
            "consensus_home_spread":_median(sp),"consensus_total":_median(to),
            "spread_market_std":float(np.std(sp)) if len(sp)>1 else 0. if sp else np.nan,
            "total_market_std":float(np.std(to)) if len(to)>1 else 0. if to else np.nan,
            "spread_market_range":spread_range,"total_market_range":total_range,
            "best_home_spread":best_home_spread,"best_away_spread":best_away_spread,
            "best_home_spread_odds":_num(home_best_q.get("home_spread_price")) if home_best_q else np.nan,
            "best_away_spread_odds":_num(away_best_q.get("away_spread_price")) if away_best_q else np.nan,
            "best_over_total":best_over_total,"best_under_total":best_under_total,
            "best_over_odds":_num(over_best_q.get("over_price")) if over_best_q else np.nan,
            "best_under_odds":_num(under_best_q.get("under_price")) if under_best_q else np.nan,
            "best_home_ml":_best(hm),"best_away_ml":_best(am),
            "consensus_home_novig_probability":float(np.mean(ph)) if ph else np.nan,
            "consensus_probability_std":float(np.std(ph)) if len(ph)>1 else 0. if ph else np.nan,
            "market_consensus_quality":float(consensus_quality),
        }

    def attach(self,games,pred):
        if pred.empty:
            return pred.copy(),{"coverage":0.,"multi_book_coverage":0.,"errors":[]}
        self._load_odds_api(); by={str(g.game_id):g for g in games}; quotes={}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fs=[ex.submit(self._espn,g) for g in games]
            for f in as_completed(fs):
                gid,x=f.result(); quotes[gid]=[] if isinstance(x,Exception) else x
        out=pred.copy(); covered=multi=0
        for i,r in out.iterrows():
            g=by[str(r.game_id)]; q=list(quotes.get(str(r.game_id),[])); q.extend(self.external.get((canon_team(g.away_team),canon_team(g.home_team)),[])); z=self._summary(g,q)
            for k,v in z.items(): out.at[i,k]=v
            covered+=int(z["market_book_count"]>0); multi+=int(z["market_book_count"]>=2)
        return out,{
            "coverage":covered/len(out),"multi_book_coverage":multi/len(out),
            "odds_api_configured":bool(self.odds_key),"odds_api_used":self.odds_api_used,
            "errors":self.errors[-20:],
            "best_line_fields":["best_home_ml","best_away_ml","best_home_spread","best_away_spread","best_over_total","best_under_total"],
        }
