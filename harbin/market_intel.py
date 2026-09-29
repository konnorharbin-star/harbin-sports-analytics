from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
import math,os
import numpy as np
import requests
from .advanced import canon_team
from .data import ESPN_CORE_ODDS,parse_espn_odds
from .market import no_vig

def _best(vals):
    x=[float(v) for v in vals if v is not None and math.isfinite(float(v))];return max(x) if x else np.nan
class MarketIntelligence:
    """Consensus/best-price layer. Score projections never consume sportsbook prices."""
    def __init__(self,max_workers=10):
        self.max_workers=max_workers;self.s=requests.Session();self.s.headers.update({"User-Agent":"HarbinSportsAnalytics/6.0","Accept":"application/json,text/plain,*/*"});self.errors=[];self.odds_key=os.getenv("THE_ODDS_API_KEY","").strip();self.odds_api_used=False;self.external={}
    def _espn(self,g):
        try:
            r=self.s.get(ESPN_CORE_ODDS.format(game_id=g.game_id),params={"limit":100},timeout=9);r.raise_for_status();items=r.json().get("items") or [];out=[]
            for it in items:
                if isinstance(it,dict) and it.get("$ref"):
                    rr=self.s.get(it["$ref"],timeout=7);rr.raise_for_status();it=rr.json()
                if isinstance(it,dict):
                    q=parse_espn_odds(it,{canon_team(g.home_team)},{canon_team(g.away_team)})
                    if any(q.get(k) is not None for k in ("home_ml","away_ml","home_spread","market_total")):out.append(q)
            return str(g.game_id),out
        except Exception as e:return str(g.game_id),e
    def _load_odds_api(self):
        if not self.odds_key:return
        try:
            r=self.s.get("https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds",params={"apiKey":self.odds_key,"regions":"us,us2","markets":"h2h,spreads,totals","oddsFormat":"american","dateFormat":"iso"},timeout=20);r.raise_for_status();data=r.json();self.odds_api_used=True
            for ev in data:
                key=(canon_team(ev.get("away_team")),canon_team(ev.get("home_team")));qs=[]
                for book in ev.get("bookmakers") or []:
                    q={"provider":book.get("title") or book.get("key"),"home_ml":None,"away_ml":None,"home_spread":None,"market_total":None}
                    for m in book.get("markets") or []:
                        mk=m.get("key");outs=m.get("outcomes") or []
                        if mk=="h2h":
                            for x in outs:
                                if canon_team(x.get("name"))==key[1]:q["home_ml"]=x.get("price")
                                elif canon_team(x.get("name"))==key[0]:q["away_ml"]=x.get("price")
                        elif mk=="spreads":
                            for x in outs:
                                if canon_team(x.get("name"))==key[1]:q["home_spread"]=x.get("point")
                        elif mk=="totals":
                            over=next((x for x in outs if str(x.get("name")).lower()=="over"),None);q["market_total"]=over.get("point") if over else None
                    if any(q[k] is not None for k in ("home_ml","away_ml","home_spread","market_total")):qs.append(q)
                self.external[key]=qs
        except Exception as e:self.errors.append(f"The Odds API: {type(e).__name__}: {e}")
    @staticmethod
    def _summary(g,quotes):
        base={"provider":g.provider or "primary","home_ml":g.home_ml,"away_ml":g.away_ml,"home_spread":g.home_spread,"market_total":g.market_total};qs=[dict(x) for x in quotes or []];bp=str(base["provider"]).lower();ex=next((q for q in qs if str(q.get("provider") or "").lower()==bp),None)
        if ex is None:qs.append(base)
        else:
            for k in ("home_ml","away_ml","home_spread","market_total"):
                if ex.get(k) is None:ex[k]=base.get(k)
        unique={}
        for q in qs:unique[str(q.get("provider") or "unknown").lower()]=q
        qs=list(unique.values());providers=[str(q.get("provider") or "unknown") for q in qs];sp=[float(q["home_spread"]) for q in qs if q.get("home_spread") is not None];to=[float(q["market_total"]) for q in qs if q.get("market_total") is not None];hm=[float(q["home_ml"]) for q in qs if q.get("home_ml") is not None];am=[float(q["away_ml"]) for q in qs if q.get("away_ml") is not None];ph=[]
        for q in qs:
            if q.get("home_ml") is not None and q.get("away_ml") is not None:
                try:_,x=no_vig(q["away_ml"],q["home_ml"]);ph.append(float(x))
                except Exception:pass
        return {"market_book_count":len(providers),"market_books":" | ".join(providers[:15]),"consensus_home_spread":float(np.median(sp)) if sp else np.nan,"consensus_total":float(np.median(to)) if to else np.nan,"spread_market_std":float(np.std(sp)) if len(sp)>1 else 0. if sp else np.nan,"total_market_std":float(np.std(to)) if len(to)>1 else 0. if to else np.nan,"best_home_ml":_best(hm),"best_away_ml":_best(am),"consensus_home_novig_probability":float(np.mean(ph)) if ph else np.nan,"market_consensus_quality":min(1.,len(providers)/4.)}
    def attach(self,games,pred):
        if pred.empty:return pred.copy(),{"coverage":0.,"multi_book_coverage":0.,"errors":[]}
        self._load_odds_api();by={str(g.game_id):g for g in games};quotes={}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fs=[ex.submit(self._espn,g) for g in games]
            for f in as_completed(fs):gid,x=f.result();quotes[gid]=[] if isinstance(x,Exception) else x
        out=pred.copy();covered=multi=0
        for i,r in out.iterrows():
            g=by[str(r.game_id)];q=list(quotes.get(str(r.game_id),[]));q.extend(self.external.get((canon_team(g.away_team),canon_team(g.home_team)),[]));z=self._summary(g,q)
            for k,v in z.items():out.at[i,k]=v
            covered+=int(z["market_book_count"]>0);multi+=int(z["market_book_count"]>=2)
        return out,{"coverage":covered/len(out),"multi_book_coverage":multi/len(out),"odds_api_configured":bool(self.odds_key),"odds_api_used":self.odds_api_used,"errors":self.errors[-20:]}
