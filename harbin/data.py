from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

SCHEDULE_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_{season}.csv"
ARCHIVE_ODDS_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/csv/cfb_line_odds.csv.gz"

ESPN_SCOREBOARD_URLS = (
    "https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
)
ESPN_CORE_ODDS = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/{game_id}/competitions/{game_id}/odds"


@dataclass
class Game:
    game_id: str; season: int; week: int; date: str
    away_id: str; away_team: str; home_id: str; home_team: str
    away_score: float | None; home_score: float | None; completed: bool; neutral_site: bool
    provider: str | None = None; away_ml: float | None = None; home_ml: float | None = None
    home_spread: float | None = None; market_total: float | None = None


def _num(v):
    try: return None if pd.isna(v) else float(v)
    except Exception: return None

def _bool(v):
    if isinstance(v,bool): return v
    return str(v).strip().lower() in {"1","true","t","yes"}

def _canon(name:str)->str:
    s=re.sub(r"[^a-z0-9]","",str(name).lower())
    aliases={"northcarolinastate":"ncstate","southernmethodist":"smu","texaschristian":"tcu","brighamyoung":"byu","centralflorida":"ucf","louisianastate":"lsu","alabamabirmingham":"uab","nevadalasvegas":"unlv","texaselpaso":"utep","texassanantonio":"utsa","floridainternational":"fiu","southernmississippi":"southernmiss"}
    return aliases.get(s,s)

def _moneyline(side:dict|None):
    side=side or {}; return _num(side.get("moneyLine",side.get("moneyline")))

def _provider_name(o:dict)->str|None:
    p=o.get("provider")
    if isinstance(p,dict): return str(p.get("name") or p.get("id") or "") or None
    return str(p) if p else None

def parse_espn_odds(odds:dict,home_names:set[str],away_names:set[str])->dict:
    out={"provider":_provider_name(odds),"home_ml":_moneyline(odds.get("homeTeamOdds")),"away_ml":_moneyline(odds.get("awayTeamOdds")),"market_total":_num(odds.get("overUnder",odds.get("overunder"))),"home_spread":None}
    details=str(odds.get("details") or "").strip(); m=re.match(r"^(.*?)\s+([+-]?\d+(?:\.\d+)?)$",details)
    if m:
        fav=_canon(m.group(1)); line=_num(m.group(2))
        if line is not None:
            if fav in home_names: out["home_spread"]=float(line)
            elif fav in away_names: out["home_spread"]=-float(line)
    if out["home_spread"] is None:
        spread=_num(odds.get("spread"))
        if spread is not None:
            hs,aws=odds.get("homeTeamOdds") or {},odds.get("awayTeamOdds") or {}
            if hs.get("favorite") is True: out["home_spread"]=-abs(spread)
            elif aws.get("favorite") is True: out["home_spread"]=abs(spread)
            else: out["home_spread"]=float(spread)
    return out


class SportsDataVerseClient:
    BOOK_PRIORITY=("draftkings","fanduel","espn","fanatics","circa","pinnacle","bovada")
    def __init__(self):
        self._seasons={}; self._archive_odds=None; self.odds_columns=[]; self.odds_source="none"; self.odds_errors=[]; self.odds_rows_matched=0; self.odds_games_attached=0
        self.session=requests.Session(); self.session.headers.update({"User-Agent":"Mozilla/5.0 (compatible; HarbinSportsAnalytics/3.0)","Accept":"application/json,text/plain,*/*"})
    def season_frame(self,season:int)->pd.DataFrame:
        if season not in self._seasons:
            df=pd.read_csv(SCHEDULE_URL.format(season=season),low_memory=False)
            if {"home_division","away_division"}.issubset(df.columns):
                fbs=df["home_division"].astype(str).str.lower().eq("fbs") & df["away_division"].astype(str).str.lower().eq("fbs")
                if fbs.any(): df=df.loc[fbs].copy()
            self._seasons[season]=df
        return self._seasons[season].copy()
    def detect(self):
        now=pd.Timestamp.now(tz="UTC"); season=now.year if now.month>=7 else now.year-1
        try: df=self.season_frame(season)
        except Exception: season-=1; df=self.season_frame(season)
        regular=df[df["season_type"].astype(str).str.lower().isin({"regular","2"})].copy(); regular["_date"]=pd.to_datetime(regular["start_date"],utc=True,errors="coerce")
        completed=regular["completed"].map(_bool); upcoming=regular[(~completed)&(regular["_date"]>=now-pd.Timedelta(days=1))]
        if len(upcoming): return season,int(upcoming.sort_values("_date").iloc[0]["week"])
        done=regular[completed]; return season,int(done["week"].max()) if len(done) else 1
    def _row_game(self,r):
        def sid(v,fallback):
            try:return str(int(v)) if not pd.isna(v) else str(fallback)
            except Exception:return str(fallback)
        return Game(str(r.get("game_id","")),int(r.get("season",0)),int(r.get("week",0)),str(r.get("start_date","")),sid(r.get("away_id"),r.get("away_team","away")),str(r.get("away_team","Away")),sid(r.get("home_id"),r.get("home_team","home")),str(r.get("home_team","Home")),_num(r.get("away_points")),_num(r.get("home_points")),_bool(r.get("completed",False)),_bool(r.get("neutral_site",False)))
    def history(self,start,target_season,target_week):
        games=[]
        for season in range(start,target_season+1):
            try: df=self.season_frame(season)
            except Exception: continue
            df=df[df["completed"].map(_bool)].copy()
            if season==target_season: df=df[df["week"].astype(int)<int(target_week)]
            for _,r in df.iterrows():
                g=self._row_game(r)
                if g.home_score is not None and g.away_score is not None: games.append(g)
        return sorted(games,key=lambda g:(g.season,g.date,g.game_id))
    def week(self,season,week):
        df=self.season_frame(season); reg=df["season_type"].astype(str).str.lower().isin({"regular","2"}); rows=df[(df["week"].astype(int)==int(week))&reg]
        games=[self._row_game(r) for _,r in rows.iterrows()]; games=self._attach_live_espn(games,season,week)
        if season<datetime.now().year or self.odds_games_attached==0: games=self._attach_archive_odds(games)
        self.odds_games_attached=sum(any(v is not None for v in (g.home_ml,g.away_ml,g.home_spread,g.market_total)) for g in games); return games
    def _event_names(self,comp):
        home_names,away_names=set(),set()
        for c in comp.get("competitors") or []:
            team=c.get("team") or {}; names={_canon(team.get(k) or "") for k in ("abbreviation","displayName","shortDisplayName","name","location")}; names.discard("")
            if c.get("homeAway")=="home": home_names|=names
            elif c.get("homeAway")=="away": away_names|=names
        return home_names,away_names
    def _apply_espn_event(self,g,event):
        comps=event.get("competitions") or []
        if not comps:return False
        comp=comps[0]; odds_list=comp.get("odds") or []
        if not odds_list:return False
        home_names,away_names=self._event_names(comp); home_names=home_names or {_canon(g.home_team)}; away_names=away_names or {_canon(g.away_team)}
        normalized=[parse_espn_odds(o,home_names,away_names) for o in odds_list]
        def rank(x):
            p=str(x.get("provider") or "").lower(); return next((i for i,n in enumerate(self.BOOK_PRIORITY) if n in p),len(self.BOOK_PRIORITY))
        x=sorted(normalized,key=rank)[0]; g.provider=x["provider"] or "ESPN"; g.home_ml=x["home_ml"]; g.away_ml=x["away_ml"]; g.home_spread=x["home_spread"]; g.market_total=x["market_total"]
        return any(v is not None for v in (g.home_ml,g.away_ml,g.home_spread,g.market_total))
    def _attach_live_espn(self,games,season,week):
        if not games:return games
        by_id={str(g.game_id):g for g in games}; attached=0
        for url in ESPN_SCOREBOARD_URLS:
            try:
                r=self.session.get(url,params={"limit":500,"groups":80,"seasontype":2,"week":int(week)},timeout=12); r.raise_for_status()
                for e in r.json().get("events") or []:
                    g=by_id.get(str(e.get("id") or ""))
                    if g and self._apply_espn_event(g,e): attached+=1
                if attached:
                    self.odds_source=f"ESPN live ({url.split('/')[2]})"; self.odds_games_attached=attached; break
            except Exception as exc: self.odds_errors.append(f"{url.split('/')[2]}: {type(exc).__name__}: {exc}")
        needs_core=[g for g in games if g.home_ml is None or g.away_ml is None or g.home_spread is None or g.market_total is None]
        if not needs_core:return games
        def load_core(g):
            url=ESPN_CORE_ODDS.format(game_id=g.game_id)
            try:
                r=self.session.get(url,params={"limit":50},timeout=8); r.raise_for_status(); items=r.json().get("items") or []; resolved=[]
                for item in items:
                    if isinstance(item,dict) and item.get("$ref"):
                        try:
                            rr=self.session.get(str(item["$ref"]).replace("http://","https://"),timeout=6); rr.raise_for_status(); item=rr.json()
                        except Exception: continue
                    if isinstance(item,dict): resolved.append(item)
                if not resolved:return g.game_id,None
                vals=[parse_espn_odds(o,{_canon(g.home_team)},{_canon(g.away_team)}) for o in resolved]; vals.sort(key=lambda x:next((i for i,n in enumerate(self.BOOK_PRIORITY) if n in str(x.get("provider") or "").lower()),len(self.BOOK_PRIORITY))); return g.game_id,vals[0]
            except Exception as exc:return g.game_id,exc
        core_supplemented=0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures=[ex.submit(load_core,g) for g in needs_core]
            for f in as_completed(futures):
                gid,val=f.result()
                if isinstance(val,Exception) or not val:continue
                g=by_id[gid]
                if g.provider is None and val["provider"]: g.provider=val["provider"]
                if g.home_ml is None:g.home_ml=val["home_ml"]
                if g.away_ml is None:g.away_ml=val["away_ml"]
                if g.home_spread is None:g.home_spread=val["home_spread"]
                if g.market_total is None:g.market_total=val["market_total"]
                if any(v is not None for v in (val["home_ml"],val["away_ml"],val["home_spread"],val["market_total"])):core_supplemented+=1
        if core_supplemented:self.odds_source="ESPN Core live" if self.odds_source=="none" else self.odds_source+" + ESPN Core supplement"
        self.odds_games_attached=sum(any(v is not None for v in (g.home_ml,g.away_ml,g.home_spread,g.market_total)) for g in games); return games
    def _load_archive_odds(self):
        if self._archive_odds is None:
            try:self._archive_odds=pd.read_csv(ARCHIVE_ODDS_URL,compression="gzip",low_memory=False)
            except Exception as exc:self.odds_errors.append(f"SportsDataVerse archive: {type(exc).__name__}: {exc}"); self._archive_odds=pd.DataFrame()
            self.odds_columns=list(self._archive_odds.columns)
        return self._archive_odds
    def _book_rank(self,book):
        s=str(book).lower(); return next((i for i,n in enumerate(self.BOOK_PRIORITY) if n in s),len(self.BOOK_PRIORITY))
    def _attach_archive_odds(self,games):
        odds=self._load_archive_odds()
        if odds.empty or "game_id" not in odds.columns:return games
        o=odds[odds["game_id"].astype(str).isin({str(g.game_id) for g in games})].copy(); self.odds_rows_matched=len(o)
        if o.empty:return games
        if not {"market_type","abbr","lines","odds","book"}.issubset(o.columns):self.odds_errors.append("SportsDataVerse archive schema missing expected long-form columns"); return games
        o["_market"]=o["market_type"].astype(str).str.lower().str.replace("-","_",regex=False).str.replace(" ","_",regex=False); o["_book"]=o["book"].astype(str); by_gid={str(k):v.copy() for k,v in o.groupby(o["game_id"].astype(str),sort=False)}; attached=0
        for g in games:
            rows=by_gid.get(str(g.game_id))
            if rows is None or rows.empty:continue
            candidates=[]
            for book,br in rows.groupby("_book",dropna=False):
                markets=set(br["_market"].tolist()); completeness=sum(any(token in m for m in markets) for token in ("spread","total","money")); candidates.append((-completeness,self._book_rank(book),str(book)))
            candidates.sort(); chosen_book=candidates[0][2]; br=rows[rows["_book"].astype(str)==chosen_book].copy(); br["_side"]=br["abbr"].map(_canon); home_key,away_key=_canon(g.home_team),_canon(g.away_team)
            def side_value(market_token,team_key,value_col):
                z=br[br["_market"].str.contains(market_token,na=False)]; exact=z[z["_side"]==team_key]; return _num(exact.iloc[-1][value_col]) if len(exact) else None
            if g.home_ml is None:g.home_ml=side_value("money",home_key,"odds")
            if g.away_ml is None:g.away_ml=side_value("money",away_key,"odds")
            if g.home_spread is None:
                g.home_spread=side_value("spread",home_key,"lines"); away_spread=side_value("spread",away_key,"lines") if g.home_spread is None else None
                if away_spread is not None:g.home_spread=-away_spread
            if g.market_total is None:
                totals=br[br["_market"].str.contains("total",na=False)]
                if len(totals):
                    over=totals[totals["_side"].str.contains("over",na=False)]; src=over.iloc[-1] if len(over) else totals.iloc[-1]; g.market_total=_num(src["lines"])
            if g.provider is None:g.provider=chosen_book
            if any(v is not None for v in (g.home_ml,g.away_ml,g.home_spread,g.market_total)):attached+=1
        if attached and self.odds_source=="none":self.odds_source="SportsDataVerse archive"
        return games
