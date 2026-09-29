from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .advanced import canon_team

INJURY_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_injuries/injuries_{season}.parquet"
TEAM_INFO_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_team_info/cfb_team_info_{season}.parquet"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


def _find_col(columns,*choices):
    low={str(c).lower():c for c in columns}
    for choice in choices:
        if isinstance(choice,str) and choice.lower() in low: return low[choice.lower()]
        words=choice if isinstance(choice,tuple) else (str(choice),)
        for c in columns:
            lc=str(c).lower()
            if all(w.lower() in lc for w in words): return c
    return None


def _severity(text):
    s=str(text or "").lower()
    if "out" in s or "season" in s: return 1.0
    if "doubt" in s: return .8
    if "question" in s: return .45
    if "day-to-day" in s or "day to day" in s: return .35
    if "prob" in s: return .15
    return .25


def haversine_miles(lat1,lon1,lat2,lon2):
    vals=(lat1,lon1,lat2,lon2)
    if any(v is None or pd.isna(v) for v in vals): return np.nan
    r=3958.7613; p1,p2=math.radians(float(lat1)),math.radians(float(lat2)); dp=math.radians(float(lat2)-float(lat1)); dl=math.radians(float(lon2)-float(lon1)); a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2; return 2*r*math.asin(math.sqrt(a))


class ContextStore:
    """Current-only injury/QB, rest, travel, altitude, and weather risk controls."""

    def __init__(self,season,schedule_frame=None,cache_dir="cache/context"):
        self.season=int(season); self.cache=Path(cache_dir); self.cache.mkdir(parents=True,exist_ok=True); self.errors=[]; self.sources=[]; self.schedule=schedule_frame.copy() if isinstance(schedule_frame,pd.DataFrame) else pd.DataFrame(); self.injuries=self._load_injuries(); self.team_info=self._load_team_info(); self.team_meta=self._team_meta(); self.session=requests.Session(); self.session.headers.update({"User-Agent":"HarbinSportsAnalytics/4.0"}); self._weather_cache={}

    def _load_injuries(self):
        p=self.cache/f"injuries_{self.season}.parquet"
        try:
            if not p.exists():
                r=requests.get(INJURY_URL.format(season=self.season),timeout=25,headers={"User-Agent":"HarbinSportsAnalytics/4.0"}); r.raise_for_status(); p.write_bytes(r.content)
            df=pd.read_parquet(p)
        except Exception as exc:
            self.errors.append(f"injuries: {type(exc).__name__}: {exc}"); return {}
        tc=_find_col(df.columns,"team",("team","name"),"school"); pc=_find_col(df.columns,"position",("position",)); sc=_find_col(df.columns,"status","type","detail",("injury","status"))
        if tc is None: return {}
        out={}
        for _,row in df.iterrows():
            key=canon_team(row[tc]); pos=str(row.get(pc,"")).upper() if pc else ""; status=str(row.get(sc,"")) if sc else ""; sev=_severity(status); d=out.setdefault(key,{"injury_count":0,"injury_risk":0.0,"qb_injury_risk":0.0}); d["injury_count"]+=1; d["injury_risk"]+=sev
            if pos=="QB" or "QUARTERBACK" in pos: d["qb_injury_risk"]=max(d["qb_injury_risk"],sev)
        self.sources.append("SportsDataverse ESPN injuries"); return out

    def _load_team_info(self):
        p=self.cache/f"cfb_team_info_{self.season}.parquet"
        try:
            if not p.exists():
                r=requests.get(TEAM_INFO_URL.format(season=self.season),timeout=25,headers={"User-Agent":"HarbinSportsAnalytics/4.0"}); r.raise_for_status(); p.write_bytes(r.content)
            df=pd.read_parquet(p)
            if len(df): self.sources.append("SportsDataverse cfb_team_info")
            return df
        except Exception as exc:
            self.errors.append(f"team_info: {type(exc).__name__}: {exc}"); return pd.DataFrame()

    def _team_meta(self):
        df=self.team_info
        if df.empty: return {}
        tc=_find_col(df.columns,"team","school",("team","name"),"location"); lat=_find_col(df.columns,"latitude",("venue","latitude"),("location","latitude")); lon=_find_col(df.columns,"longitude",("venue","longitude"),("location","longitude")); elev=_find_col(df.columns,"elevation",("venue","elevation"))
        if tc is None: return {}
        out={}
        for _,row in df.iterrows():
            def num(c):
                try: return float(row[c]) if c and pd.notna(row[c]) else np.nan
                except Exception: return np.nan
            out[canon_team(row[tc])] = {"latitude":num(lat),"longitude":num(lon),"elevation":num(elev)}
        return out

    def team_context(self,team): return dict(self.injuries.get(canon_team(team),{"injury_count":0,"injury_risk":0.0,"qb_injury_risk":0.0}))

    def _rest_days(self,team,game_date):
        if self.schedule.empty: return np.nan
        try:
            dates=pd.to_datetime(self.schedule["start_date"],utc=True,errors="coerce"); before=self.schedule[dates<game_date].copy(); home=before.get("home_team",pd.Series(index=before.index,dtype=str)).map(canon_team)==canon_team(team); away=before.get("away_team",pd.Series(index=before.index,dtype=str)).map(canon_team)==canon_team(team); before=before[home|away]
            if "completed" in before.columns: before=before[before["completed"].astype(str).str.lower().isin({"true","1","t","yes"})]
            if before.empty: return np.nan
            last=pd.to_datetime(before["start_date"],utc=True,errors="coerce").max(); return float((game_date-last).total_seconds()/86400)
        except Exception: return np.nan

    def _weather(self,lat,lon,kick):
        if pd.isna(lat) or pd.isna(lon) or pd.isna(kick): return {}
        day=kick.strftime("%Y-%m-%d"); key=(round(float(lat),2),round(float(lon),2),day)
        if key in self._weather_cache: return self._weather_cache[key]
        try:
            r=self.session.get(OPEN_METEO,params={"latitude":float(lat),"longitude":float(lon),"hourly":"temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m","temperature_unit":"fahrenheit","wind_speed_unit":"mph","timezone":"UTC","start_date":day,"end_date":day},timeout=10); r.raise_for_status(); h=r.json().get("hourly") or {}; times=pd.to_datetime(h.get("time",[]),utc=True,errors="coerce")
            if len(times)==0: return {}
            j=int(np.argmin(np.abs((times-kick).total_seconds()))); out={"temperature_f":float(h.get("temperature_2m",[np.nan]*len(times))[j]),"precip_probability":float(h.get("precipitation_probability",[np.nan]*len(times))[j]),"wind_mph":float(h.get("wind_speed_10m",[np.nan]*len(times))[j]),"wind_gust_mph":float(h.get("wind_gusts_10m",[np.nan]*len(times))[j])}; self._weather_cache[key]=out; return out
        except Exception as exc:
            self.errors.append(f"weather {key}: {type(exc).__name__}: {exc}"); self._weather_cache[key]={}; return {}

    def attach(self,pred):
        out=pred.copy()
        if out.empty: return out,{"sources":self.sources,"errors":self.errors,"coverage":0.0,"weather_coverage":0.0}
        hits=weather_hits=0
        for i,row in out.iterrows():
            h=self.team_context(str(row.home_team)); a=self.team_context(str(row.away_team)); hm=self.team_meta.get(canon_team(row.home_team),{}); am=self.team_meta.get(canon_team(row.away_team),{}); availability=min(1.0,.10*(h["injury_risk"]+a["injury_risk"])+.35*max(h["qb_injury_risk"],a["qb_injury_risk"])); out.at[i,"home_injury_count"]=h["injury_count"]; out.at[i,"away_injury_count"]=a["injury_count"]; out.at[i,"home_injury_risk"]=h["injury_risk"]; out.at[i,"away_injury_risk"]=a["injury_risk"]; out.at[i,"home_qb_injury_risk"]=h["qb_injury_risk"]; out.at[i,"away_qb_injury_risk"]=a["qb_injury_risk"]; out.at[i,"availability_risk"]=availability
            kick=pd.to_datetime(row.date,utc=True,errors="coerce"); hr=self._rest_days(str(row.home_team),kick); ar=self._rest_days(str(row.away_team),kick); out.at[i,"home_rest_days"]=hr; out.at[i,"away_rest_days"]=ar; out.at[i,"rest_edge_home"]=(hr-ar) if pd.notna(hr) and pd.notna(ar) else np.nan; travel=haversine_miles(am.get("latitude"),am.get("longitude"),hm.get("latitude"),hm.get("longitude")); out.at[i,"away_travel_miles"]=travel; elev=hm.get("elevation",np.nan); out.at[i,"venue_elevation_ft"]=elev; out.at[i,"altitude_change_away_ft"]=(elev-am.get("elevation",np.nan)) if pd.notna(elev) and pd.notna(am.get("elevation",np.nan)) else np.nan
            w=self._weather(hm.get("latitude"),hm.get("longitude"),kick)
            for k,v in w.items(): out.at[i,k]=v
            if w: weather_hits+=1
            wr=min(1.0,max(0,(w.get("wind_mph",0)-15)/25)*.55+max(0,(w.get("precip_probability",0)-40)/60)*.3+max(0,(w.get("wind_gust_mph",0)-25)/35)*.15) if w else 0.0; tr=min(.5,max(0,(float(travel)-800)/2200)) if pd.notna(travel) else 0.0; rr=.35 if pd.notna(ar) and ar<6 else 0.0; out.at[i,"weather_risk"]=wr; out.at[i,"travel_rest_risk"]=min(1.0,tr+rr); out.at[i,"context_risk"]=min(1.0,.62*availability+.23*wr+.15*min(1.0,tr+rr))
            if h["injury_count"] or a["injury_count"] or hm or am: hits+=1
        if weather_hits: self.sources.append("Open-Meteo forecast")
        return out,{"sources":list(dict.fromkeys(self.sources)),"errors":self.errors[-25:],"coverage":hits/len(out),"weather_coverage":weather_hits/len(out)}
