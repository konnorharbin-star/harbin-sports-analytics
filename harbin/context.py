from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from .advanced import canon_team,canon_id

INJURY_URL="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_injuries/injuries_{season}.parquet"
TEAM_INFO_URL="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_team_info/cfb_team_info_{season}.parquet"
OPEN_METEO="https://api.open-meteo.com/v1/forecast"; GEO="https://geocoding-api.open-meteo.com/v1/search"
def _col(cols,*choices):
    low={str(c).lower():c for c in cols}
    for ch in choices:
        if isinstance(ch,str) and ch.lower() in low:return low[ch.lower()]
        words=ch if isinstance(ch,tuple) else (str(ch),)
        for c in cols:
            if all(w.lower() in str(c).lower() for w in words):return c
    return None
def _severity(x):
    s=str(x or "").lower();return 1. if "out" in s or "season" in s else .8 if "doubt" in s else .45 if "question" in s else .35 if "day-to-day" in s or "day to day" in s else .15 if "prob" in s else .25
def haversine_miles(a,b,c,d):
    if any(x is None or pd.isna(x) for x in (a,b,c,d)):return np.nan
    r=3958.7613;p1,p2=math.radians(float(a)),math.radians(float(c));dp=math.radians(float(c)-float(a));dl=math.radians(float(d)-float(b));x=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2;return 2*r*math.asin(math.sqrt(x))

class ContextStore:
    def __init__(self,season,schedule_frame=None,cache_dir="cache/context"):
        self.season=int(season);self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True);self.errors=[];self.sources=[];self.schedule=schedule_frame.copy() if isinstance(schedule_frame,pd.DataFrame) else pd.DataFrame();self.session=requests.Session();self.session.headers.update({"User-Agent":"HarbinSportsAnalytics/6.0"});self._geo_cache={};self._weather_cache={};self.injury_name={};self.injury_id={};self.team_by_id={};self.team_by_name={};self.game_rows={};self._index_schedule();self._load_injuries();self._load_team_info()
    def _download_parquet(self,url,path):
        if not path.exists():r=requests.get(url,timeout=35,headers={"User-Agent":"HarbinSportsAnalytics/6.0"});r.raise_for_status();path.write_bytes(r.content)
        return pd.read_parquet(path)
    def _index_schedule(self):
        if self.schedule.empty:return
        gc=_col(self.schedule.columns,"game_id","id")
        if gc:
            for _,r in self.schedule.iterrows():self.game_rows[canon_id(r[gc])]=r
    def _load_injuries(self):
        try:d=self._download_parquet(INJURY_URL.format(season=self.season),self.cache/f"injuries_{self.season}.parquet")
        except Exception as e:self.errors.append(f"injuries: {type(e).__name__}: {e}");return
        tc=_col(d.columns,"team","team_name","school");ic=_col(d.columns,"team_id",("team","id"));pc=_col(d.columns,"position",("position",));sc=_col(d.columns,"status","type","detail",("injury","status"))
        for _,r in d.iterrows():
            dest=None
            if ic and canon_id(r.get(ic)):dest=self.injury_id.setdefault(canon_id(r.get(ic)),{"injury_count":0,"injury_risk":0.,"qb_injury_risk":0.})
            if dest is None and tc:dest=self.injury_name.setdefault(canon_team(r.get(tc)),{"injury_count":0,"injury_risk":0.,"qb_injury_risk":0.})
            if dest is None:continue
            pos=str(r.get(pc,"")) if pc else "";sev=_severity(r.get(sc,"")) if sc else .25;dest["injury_count"]+=1;dest["injury_risk"]+=sev
            if "QB" in pos.upper() or "QUARTERBACK" in pos.upper():dest["qb_injury_risk"]=max(dest["qb_injury_risk"],sev)
        self.sources.append("SportsDataverse ESPN injuries")
    def _load_team_info(self):
        try:d=self._download_parquet(TEAM_INFO_URL.format(season=self.season),self.cache/f"cfb_team_info_{self.season}.parquet")
        except Exception as e:self.errors.append(f"team_info: {type(e).__name__}: {e}");return
        ic=_col(d.columns,"team_id","id");nc=_col(d.columns,"team","school","display_name","location");lat=_col(d.columns,"latitude",("venue","latitude"),"lat");lon=_col(d.columns,"longitude",("venue","longitude"),"lon");elev=_col(d.columns,"elevation",("venue","elevation"));city=_col(d.columns,"venue_city","city",("venue","city"));state=_col(d.columns,"venue_state","state",("venue","state"));vc=_col(d.columns,"venue_id",("venue","id"))
        for _,r in d.iterrows():
            def num(c):
                try:return float(r[c]) if c and pd.notna(r[c]) else np.nan
                except Exception:return np.nan
            meta={"latitude":num(lat),"longitude":num(lon),"elevation":num(elev),"city":str(r.get(city,"")) if city else "","state":str(r.get(state,"")) if state else "","venue_id":canon_id(r.get(vc)) if vc else ""}
            if ic and canon_id(r.get(ic)):self.team_by_id[canon_id(r.get(ic))]=meta
            if nc:self.team_by_name[canon_team(r.get(nc))]=meta
        if len(d):self.sources.append("SportsDataverse cfb_team_info")
    def _team_meta(self,team,tid=""):
        return dict(self.team_by_id.get(canon_id(tid),self.team_by_name.get(canon_team(team),{})))
    def _injury(self,team,tid=""):
        return dict(self.injury_id.get(canon_id(tid),self.injury_name.get(canon_team(team),{"injury_count":0,"injury_risk":0.,"qb_injury_risk":0.})))
    def _geocode(self,meta):
        if not meta:return meta
        if pd.notna(meta.get("latitude",np.nan)) and pd.notna(meta.get("longitude",np.nan)):return meta
        q=", ".join(x for x in (meta.get("city",""),meta.get("state","")) if x and x.lower()!="nan");
        if not q:return meta
        if q in self._geo_cache:return {**meta,**self._geo_cache[q]}
        try:
            r=self.session.get(GEO,params={"name":q,"count":1,"language":"en","format":"json"},timeout=8);r.raise_for_status();res=(r.json().get("results") or [None])[0]
            v={"latitude":float(res["latitude"]),"longitude":float(res["longitude"]),"elevation":float(res.get("elevation",np.nan))} if res else {};self._geo_cache[q]=v;return {**meta,**v}
        except Exception as e:self.errors.append(f"geocode {q}: {type(e).__name__}");self._geo_cache[q]={};return meta
    def _rest(self,tid,team,kick):
        if self.schedule.empty:return np.nan
        try:
            dates=pd.to_datetime(self.schedule["start_date"],utc=True,errors="coerce");b=self.schedule[dates<kick].copy();hid=_col(b.columns,"home_id");aid=_col(b.columns,"away_id")
            if hid and aid and canon_id(tid):mask=b[hid].map(canon_id).eq(canon_id(tid))|b[aid].map(canon_id).eq(canon_id(tid))
            else:mask=b.get("home_team",pd.Series(index=b.index,dtype=str)).map(canon_team).eq(canon_team(team))|b.get("away_team",pd.Series(index=b.index,dtype=str)).map(canon_team).eq(canon_team(team))
            b=b[mask];
            if "completed" in b:b=b[b.completed.astype(str).str.lower().isin({"true","1","t","yes"})]
            if b.empty:return np.nan
            return float((kick-pd.to_datetime(b.start_date,utc=True,errors="coerce").max()).total_seconds()/86400)
        except Exception:return np.nan
    def _weather(self,meta,kick):
        meta=self._geocode(meta);lat,lon=meta.get("latitude",np.nan),meta.get("longitude",np.nan)
        if pd.isna(lat) or pd.isna(lon) or pd.isna(kick):return {},meta
        day=kick.strftime("%Y-%m-%d");key=(round(float(lat),2),round(float(lon),2),day)
        if key in self._weather_cache:return self._weather_cache[key],meta
        try:
            r=self.session.get(OPEN_METEO,params={"latitude":lat,"longitude":lon,"hourly":"temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m","temperature_unit":"fahrenheit","wind_speed_unit":"mph","timezone":"UTC","start_date":day,"end_date":day},timeout=10);r.raise_for_status();jdata=r.json();h=jdata.get("hourly") or {};ts=pd.to_datetime(h.get("time",[]),utc=True,errors="coerce")
            if not len(ts):return {},meta
            j=int(np.argmin(np.abs((ts-kick).total_seconds())));out={"temperature_f":float(h.get("temperature_2m",[np.nan]*len(ts))[j]),"precip_probability":float(h.get("precipitation_probability",[np.nan]*len(ts))[j]),"wind_mph":float(h.get("wind_speed_10m",[np.nan]*len(ts))[j]),"wind_gust_mph":float(h.get("wind_gusts_10m",[np.nan]*len(ts))[j])};
            if pd.isna(meta.get("elevation",np.nan)) and jdata.get("elevation") is not None:meta["elevation"]=float(jdata["elevation"])*3.28084
            self._weather_cache[key]=out;return out,meta
        except Exception as e:self.errors.append(f"weather {key}: {type(e).__name__}");self._weather_cache[key]={};return {},meta
    def attach(self,pred):
        out=pred.copy();hits=weather_hits=injury_hits=travel_hits=0
        if out.empty:return out,{"sources":self.sources,"errors":self.errors,"coverage":0.,"weather_coverage":0.}
        for i,r in out.iterrows():
            sr=self.game_rows.get(canon_id(r.game_id));hid=canon_id(sr.get("home_id")) if sr is not None and "home_id" in sr else "";aid=canon_id(sr.get("away_id")) if sr is not None and "away_id" in sr else "";h=self._injury(r.home_team,hid);a=self._injury(r.away_team,aid);hm=self._geocode(self._team_meta(r.home_team,hid));am=self._geocode(self._team_meta(r.away_team,aid));injury_hits+=int(h["injury_count"]+a["injury_count"]>0);availability=min(1.,.10*(h["injury_risk"]+a["injury_risk"])+.35*max(h["qb_injury_risk"],a["qb_injury_risk"]));out.at[i,"home_injury_count"]=h["injury_count"];out.at[i,"away_injury_count"]=a["injury_count"];out.at[i,"home_qb_injury_risk"]=h["qb_injury_risk"];out.at[i,"away_qb_injury_risk"]=a["qb_injury_risk"];out.at[i,"availability_risk"]=availability
            kick=pd.to_datetime(r.date,utc=True,errors="coerce");hr=self._rest(hid,r.home_team,kick);ar=self._rest(aid,r.away_team,kick);out.at[i,"home_rest_days"]=hr;out.at[i,"away_rest_days"]=ar;out.at[i,"rest_edge_home"]=hr-ar if pd.notna(hr) and pd.notna(ar) else np.nan;travel=haversine_miles(am.get("latitude"),am.get("longitude"),hm.get("latitude"),hm.get("longitude"));out.at[i,"away_travel_miles"]=travel;travel_hits+=int(pd.notna(travel));w,hm=self._weather(hm,kick);weather_hits+=int(bool(w));elev=hm.get("elevation",np.nan);out.at[i,"venue_elevation_ft"]=elev;out.at[i,"altitude_change_away_ft"]=elev-am.get("elevation",np.nan) if pd.notna(elev) and pd.notna(am.get("elevation",np.nan)) else np.nan
            for k,v in w.items():out.at[i,k]=v
            wr=min(1.,max(0,(w.get("wind_mph",0)-15)/25)*.55+max(0,(w.get("precip_probability",0)-40)/60)*.3+max(0,(w.get("wind_gust_mph",0)-25)/35)*.15) if w else 0.;tr=min(.5,max(0,(float(travel)-800)/2200)) if pd.notna(travel) else 0.;rr=.35 if pd.notna(ar) and ar<6 else 0.;out.at[i,"weather_risk"]=wr;out.at[i,"travel_rest_risk"]=min(1.,tr+rr);out.at[i,"context_risk"]=min(1.,.62*availability+.23*wr+.15*min(1.,tr+rr));hits+=int(bool(hm or am or h["injury_count"] or a["injury_count"]))
        if weather_hits:self.sources.append("Open-Meteo forecast");
        if travel_hits:self.sources.append("Open-Meteo geocoding / team venue metadata")
        return out,{"sources":list(dict.fromkeys(self.sources)),"errors":self.errors[-30:],"coverage":hits/len(out),"weather_coverage":weather_hits/len(out),"injury_coverage":injury_hits/len(out),"travel_coverage":travel_hits/len(out)}
