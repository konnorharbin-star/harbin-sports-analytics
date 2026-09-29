from __future__ import annotations

import math,re
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ADV_TEAM_URL="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_adv_team/adv_team_{season}.csv"
RAW_BASE="https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-data/main/data"

def canon_team(v):
    s=re.sub(r"[^a-z0-9]","",str(v or "").lower()); aliases={"northcarolinastate":"ncstate","southernmethodist":"smu","texaschristian":"tcu","brighamyoung":"byu","centralflorida":"ucf","louisianastate":"lsu","alabamabirmingham":"uab","nevadalasvegas":"unlv","texaselpaso":"utep","texassanantonio":"utsa","floridainternational":"fiu","southernmississippi":"southernmiss","miamiflorida":"miami"}; return aliases.get(s,s)
def canon_id(v):
    try:
        if v is None or pd.isna(v): return ""
        return str(int(float(v)))
    except Exception: return ""
def _cached(url,path):
    if path.exists() and path.stat().st_size>100:return pd.read_csv(path,low_memory=False)
    path.parent.mkdir(parents=True,exist_ok=True); r=requests.get(url,timeout=40,headers={"User-Agent":"HarbinSportsAnalytics/7.0"}); r.raise_for_status(); path.write_bytes(r.content); return pd.read_csv(path,low_memory=False)
def _col(cols,*choices):
    low={str(c).lower():c for c in cols}
    for ch in choices:
        if isinstance(ch,str) and ch.lower() in low:return low[ch.lower()]
        words=ch if isinstance(ch,tuple) else (str(ch),)
        for c in cols:
            if all(w.lower() in str(c).lower() for w in words):return c
    return None

def _looks_numeric_id(series):
    if series is None:return False
    s=pd.to_numeric(series,errors="coerce")
    return bool(len(s) and s.notna().mean()>=.75 and s.dropna().between(1,999999).mean()>=.95)

class AdvancedFeatureStore:
    """Pregame as-of advanced efficiency plus roster priors, keyed by ESPN team ID/name.

    Dynamic rows are explicitly keyed using both numeric team IDs and canonical team names.
    This prevents silent zero-coverage when upstream columns such as ``pos_team`` are names
    rather than IDs, which was the main failure mode in the earlier implementation.
    """
    KEYWORDS=("epa","success","explos","line_yard","stuff","power","first_down","scoring","points_per","sec_per","plays_per","havoc","turnover","penalty","field_pos","starting_fp","yards_per")
    BLOCK=("game_id","season","week","team","pos_team","opponent","opp_team","home","away","score","points","win","result","id")
    def __init__(self,start_season,end_season,cache_dir="cache/advanced"):
        self.start_season=int(start_season); self.end_season=int(end_season); self.cache=Path(cache_dir); self.errors=[]; self.sources=[]; self.id_lookup={}; self.name_lookup={}; self.static_lookup={}; self.feature_names=[]; self.dynamic_names=[]; self.identity_diagnostics={}; self._build()
    def _load_adv(self):
        fs=[]
        for s in range(self.start_season,self.end_season+1):
            try:
                d=_cached(ADV_TEAM_URL.format(season=s),self.cache/f"adv_team_{s}.csv");
                if len(d):fs.append(d)
            except Exception as e:self.errors.append(f"adv_team {s}: {type(e).__name__}: {e}")
        if fs:self.sources.append("SportsDataverse espn_cfb_adv_team");return pd.concat(fs,ignore_index=True,sort=False)
        return pd.DataFrame()
    def _metric_frame(self,df):
        keep=[]
        for c in df.columns:
            lc=str(c).lower(); s=pd.to_numeric(df[c],errors="coerce")
            if s.notna().sum()<max(50,int(len(df)*.03)):continue
            if any(x==lc for x in self.BLOCK) or lc.endswith("_id") or lc in {"season","week"}:continue
            if any(k in lc for k in self.KEYWORDS):keep.append(c)
        keep=keep[:32]
        out=pd.DataFrame(index=df.index)
        for c in keep:
            name="adv_"+re.sub(r"[^a-z0-9]+","_",str(c).lower()).strip("_"); out[name]=pd.to_numeric(df[c],errors="coerce")
        return out
    def _build_asof(self,df):
        if df.empty:return
        sc=_col(df.columns,"season","year"); wc=_col(df.columns,"week")
        raw_id=_col(df.columns,"team_id",("team","id"),"pos_team_id")
        pos_team=_col(df.columns,"pos_team")
        nc=_col(df.columns,"team","team_name","school","pos_team")
        if raw_id is None and pos_team is not None and _looks_numeric_id(df[pos_team]):raw_id=pos_team
        if nc is not None and _looks_numeric_id(df[nc]):
            alt=_col(df.columns,"team_name","school","pos_team")
            if alt is not None and not _looks_numeric_id(df[alt]):nc=alt
        if sc is None or wc is None or (raw_id is None and nc is None):self.errors.append("adv_team: missing season/week/team identity keys");return
        vals=self._metric_frame(df)
        if vals.empty:self.errors.append("adv_team: no recognized numeric efficiency columns");return
        team_ids=df[raw_id].map(canon_id) if raw_id is not None else pd.Series("",index=df.index)
        team_names=df[nc].map(canon_team) if nc is not None else pd.Series("",index=df.index)
        if pos_team is not None and not _looks_numeric_id(df[pos_team]):
            pos_names=df[pos_team].map(canon_team); team_names=team_names.where(team_names.astype(bool),pos_names)
        self.identity_diagnostics={"id_column":str(raw_id) if raw_id is not None else None,"name_column":str(nc) if nc is not None else None,"id_coverage":float(team_ids.astype(bool).mean()),"name_coverage":float(team_names.astype(bool).mean())}
        work=pd.DataFrame({"season":pd.to_numeric(df[sc],errors="coerce"),"week":pd.to_numeric(df[wc],errors="coerce"),"team_id":team_ids,"team_name":team_names}); work=pd.concat([work,vals],axis=1).dropna(subset=["season","week"]); work.season=work.season.astype(int);work.week=work.week.astype(int); metrics=list(vals.columns)
        wr=work.groupby(["season","week","team_id","team_name"],as_index=False,dropna=False)[metrics].mean(numeric_only=True).sort_values(["season","week"])
        prior={}
        for season in sorted(wr.season.unique()):
            states={}; sub=wr[wr.season==season]
            for _,row in sub.iterrows():
                key=(str(row.team_id or ""),str(row.team_name or "")); st=states.setdefault(key,{m:[0.,0] for m in metrics}); carry=prior.get(key,{})
                pre={m:(st[m][0]/st[m][1] if st[m][1] else (.62*carry[m] if m in carry and pd.notna(carry[m]) else np.nan)) for m in metrics}
                if row.team_id:self.id_lookup[(int(season),int(row.week),str(row.team_id))]=pre
                if row.team_name:self.name_lookup[(int(season),int(row.week),str(row.team_name))]=pre
                for m in metrics:
                    v=row.get(m)
                    if pd.notna(v) and math.isfinite(float(v)):st[m][0]+=float(v);st[m][1]+=1
            for key,st in states.items():prior[key]={m:(v[0]/v[1] if v[1] else prior.get(key,{}).get(m,np.nan)) for m,v in st.items()}
        self.dynamic_names=metrics; self.feature_names=list(metrics)
    def _load_static(self):
        for label,fn in {"talent":"cfb_matchup_roster_talent.csv","returning":"cfb_matchup_returning_production.csv","continuity":"cfb_matchup_coach_continuity.csv"}.items():
            try:d=_cached(f"{RAW_BASE}/{fn}",self.cache/fn)
            except Exception as e:self.errors.append(f"{label}: {type(e).__name__}: {e}");continue
            sc=_col(d.columns,"season","year");tc=_col(d.columns,"team","school","team_name");
            if sc is None or tc is None:continue
            nums=[]
            for c in d.columns:
                lc=str(c).lower(); s=pd.to_numeric(d[c],errors="coerce")
                if c not in (sc,tc) and any(k in lc for k in ("talent","rtprod","return","cont","experience")) and s.notna().sum()>=max(5,int(len(d)*.1)):nums.append(c)
            for _,r in d.iterrows():
                try:key=(int(r[sc]),canon_team(r[tc]))
                except Exception:continue
                dest=self.static_lookup.setdefault(key,{})
                for c in nums:
                    v=pd.to_numeric(pd.Series([r[c]]),errors="coerce").iloc[0]
                    if pd.notna(v):dest["prior_"+re.sub(r"[^a-z0-9]+","_",str(c).lower()).strip("_")]=float(v)
            if nums:self.sources.append(f"SportsDataverse {fn}")
    def _build(self):self._build_asof(self._load_adv());self._load_static();self.feature_names=list(dict.fromkeys(self.feature_names+[k for v in self.static_lookup.values() for k in v]))
    def _team(self,season,week,team,team_id=""):
        out={}; cid=canon_id(team_id)
        if cid:out.update(self.id_lookup.get((int(season),int(week),cid),{}))
        by_name=self.name_lookup.get((int(season),int(week),canon_team(team)),{})
        for k,v in by_name.items():
            if k not in out or pd.isna(out[k]):out[k]=v
        out.update(self.static_lookup.get((int(season),canon_team(team)),{}));return out
    def enrich(self,frame):
        base_meta={"sources":list(dict.fromkeys(self.sources)),"errors":self.errors[-30:],"feature_count":len(self.feature_names),"dynamic_feature_count":len(self.dynamic_names),"identity":self.identity_diagnostics}
        if frame.empty:return frame.copy(),{"coverage":0.,"dynamic_coverage":0.,**base_meta}
        out=frame.copy();hits=dynamic_hits=0; names=set(self.feature_names)
        for i,r in out.iterrows():
            hv=self._team(int(r.season),int(r.week),str(r.home_team),r.get("home_id",""));av=self._team(int(r.season),int(r.week),str(r.away_team),r.get("away_id",""));hits+=int(bool(hv or av));dynamic_hits+=int(any(k in hv or k in av for k in self.dynamic_names))
            for n in names:
                h,a=hv.get(n,np.nan),av.get(n,np.nan);out.at[i,"home_"+n]=h;out.at[i,"away_"+n]=a
                if pd.notna(h) and pd.notna(a):out.at[i,"diff_"+n]=float(h)-float(a);out.at[i,"avg_"+n]=(float(h)+float(a))/2
                else:out.at[i,"diff_"+n]=np.nan;out.at[i,"avg_"+n]=np.nan
        return out,{"coverage":hits/len(out),"dynamic_coverage":dynamic_hits/len(out),**base_meta}
