from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .advanced import canon_team

INJURY_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_injuries/injuries_{season}.parquet"
TEAM_INFO_CANDIDATES = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_team_info/cfb_team_info_{season}.csv",
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_team_info/team_info_{season}.csv",
)


def _find_col(columns, words):
    for c in columns:
        lc = str(c).lower()
        if all(w in lc for w in words): return c
    return None


def _severity(text: str) -> float:
    s = str(text or "").lower()
    if "out" in s or "season" in s: return 1.0
    if "doubt" in s: return .8
    if "question" in s: return .45
    if "day-to-day" in s or "day to day" in s: return .35
    if "prob" in s: return .15
    return .25


class ContextStore:
    """Current availability/weather context.

    Context is deliberately kept OUT of the trained fair-score model unless a
    historical, timestamp-correct source exists. It is used for risk/stake
    gating so current injury/weather information cannot create backtest leakage.
    """

    def __init__(self, season: int, cache_dir="cache/context"):
        self.season = int(season); self.cache = Path(cache_dir)
        self.errors: list[str] = []; self.sources: list[str] = []
        self.injuries = self._load_injuries()
        self.team_info = self._load_team_info()

    def _load_injuries(self) -> dict[str, dict]:
        path = self.cache / f"injuries_{self.season}.parquet"; path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not path.exists():
                r = requests.get(INJURY_URL.format(season=self.season), timeout=25, headers={"User-Agent":"HarbinSportsAnalytics/4.0"}); r.raise_for_status(); path.write_bytes(r.content)
            df = pd.read_parquet(path)
        except Exception as exc:
            self.errors.append(f"injuries: {type(exc).__name__}: {exc}"); return {}
        tc = _find_col(df.columns, ("team",))
        pc = _find_col(df.columns, ("position",))
        sc = next((c for c in df.columns if any(x in str(c).lower() for x in ("status","type","detail"))), None)
        if not tc: return {}
        out = {}
        for _, r in df.iterrows():
            key = canon_team(r[tc]); pos = str(r.get(pc, "")).upper() if pc else ""; status = str(r.get(sc, "")) if sc else ""
            sev = _severity(status); d = out.setdefault(key, {"injury_count":0,"injury_risk":0.0,"qb_injury_risk":0.0})
            d["injury_count"] += 1; d["injury_risk"] += sev
            if pos == "QB" or "quarterback" in pos.lower(): d["qb_injury_risk"] = max(d["qb_injury_risk"], sev)
        self.sources.append("SportsDataverse ESPN injuries")
        return out

    def _load_team_info(self) -> pd.DataFrame:
        self.cache.mkdir(parents=True, exist_ok=True)
        for tmpl in TEAM_INFO_CANDIDATES:
            url = tmpl.format(season=self.season); name = url.rsplit("/",1)[-1]; path=self.cache/name
            try:
                if path.exists() and path.stat().st_size>100: df=pd.read_csv(path,low_memory=False)
                else:
                    r=requests.get(url,timeout=20,headers={"User-Agent":"HarbinSportsAnalytics/4.0"}); r.raise_for_status(); path.write_bytes(r.content); df=pd.read_csv(path,low_memory=False)
                if len(df): self.sources.append("SportsDataverse team info"); return df
            except Exception: pass
        return pd.DataFrame()

    def team_context(self, team: str) -> dict:
        return dict(self.injuries.get(canon_team(team), {"injury_count":0,"injury_risk":0.0,"qb_injury_risk":0.0}))

    def attach(self, pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        out=pred.copy()
        if out.empty: return out,{"sources":self.sources,"errors":self.errors,"coverage":0.0}
        hits=0
        for i,r in out.iterrows():
            h=self.team_context(str(r.home_team)); a=self.team_context(str(r.away_team))
            out.at[i,"home_injury_count"]=h["injury_count"]; out.at[i,"away_injury_count"]=a["injury_count"]
            out.at[i,"home_injury_risk"]=h["injury_risk"]; out.at[i,"away_injury_risk"]=a["injury_risk"]
            out.at[i,"home_qb_injury_risk"]=h["qb_injury_risk"]; out.at[i,"away_qb_injury_risk"]=a["qb_injury_risk"]
            risk=min(1.0,.10*(h["injury_risk"]+a["injury_risk"])+.35*max(h["qb_injury_risk"],a["qb_injury_risk"]))
            out.at[i,"availability_risk"]=risk
            if h["injury_count"] or a["injury_count"]: hits+=1
        return out,{"sources":self.sources,"errors":self.errors,"coverage":hits/len(out)}


def haversine_miles(lat1, lon1, lat2, lon2):
    vals=(lat1,lon1,lat2,lon2)
    if any(v is None or (isinstance(v,float) and np.isnan(v)) for v in vals): return np.nan
    r=3958.7613; p1,p2=math.radians(float(lat1)),math.radians(float(lat2)); dp=math.radians(float(lat2)-float(lat1)); dl=math.radians(float(lon2)-float(lon1))
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))
