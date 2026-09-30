from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .advanced import canon_id, canon_team

INJURY_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_injuries/injuries_{season}.parquet"
ROSTER_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_rosters/cfb_rosters_{season}.parquet"
TEAM_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_teams/cfb_teams_{season}.parquet"
LEGACY_TEAM_INFO_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_team_info/cfb_team_info_{season}.parquet"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
GEO = "https://geocoding-api.open-meteo.com/v1/search"


def _col(cols, *choices):
    low = {str(c).lower(): c for c in cols}
    for ch in choices:
        if isinstance(ch, str) and ch.lower() in low:
            return low[ch.lower()]
        words = ch if isinstance(ch, tuple) else (str(ch),)
        for c in cols:
            if all(w.lower() in str(c).lower() for w in words):
                return c
    return None


def _utc(value=None):
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    x = pd.Timestamp(value)
    if x.tzinfo is None:
        return x.tz_localize("UTC")
    return x.tz_convert("UTC")


def _finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _boolish(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "indoor", "closed"}


def _severity(value):
    """Conservative current-availability severity from ESPN-style status text."""
    s = str(value or "").lower()
    if any(x in s for x in ("active", "available", "cleared", "healthy", "will play", "returned")):
        return 0.0
    if any(x in s for x in ("season-ending", "season ending", "out for season", "injured reserve")):
        return 1.0
    if "out" in s or "suspended" in s:
        return 1.0
    if "doubt" in s:
        return 0.80
    if "question" in s:
        return 0.45
    if "day-to-day" in s or "day to day" in s:
        return 0.30
    if "prob" in s:
        return 0.10
    if any(x in s for x in ("injur", "limited", "game-time", "game time")):
        return 0.25
    return 0.0


def _season_ending(value):
    s = str(value or "").lower()
    return any(x in s for x in ("season-ending", "season ending", "out for season", "injured reserve"))


def normalize_injury_rows(df: pd.DataFrame, as_of=None, target_week=None) -> pd.DataFrame:
    """Return the latest point-in-time injury state per athlete.

    Rows reported after ``as_of`` or after ``target_week`` are excluded. Multiple reports
    for one athlete collapse to the latest admissible state, so a later ACTIVE/CLEARED
    report supersedes an older OUT report rather than being double-counted.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    as_of = _utc(as_of)
    out = df.copy()
    tc = _col(out.columns, "team", "team_name", "school", "display_name")
    tic = _col(out.columns, "team_id", ("team", "id"))
    aid = _col(out.columns, "athlete_id", "player_id", "id")
    an = _col(out.columns, "athlete_name", "player_name", "name")
    pc = _col(out.columns, "position", "position_abbreviation", "position_name")
    sc = _col(out.columns, "status", "status_name", "status_type")
    typ = _col(out.columns, "type", "injury_type")
    short = _col(out.columns, "short_comment", "short_detail")
    longc = _col(out.columns, "long_comment", "detail", "description")
    dc = _col(out.columns, "date", "updated_at", "last_update", "timestamp")
    wc = _col(out.columns, "week")

    out["_team_id"] = out[tic].map(canon_id) if tic else ""
    out["_team_name"] = out[tc].map(canon_team) if tc else ""
    out["_athlete_id"] = out[aid].map(canon_id) if aid else ""
    out["_athlete_name"] = out[an].map(canon_team) if an else ""
    out["_position"] = out[pc].fillna("").astype(str) if pc else ""
    out["_week"] = pd.to_numeric(out[wc], errors="coerce") if wc else np.nan
    out["_report_ts"] = pd.to_datetime(out[dc], utc=True, errors="coerce") if dc else pd.NaT

    if dc:
        out = out[out["_report_ts"].isna() | (out["_report_ts"] <= as_of)]
    if wc and target_week is not None:
        out = out[out["_week"].isna() | (out["_week"] <= int(target_week))]
    if out.empty:
        return out

    text = pd.Series("", index=out.index, dtype=object)
    for c in (sc, typ, short, longc):
        if c:
            text = text.str.cat(out[c].fillna("").astype(str), sep=" ")
    out["_status_text"] = text.str.strip()
    out["_severity"] = out["_status_text"].map(_severity).astype(float)
    out["_season_ending"] = out["_status_text"].map(_season_ending)

    age = (as_of - out["_report_ts"]).dt.total_seconds() / 86400.0
    age = age.where(age >= 0)
    decay = np.power(0.5, age.fillna(0).clip(lower=0) / 28.0)
    out.loc[~out["_season_ending"], "_severity"] *= decay[~out["_season_ending"]]
    out.loc[(~out["_season_ending"]) & age.gt(70), "_severity"] = 0.0
    out["_report_age_days"] = age

    team_key = out["_team_id"].where(out["_team_id"].ne(""), out["_team_name"])
    athlete_key = out["_athlete_id"].where(out["_athlete_id"].ne(""), out["_athlete_name"])
    athlete_key = athlete_key.where(athlete_key.ne(""), "row:" + out.index.astype(str))
    out["_team_key"] = team_key
    out["_athlete_key"] = athlete_key
    out = out.sort_values(["_team_key", "_athlete_key", "_week", "_report_ts"], na_position="first")
    return out.drop_duplicates(["_team_key", "_athlete_key"], keep="last").reset_index(drop=True)


def summarize_injury_rows(df: pd.DataFrame) -> dict:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"injury_count": 0, "injury_risk": 0.0, "qb_injury_risk": 0.0, "injury_report_age_days": np.nan, "injury_reports": 0}
    sev = pd.to_numeric(df.get("_severity", 0), errors="coerce").fillna(0.0)
    active = sev >= 0.05
    pos = df.get("_position", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    qb = pos.str.contains(r"(?:^|\b)QB(?:\b|$)|QUARTERBACK", regex=True)
    ages = pd.to_numeric(df.get("_report_age_days", np.nan), errors="coerce")
    return {
        "injury_count": int(active.sum()),
        "injury_risk": float(sev[active].sum()),
        "qb_injury_risk": float(sev[qb].max()) if qb.any() else 0.0,
        "injury_report_age_days": float(ages.min()) if ages.notna().any() else np.nan,
        "injury_reports": int(len(df)),
    }


def _active_value(row, active_col=None, status_col=None):
    if active_col:
        v = row.get(active_col)
        if isinstance(v, bool):
            return v, True
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "active", "available"}:
            return True, True
        if s in {"0", "false", "f", "no", "inactive"}:
            return False, True
    if status_col:
        s = str(row.get(status_col) or "").lower()
        if any(x in s for x in ("inactive", "suspended", "injured reserve", "out")):
            return False, True
        if any(x in s for x in ("active", "available", "healthy")):
            return True, True
    return True, False


def summarize_roster_rows(df: pd.DataFrame) -> dict:
    """Summarize current roster availability without inventing missing status fields."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "roster_count": 0, "roster_active_count": 0, "roster_inactive_count": 0,
            "roster_inactive_share": np.nan, "qb_roster_count": 0, "active_qb_count": 0,
            "roster_availability_risk": 0.0, "roster_status_known": False, "roster_position_known": False,
        }
    aid = _col(df.columns, "athlete_id", "player_id", "id")
    an = _col(df.columns, "athlete_name", "player_name", "full_name", "name")
    posc = _col(df.columns, "position_abbreviation", "position_name", "position")
    activec = _col(df.columns, "active", "is_active")
    statusc = _col(df.columns, "status", "status_name", "status_type")
    work = df.copy()
    if aid:
        key = work[aid].map(canon_id)
    elif an:
        key = work[an].map(canon_team)
    else:
        key = pd.Series(work.index.astype(str), index=work.index)
    work["_athlete_key"] = key.where(key.ne(""), "row:" + work.index.astype(str))
    work = work.drop_duplicates("_athlete_key", keep="last")
    states = work.apply(lambda r: _active_value(r, activec, statusc), axis=1)
    active = states.map(lambda x: bool(x[0]))
    known = states.map(lambda x: bool(x[1]))
    roster_count = int(len(work))
    active_count = int(active.sum())
    inactive_count = roster_count - active_count if bool(known.any()) else 0
    inactive_share = inactive_count / roster_count if roster_count and bool(known.any()) else np.nan
    if posc:
        pos = work[posc].fillna("").astype(str).str.upper()
        qb = pos.str.contains(r"(?:^|\b)QB(?:\b|$)|QUARTERBACK", regex=True)
        qb_count = int(qb.sum())
        active_qbs = int((qb & active).sum())
        position_known = bool(pos.str.len().gt(0).any())
    else:
        qb_count = active_qbs = 0
        position_known = False
    risk = min(0.45, 2.0 * float(inactive_share)) if _finite(inactive_share) else 0.0
    if position_known and qb_count > 0:
        if active_qbs == 0:
            risk = max(risk, 0.75)
        elif active_qbs == 1:
            risk = max(risk, 0.08)
    return {
        "roster_count": roster_count, "roster_active_count": active_count,
        "roster_inactive_count": inactive_count,
        "roster_inactive_share": float(inactive_share) if _finite(inactive_share) else np.nan,
        "qb_roster_count": qb_count, "active_qb_count": active_qbs,
        "roster_availability_risk": float(min(1.0, risk)),
        "roster_status_known": bool(known.any()), "roster_position_known": position_known,
    }


def weather_risk(weather: dict, indoor=False) -> float:
    if indoor or not weather:
        return 0.0
    wind = float(weather.get("wind_mph", 0) or 0)
    gust = float(weather.get("wind_gust_mph", 0) or 0)
    precip = float(weather.get("precip_probability", 0) or 0)
    temp = weather.get("temperature_f")
    temp_penalty = 0.0
    if _finite(temp):
        t = float(temp)
        temp_penalty = max(0.0, (35.0 - t) / 35.0) * 0.08 + max(0.0, (t - 95.0) / 25.0) * 0.05
    risk = max(0.0, (wind - 15.0) / 25.0) * 0.52 + max(0.0, (precip - 40.0) / 60.0) * 0.25 + max(0.0, (gust - 25.0) / 35.0) * 0.15 + temp_penalty
    return float(max(0.0, min(1.0, risk)))


def _context_quality(injury_ok, roster_ok, venue_ok, weather_ok, freshness=1.0):
    raw = 0.35 * float(bool(injury_ok)) + 0.25 * float(bool(roster_ok)) + 0.20 * float(bool(venue_ok)) + 0.20 * float(bool(weather_ok))
    return float(max(0.0, min(1.0, raw * max(0.0, min(1.0, float(freshness))))))


def haversine_miles(a, b, c, d):
    if any(x is None or pd.isna(x) for x in (a, b, c, d)):
        return np.nan
    r = 3958.7613
    p1, p2 = math.radians(float(a)), math.radians(float(c))
    dp = math.radians(float(c) - float(a))
    dl = math.radians(float(d) - float(b))
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


class ContextStore:
    """Current-only availability, roster, venue, weather, travel, and rest context."""

    INJURY_TTL_HOURS = 2.0
    ROSTER_TTL_HOURS = 12.0
    TEAM_TTL_HOURS = 24.0

    def __init__(self, season, schedule_frame=None, cache_dir="cache/context", as_of=None):
        self.season = int(season)
        self.as_of = _utc(as_of)
        current_cfb_season = self.as_of.year if self.as_of.month >= 7 else self.as_of.year - 1
        self.live_context_enabled = self.season == current_cfb_season
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.errors, self.sources, self.cache_diagnostics = [], [], []
        self.schedule = schedule_frame.copy() if isinstance(schedule_frame, pd.DataFrame) else pd.DataFrame()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HarbinSportsAnalytics/7.2", "Accept": "application/json,*/*"})
        self._geo_cache, self._weather_cache, self._injury_views = {}, {}, {}
        self.injuries_raw = pd.DataFrame()
        self.roster_id, self.roster_name, self.team_by_id, self.team_by_name, self.game_rows = {}, {}, {}, {}, {}
        self._index_schedule()
        if self.live_context_enabled:
            self._load_injuries()
            self._load_rosters()
        self._load_team_info()

    def _cache_fresh(self, path, max_age_hours):
        path = Path(path)
        if not path.exists() or path.stat().st_size <= 100:
            return False
        return True if max_age_hours is None else (time.time() - path.stat().st_mtime) / 3600.0 <= float(max_age_hours)

    def _download_parquet(self, url, path, source, max_age_hours=None):
        path = Path(path)
        age = (time.time() - path.stat().st_mtime) / 3600.0 if path.exists() else None
        if self._cache_fresh(path, max_age_hours):
            self.cache_diagnostics.append({"source": source, "status": "cache_hit", "age_hours": round(max(0.0, age or 0.0), 2)})
            return pd.read_parquet(path)
        try:
            r = self.session.get(url, timeout=35)
            r.raise_for_status()
            path.write_bytes(r.content)
            self.cache_diagnostics.append({"source": source, "status": "refreshed", "age_hours": 0.0})
            return pd.read_parquet(path)
        except Exception as exc:
            if path.exists() and path.stat().st_size > 100:
                self.cache_diagnostics.append({"source": source, "status": "stale_fallback", "age_hours": round(max(0.0, age or 0.0), 2)})
                self.errors.append(f"{source} refresh: {type(exc).__name__}: {exc}")
                return pd.read_parquet(path)
            self.cache_diagnostics.append({"source": source, "status": "unavailable", "age_hours": None})
            raise

    def _index_schedule(self):
        if self.schedule.empty:
            return
        gc = _col(self.schedule.columns, "game_id", "id")
        if gc:
            for _, r in self.schedule.iterrows():
                self.game_rows[canon_id(r[gc])] = r

    def _load_injuries(self):
        try:
            self.injuries_raw = self._download_parquet(INJURY_URL.format(season=self.season), self.cache / f"injuries_{self.season}.parquet", "injuries", self.INJURY_TTL_HOURS)
        except Exception as exc:
            self.errors.append(f"injuries: {type(exc).__name__}: {exc}")
            self.injuries_raw = pd.DataFrame()
            return
        if len(self.injuries_raw):
            self.sources.append("SportsDataverse ESPN injuries")

    def _load_rosters(self):
        try:
            d = self._download_parquet(ROSTER_URL.format(season=self.season), self.cache / f"cfb_rosters_{self.season}.parquet", "rosters", self.ROSTER_TTL_HOURS)
        except Exception as exc:
            self.errors.append(f"rosters: {type(exc).__name__}: {exc}")
            return
        tic = _col(d.columns, "team_id", ("team", "id"))
        tc = _col(d.columns, "team", "team_name", "school", "team_display_name")
        if tic is None and tc is None:
            self.errors.append("rosters: missing team identity")
            return
        work = d.copy()
        work["_team_id"] = d[tic].map(canon_id) if tic else ""
        work["_team_name"] = d[tc].map(canon_team) if tc else ""
        for (tid, name), z in work.groupby(["_team_id", "_team_name"], dropna=False):
            s = summarize_roster_rows(z)
            if tid:
                self.roster_id[str(tid)] = dict(s)
            if name:
                self.roster_name[str(name)] = dict(s)
        if len(d):
            self.sources.append("SportsDataverse ESPN rosters")

    def _load_team_info(self):
        source_name = "teams"
        try:
            d = self._download_parquet(TEAM_URL.format(season=self.season), self.cache / f"cfb_teams_{self.season}.parquet", source_name, self.TEAM_TTL_HOURS)
        except Exception as first_exc:
            self.errors.append(f"teams: {type(first_exc).__name__}: {first_exc}")
            source_name = "team_info"
            try:
                d = self._download_parquet(LEGACY_TEAM_INFO_URL.format(season=self.season), self.cache / f"cfb_team_info_{self.season}.parquet", source_name, self.TEAM_TTL_HOURS)
            except Exception as exc:
                self.errors.append(f"team_info: {type(exc).__name__}: {exc}")
                return
        ic = _col(d.columns, "team_id", "id")
        nc = _col(d.columns, "team", "team_name", "school", "display_name", "location")
        lat = _col(d.columns, "venue_latitude", ("venue", "latitude"), "latitude", "lat")
        lon = _col(d.columns, "venue_longitude", ("venue", "longitude"), "longitude", "lon")
        elev = _col(d.columns, "venue_elevation", ("venue", "elevation"), "elevation")
        city = _col(d.columns, "venue_city", ("venue", "city"), "city")
        state = _col(d.columns, "venue_state", ("venue", "state"), "state")
        vc = _col(d.columns, "venue_id", ("venue", "id"))
        indoorc = _col(d.columns, "venue_indoor", ("venue", "indoor"), "indoor")
        for _, r in d.iterrows():
            def num(c):
                try:
                    return float(r[c]) if c and pd.notna(r[c]) else np.nan
                except Exception:
                    return np.nan
            meta = {
                "latitude": num(lat), "longitude": num(lon), "elevation": num(elev),
                "city": str(r.get(city, "")) if city else "", "state": str(r.get(state, "")) if state else "",
                "venue_id": canon_id(r.get(vc)) if vc else "", "indoor": _boolish(r.get(indoorc)) if indoorc else False,
                "indoor_known": bool(indoorc),
            }
            if ic and canon_id(r.get(ic)):
                self.team_by_id[canon_id(r.get(ic))] = meta
            if nc:
                self.team_by_name[canon_team(r.get(nc))] = meta
        if len(d):
            self.sources.append("SportsDataverse ESPN teams" if source_name == "teams" else "SportsDataverse cfb_team_info")

    def _team_meta(self, team, tid=""):
        return dict(self.team_by_id.get(canon_id(tid), self.team_by_name.get(canon_team(team), {})))

    def _roster(self, team, tid=""):
        return dict(self.roster_id.get(canon_id(tid), self.roster_name.get(canon_team(team), summarize_roster_rows(pd.DataFrame()))))

    def _injury_maps(self, target_week=None):
        key = int(target_week) if target_week is not None and _finite(target_week) else None
        if key in self._injury_views:
            return self._injury_views[key]
        norm = normalize_injury_rows(self.injuries_raw, self.as_of, key)
        by_id, by_name = {}, {}
        if not norm.empty:
            for _, z in norm.groupby("_team_key", dropna=False):
                summary = summarize_injury_rows(z)
                for tid in [x for x in z["_team_id"].unique().tolist() if x]:
                    by_id[str(tid)] = dict(summary)
                for name in [x for x in z["_team_name"].unique().tolist() if x]:
                    by_name[str(name)] = dict(summary)
        self._injury_views[key] = (by_id, by_name)
        return by_id, by_name

    def _injury(self, team, tid="", target_week=None):
        by_id, by_name = self._injury_maps(target_week)
        return dict(by_id.get(canon_id(tid), by_name.get(canon_team(team), summarize_injury_rows(pd.DataFrame()))))

    def _geocode(self, meta):
        if not meta:
            return meta
        if pd.notna(meta.get("latitude", np.nan)) and pd.notna(meta.get("longitude", np.nan)):
            return meta
        q = ", ".join(x for x in (meta.get("city", ""), meta.get("state", "")) if x and x.lower() != "nan")
        if not q:
            return meta
        if q in self._geo_cache:
            return {**meta, **self._geo_cache[q]}
        try:
            r = self.session.get(GEO, params={"name": q, "count": 1, "language": "en", "format": "json"}, timeout=8)
            r.raise_for_status()
            res = (r.json().get("results") or [None])[0]
            v = {"latitude": float(res["latitude"]), "longitude": float(res["longitude"]), "elevation": float(res.get("elevation", np.nan))} if res else {}
            self._geo_cache[q] = v
            return {**meta, **v}
        except Exception as exc:
            self.errors.append(f"geocode {q}: {type(exc).__name__}")
            self._geo_cache[q] = {}
            return meta

    def _rest(self, tid, team, kick):
        if self.schedule.empty or pd.isna(kick):
            return np.nan
        try:
            dates = pd.to_datetime(self.schedule["start_date"], utc=True, errors="coerce")
            b = self.schedule[dates < kick].copy()
            hid, aid = _col(b.columns, "home_id"), _col(b.columns, "away_id")
            if hid and aid and canon_id(tid):
                mask = b[hid].map(canon_id).eq(canon_id(tid)) | b[aid].map(canon_id).eq(canon_id(tid))
            else:
                mask = b.get("home_team", pd.Series(index=b.index, dtype=str)).map(canon_team).eq(canon_team(team)) | b.get("away_team", pd.Series(index=b.index, dtype=str)).map(canon_team).eq(canon_team(team))
            b = b[mask]
            if "completed" in b:
                b = b[b.completed.astype(str).str.lower().isin({"true", "1", "t", "yes"})]
            if b.empty:
                return np.nan
            return float((kick - pd.to_datetime(b.start_date, utc=True, errors="coerce").max()).total_seconds() / 86400)
        except Exception:
            return np.nan

    def _weather(self, meta, kick):
        meta = self._geocode(meta)
        if _boolish(meta.get("indoor", False)):
            return {"indoor_weather_suppressed": 1.0}, meta
        if not self.live_context_enabled or pd.isna(kick):
            return {}, meta
        if kick < self.as_of - pd.Timedelta(hours=6) or kick > self.as_of + pd.Timedelta(days=16):
            return {}, meta
        lat, lon = meta.get("latitude", np.nan), meta.get("longitude", np.nan)
        if pd.isna(lat) or pd.isna(lon):
            return {}, meta
        day = kick.strftime("%Y-%m-%d")
        key = (round(float(lat), 2), round(float(lon), 2), kick.floor("h").isoformat())
        if key in self._weather_cache:
            return self._weather_cache[key], meta
        try:
            r = self.session.get(OPEN_METEO, params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
                "start_date": day, "end_date": day,
            }, timeout=10)
            r.raise_for_status()
            jdata = r.json()
            h = jdata.get("hourly") or {}
            ts = pd.to_datetime(h.get("time", []), utc=True, errors="coerce")
            if not len(ts):
                return {}, meta
            j = int(np.argmin(np.abs((ts - kick).total_seconds())))
            def val(name):
                values = h.get(name) or [np.nan] * len(ts)
                try:
                    return float(values[j])
                except Exception:
                    return np.nan
            precip = val("precipitation")
            result = {
                "temperature_f": val("temperature_2m"), "precip_probability": val("precipitation_probability"),
                "precipitation_in": precip / 25.4 if _finite(precip) else np.nan,
                "weather_code": val("weather_code"), "wind_mph": val("wind_speed_10m"), "wind_gust_mph": val("wind_gusts_10m"),
            }
            if pd.isna(meta.get("elevation", np.nan)) and jdata.get("elevation") is not None:
                meta["elevation"] = float(jdata["elevation"]) * 3.28084
            self._weather_cache[key] = result
            return result, meta
        except Exception as exc:
            self.errors.append(f"weather {key}: {type(exc).__name__}: {exc}")
            self._weather_cache[key] = {}
            return {}, meta

    def _freshness_factor(self):
        latest = {d.get("source"): d.get("status") for d in self.cache_diagnostics}
        good = {"refreshed", "cache_hit", "stale_fallback"}
        if latest.get("teams") in good:
            required = ["teams"]
        elif latest.get("team_info") in good:
            required = ["team_info"]
        else:
            required = ["teams"] if "teams" in latest else (["team_info"] if "team_info" in latest else [])
        if self.live_context_enabled:
            required += ["injuries", "rosters"]
        if not required:
            return 0.0
        score = {"refreshed": 1.0, "cache_hit": 1.0, "stale_fallback": 0.65, "unavailable": 0.0}
        return float(np.mean([score.get(latest.get(x), 0.0) for x in required]))

    def attach(self, pred):
        out = pred.copy()
        if out.empty:
            return out, {
                "sources": list(dict.fromkeys(self.sources)), "errors": self.errors[-30:],
                "coverage": 0.0, "core_coverage": 0.0, "quality_score": 0.0,
                "weather_coverage": 0.0, "roster_coverage": 0.0, "injury_source_coverage": 0.0,
                "travel_coverage": 0.0, "venue_coverage": 0.0, "live_context_enabled": self.live_context_enabled,
            }
        n = len(out)
        weather_resolved = roster_hits = travel_hits = venue_hits = injury_active_games = 0
        row_quality = []
        freshness = self._freshness_factor()
        injury_source_ok = self.live_context_enabled and not self.injuries_raw.empty
        for i, r in out.iterrows():
            sr = self.game_rows.get(canon_id(r.game_id))
            hid = canon_id(sr.get("home_id")) if sr is not None and "home_id" in sr else ""
            aid = canon_id(sr.get("away_id")) if sr is not None and "away_id" in sr else ""
            target_week = int(r.get("week")) if _finite(r.get("week")) else None
            h = self._injury(r.home_team, hid, target_week) if injury_source_ok else summarize_injury_rows(pd.DataFrame())
            a = self._injury(r.away_team, aid, target_week) if injury_source_ok else summarize_injury_rows(pd.DataFrame())
            hrst = self._roster(r.home_team, hid) if self.live_context_enabled else summarize_roster_rows(pd.DataFrame())
            arst = self._roster(r.away_team, aid) if self.live_context_enabled else summarize_roster_rows(pd.DataFrame())
            roster_ok = hrst["roster_count"] > 0 and arst["roster_count"] > 0
            roster_hits += int(roster_ok)
            injury_active_games += int(h["injury_count"] + a["injury_count"] > 0)
            def team_availability(inj, roster):
                injury_component = min(1.0, 0.12 * float(inj["injury_risk"]) + 0.45 * float(inj["qb_injury_risk"]))
                return min(1.0, 0.75 * injury_component + 0.25 * float(roster["roster_availability_risk"]))
            home_avail, away_avail = team_availability(h, hrst), team_availability(a, arst)
            availability = max(home_avail, away_avail)
            out.at[i, "home_injury_count"] = h["injury_count"]
            out.at[i, "away_injury_count"] = a["injury_count"]
            out.at[i, "home_qb_injury_risk"] = h["qb_injury_risk"]
            out.at[i, "away_qb_injury_risk"] = a["qb_injury_risk"]
            out.at[i, "home_injury_report_age_days"] = h["injury_report_age_days"]
            out.at[i, "away_injury_report_age_days"] = a["injury_report_age_days"]
            out.at[i, "home_roster_count"] = hrst["roster_count"]
            out.at[i, "away_roster_count"] = arst["roster_count"]
            out.at[i, "home_active_qb_count"] = hrst["active_qb_count"]
            out.at[i, "away_active_qb_count"] = arst["active_qb_count"]
            out.at[i, "home_roster_availability_risk"] = hrst["roster_availability_risk"]
            out.at[i, "away_roster_availability_risk"] = arst["roster_availability_risk"]
            out.at[i, "home_availability_risk"] = home_avail
            out.at[i, "away_availability_risk"] = away_avail
            out.at[i, "availability_edge_home"] = away_avail - home_avail
            out.at[i, "availability_risk"] = availability

            hm = self._geocode(self._team_meta(r.home_team, hid))
            am = self._geocode(self._team_meta(r.away_team, aid))
            venue_ok = bool(hm) and (_boolish(hm.get("indoor", False)) or (_finite(hm.get("latitude")) and _finite(hm.get("longitude"))))
            venue_hits += int(venue_ok)
            kick = pd.to_datetime(r.date, utc=True, errors="coerce")
            home_rest, away_rest = self._rest(hid, r.home_team, kick), self._rest(aid, r.away_team, kick)
            out.at[i, "home_rest_days"] = home_rest
            out.at[i, "away_rest_days"] = away_rest
            out.at[i, "rest_edge_home"] = home_rest - away_rest if pd.notna(home_rest) and pd.notna(away_rest) else np.nan
            travel = haversine_miles(am.get("latitude"), am.get("longitude"), hm.get("latitude"), hm.get("longitude"))
            out.at[i, "away_travel_miles"] = travel
            travel_hits += int(pd.notna(travel))
            indoor = _boolish(hm.get("indoor", False))
            w, hm = self._weather(hm, kick)
            weather_ok = indoor or bool(w)
            weather_resolved += int(weather_ok)
            out.at[i, "venue_indoor"] = bool(indoor)
            elev = hm.get("elevation", np.nan)
            out.at[i, "venue_elevation_ft"] = elev
            out.at[i, "altitude_change_away_ft"] = elev - am.get("elevation", np.nan) if pd.notna(elev) and pd.notna(am.get("elevation", np.nan)) else np.nan
            for k, v in w.items():
                out.at[i, k] = v
            wr = weather_risk(w, indoor=indoor)
            tr = min(0.5, max(0.0, (float(travel) - 800.0) / 2200.0)) if pd.notna(travel) else 0.0
            rr = 0.35 if pd.notna(away_rest) and away_rest < 6 else 0.0
            travel_rest = min(1.0, tr + rr)
            quality = _context_quality(injury_source_ok, roster_ok, venue_ok, weather_ok, freshness)
            base_context_risk = 0.64 * availability + 0.21 * wr + 0.15 * travel_rest
            context_risk = min(1.0, base_context_risk + 0.30 * (1.0 - quality))
            row_quality.append(quality)
            out.at[i, "weather_risk"] = wr
            out.at[i, "travel_rest_risk"] = travel_rest
            out.at[i, "context_risk"] = context_risk
            out.at[i, "context_quality_score"] = quality
        if weather_resolved:
            self.sources.append("Open-Meteo forecast / indoor venue suppression")
        if travel_hits:
            self.sources.append("Venue geocoding / team metadata")
        injury_cov = 1.0 if injury_source_ok else 0.0
        roster_cov, venue_cov, weather_cov = roster_hits / n, venue_hits / n, weather_resolved / n
        core_cov = min(injury_cov, roster_cov, venue_cov, weather_cov)
        quality_score = float(np.mean(row_quality)) if row_quality else 0.0
        return out, {
            "sources": list(dict.fromkeys(self.sources)), "errors": self.errors[-30:],
            "coverage": quality_score, "core_coverage": core_cov, "quality_score": quality_score,
            "freshness_factor": freshness, "weather_coverage": weather_cov,
            "injury_source_coverage": injury_cov, "active_injury_game_rate": injury_active_games / n,
            "roster_coverage": roster_cov, "venue_coverage": venue_cov, "travel_coverage": travel_hits / n,
            "live_context_enabled": self.live_context_enabled, "as_of": self.as_of.isoformat(),
            "cache_diagnostics": self.cache_diagnostics[-20:],
        }
