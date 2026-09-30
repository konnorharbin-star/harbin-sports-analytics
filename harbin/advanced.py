from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ADV_TEAM_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_adv_team/adv_team_{season}.csv"
RAW_BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-data/main/data"


def canon_team(v):
    s = re.sub(r"[^a-z0-9]", "", str(v or "").lower())
    aliases = {
        "northcarolinastate": "ncstate", "southernmethodist": "smu", "texaschristian": "tcu",
        "brighamyoung": "byu", "centralflorida": "ucf", "louisianastate": "lsu",
        "alabamabirmingham": "uab", "nevadalasvegas": "unlv", "texaselpaso": "utep",
        "texassanantonio": "utsa", "floridainternational": "fiu",
        "southernmississippi": "southernmiss", "miamiflorida": "miami",
    }
    return aliases.get(s, s)


def canon_id(v):
    try:
        if v is None or pd.isna(v):
            return ""
        return str(int(float(v)))
    except Exception:
        return str(v).strip()


def _cache_is_fresh(path: Path, max_age_hours: float | None) -> bool:
    if not path.exists() or path.stat().st_size <= 100:
        return False
    if max_age_hours is None:
        return True
    age_hours = max(0.0, (time.time() - path.stat().st_mtime) / 3600.0)
    return age_hours <= float(max_age_hours)


def _cached(url, path, max_age_hours=None, diagnostics=None):
    """Read-through CSV cache with explicit current-data freshness.

    Historical seasons can be cached indefinitely. Current-season and preseason-prior
    files use a TTL. If refresh fails but a valid cached file exists, use the stale copy
    and record that fallback rather than silently freezing or crashing the feature stack.
    """
    path = Path(path)
    diagnostics = diagnostics if diagnostics is not None else []
    if _cache_is_fresh(path, max_age_hours):
        diagnostics.append({"path": str(path), "status": "cache_hit", "max_age_hours": max_age_hours})
        return pd.read_csv(path, low_memory=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=40, headers={"User-Agent": "HarbinSportsAnalytics/7.2"})
        r.raise_for_status()
        path.write_bytes(r.content)
        diagnostics.append({"path": str(path), "status": "refreshed", "max_age_hours": max_age_hours})
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        if path.exists() and path.stat().st_size > 100:
            diagnostics.append({
                "path": str(path), "status": "stale_fallback", "max_age_hours": max_age_hours,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return pd.read_csv(path, low_memory=False)
        raise


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


def _numeric_ratio(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    s = series.dropna()
    if s.empty:
        return 0.0
    return float(pd.to_numeric(s, errors="coerce").notna().mean())


def identify_team_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (numeric ESPN-team-id column, textual team-name column)."""
    cols = list(df.columns)
    exact_id = ["team_id", "teamId", "pos_team_id", "posTeamId", "team.id"]
    exact_name = ["team", "team_name", "school", "display_name", "pos_team", "posTeam"]
    id_col = None
    for name in exact_id:
        c = next((x for x in cols if str(x).lower() == name.lower()), None)
        if c is not None and _numeric_ratio(df[c]) >= 0.60:
            id_col = c
            break
    if id_col is None:
        for c in cols:
            lc = str(c).lower()
            if (lc.endswith("team_id") or ("team" in lc and lc.endswith("id"))) and _numeric_ratio(df[c]) >= 0.60:
                id_col = c
                break
    name_col = None
    for name in exact_name:
        c = next((x for x in cols if str(x).lower() == name.lower()), None)
        if c is not None and c != id_col and _numeric_ratio(df[c]) < 0.60:
            name_col = c
            break
    if name_col is None:
        for c in cols:
            lc = str(c).lower()
            if c != id_col and any(k in lc for k in ("team", "school")) and not lc.endswith("id") and _numeric_ratio(df[c]) < 0.60:
                name_col = c
                break
    pos = next((x for x in cols if str(x).lower() in {"pos_team", "posteam"}), None)
    if pos is not None:
        ratio = _numeric_ratio(df[pos])
        if id_col is None and ratio >= 0.60:
            id_col = pos
        if name_col is None and ratio < 0.60:
            name_col = pos
    return id_col, name_col


class AdvancedFeatureStore:
    """Leakage-safe pregame efficiency and roster priors keyed by ESPN team identity.

    Week W sees only observations from weeks < W. Each dynamic metric is represented by
    both a season-to-date mean and a pregame EWMA recent-form value. Future schedule
    weeks and bye weeks fall back to the latest leak-free snapshot.
    """

    KEYWORDS = (
        "epa", "success", "explos", "line_yard", "stuff", "power", "first_down",
        "scoring", "points_per", "sec_per", "plays_per", "havoc", "turnover",
        "penalty", "field_pos", "starting_fp", "yards_per",
    )
    BLOCK = (
        "game_id", "season", "week", "team", "pos_team", "opponent", "opp_team",
        "home", "away", "score", "points", "win", "result", "id",
    )
    RECENT_ALPHA = 0.35
    CURRENT_SEASON_TTL_HOURS = 6.0
    STATIC_PRIOR_TTL_HOURS = 24.0

    def __init__(self, start_season, end_season, cache_dir="cache/advanced"):
        self.start_season = int(start_season)
        self.end_season = int(end_season)
        self.cache = Path(cache_dir)
        self.errors = []
        self.sources = []
        self.cache_diagnostics = []
        self.id_lookup = {}
        self.name_lookup = {}
        self.static_lookup = {}
        self.feature_names = []
        self.dynamic_names = []
        self.identity_diagnostics = {}
        self.source_rows = {}
        self._build()

    def _load_adv(self):
        frames = []
        for season in range(self.start_season, self.end_season + 1):
            try:
                ttl = self.CURRENT_SEASON_TTL_HOURS if season == self.end_season else None
                d = _cached(
                    ADV_TEAM_URL.format(season=season), self.cache / f"adv_team_{season}.csv",
                    max_age_hours=ttl, diagnostics=self.cache_diagnostics,
                )
                if len(d):
                    d = d.copy()
                    if "season" not in d.columns:
                        d["season"] = season
                    self.source_rows[str(season)] = int(len(d))
                    frames.append(d)
            except Exception as e:
                self.errors.append(f"adv_team {season}: {type(e).__name__}: {e}")
        if frames:
            self.sources.append("SportsDataverse espn_cfb_adv_team")
            return pd.concat(frames, ignore_index=True, sort=False)
        return pd.DataFrame()

    def _metric_frame(self, df):
        keep = []
        for c in df.columns:
            lc = str(c).lower()
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() < max(50, int(len(df) * .03)):
                continue
            if any(x == lc for x in self.BLOCK) or lc.endswith("_id") or lc in {"season", "week"}:
                continue
            if any(k in lc for k in self.KEYWORDS):
                keep.append(c)
        keep = keep[:32]
        out = pd.DataFrame(index=df.index)
        for c in keep:
            name = "adv_" + re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
            out[name] = pd.to_numeric(df[c], errors="coerce")
        return out

    def _store(self, season, week, team_id, team_name, values):
        if str(team_id):
            self.id_lookup[(int(season), int(week), str(team_id))] = dict(values)
        if str(team_name):
            self.name_lookup[(int(season), int(week), str(team_name))] = dict(values)

    def _snapshot(self, state, carry, metrics):
        out = {}
        for m in metrics:
            st = state[m]
            carry_m = carry.get(m, {})
            if st["count"]:
                mean_value = st["sum"] / st["count"]
            else:
                v = carry_m.get("mean", np.nan)
                mean_value = 0.62 * float(v) if pd.notna(v) else np.nan
            if pd.notna(st["recent"]):
                recent_value = float(st["recent"])
            else:
                v = carry_m.get("recent", np.nan)
                recent_value = 0.62 * float(v) if pd.notna(v) else np.nan
            out[m] = mean_value
            out["recent_" + m] = recent_value
        return out

    def _build_asof(self, df):
        if df.empty:
            return
        sc = _col(df.columns, "season", "year")
        wc = _col(df.columns, "week")
        ic, nc = identify_team_columns(df)
        self.identity_diagnostics = {
            "id_column": str(ic) if ic is not None else None,
            "name_column": str(nc) if nc is not None else None,
            "id_numeric_ratio": _numeric_ratio(df[ic]) if ic is not None else 0.0,
        }
        if sc is None or wc is None or (ic is None and nc is None):
            self.errors.append("adv_team: missing season/week/team identity keys")
            return
        vals = self._metric_frame(df)
        if vals.empty:
            self.errors.append("adv_team: no recognized numeric efficiency columns")
            return
        work = pd.DataFrame({
            "season": pd.to_numeric(df[sc], errors="coerce"),
            "week": pd.to_numeric(df[wc], errors="coerce"),
            "team_id": df[ic].map(canon_id) if ic is not None else "",
            "team_name": df[nc].map(canon_team) if nc is not None else "",
        })
        work = pd.concat([work, vals], axis=1).dropna(subset=["season", "week"])
        work.season = work.season.astype(int)
        work.week = work.week.astype(int)
        metrics = list(vals.columns)
        wr = work.groupby(
            ["season", "week", "team_id", "team_name"], as_index=False, dropna=False
        )[metrics].mean(numeric_only=True)
        wr = wr.sort_values(["season", "week", "team_id", "team_name"])

        prior = {}
        for season in sorted(wr.season.unique()):
            states = {}
            sub = wr[wr.season == season]
            for _, row in sub.iterrows():
                persistent = ("id", str(row.team_id)) if str(row.team_id) else ("name", str(row.team_name))
                st = states.setdefault(
                    persistent, {m: {"sum": 0.0, "count": 0, "recent": np.nan} for m in metrics}
                )
                carry = prior.get(persistent, {})
                # Snapshot BEFORE consuming the current row: strict pregame as-of semantics.
                pre = self._snapshot(st, carry, metrics)
                self._store(season, int(row.week), row.team_id, row.team_name, pre)
                for m in metrics:
                    v = row.get(m)
                    if pd.notna(v) and math.isfinite(float(v)):
                        v = float(v)
                        state = st[m]
                        state["sum"] += v
                        state["count"] += 1
                        state["recent"] = v if pd.isna(state["recent"]) else (
                            (1 - self.RECENT_ALPHA) * float(state["recent"]) + self.RECENT_ALPHA * v
                        )
                # Make next week available even when upstream has no row yet (or after a bye).
                post = self._snapshot(st, {}, metrics)
                self._store(season, int(row.week) + 1, row.team_id, row.team_name, post)

            for key, st in states.items():
                old = prior.get(key, {})
                nxt = {}
                for m, state in st.items():
                    old_m = old.get(m, {})
                    nxt[m] = {
                        "mean": state["sum"] / state["count"] if state["count"] else old_m.get("mean", np.nan),
                        "recent": float(state["recent"]) if pd.notna(state["recent"]) else old_m.get("recent", np.nan),
                    }
                prior[key] = nxt

        recent = ["recent_" + m for m in metrics]
        self.dynamic_names = metrics + recent
        self.feature_names = list(self.dynamic_names)

    def _load_static(self):
        files = {
            "talent": "cfb_matchup_roster_talent.csv",
            "returning": "cfb_matchup_returning_production.csv",
            "continuity": "cfb_matchup_coach_continuity.csv",
        }
        for label, fn in files.items():
            try:
                d = _cached(
                    f"{RAW_BASE}/{fn}", self.cache / fn,
                    max_age_hours=self.STATIC_PRIOR_TTL_HOURS, diagnostics=self.cache_diagnostics,
                )
            except Exception as e:
                self.errors.append(f"{label}: {type(e).__name__}: {e}")
                continue
            sc = _col(d.columns, "season", "year")
            tc = _col(d.columns, "team", "school", "team_name")
            if sc is None or tc is None:
                self.errors.append(f"{label}: missing season/team keys")
                continue
            nums = []
            for c in d.columns:
                lc = str(c).lower()
                s = pd.to_numeric(d[c], errors="coerce")
                if c not in (sc, tc) and any(k in lc for k in ("talent", "rtprod", "return", "cont", "experience")) and s.notna().sum() >= max(5, int(len(d) * .1)):
                    nums.append(c)
            for _, r in d.iterrows():
                try:
                    key = (int(r[sc]), canon_team(r[tc]))
                except Exception:
                    continue
                dest = self.static_lookup.setdefault(key, {})
                for c in nums:
                    v = pd.to_numeric(pd.Series([r[c]]), errors="coerce").iloc[0]
                    if pd.notna(v):
                        dest["prior_" + re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")] = float(v)
            if nums:
                self.sources.append(f"SportsDataverse {fn}")

    def _build(self):
        self._build_asof(self._load_adv())
        self._load_static()
        self.feature_names = list(dict.fromkeys(
            self.feature_names + [k for v in self.static_lookup.values() for k in v]
        ))

    def _team(self, season, week, team, team_id=""):
        out = {}
        season, week = int(season), int(week)
        cid, cname = canon_id(team_id), canon_team(team)
        if cid:
            for w in range(week, 0, -1):
                hit = self.id_lookup.get((season, w, cid))
                if hit is not None:
                    out.update(hit)
                    break
        if not any(k in out for k in self.dynamic_names):
            for w in range(week, 0, -1):
                hit = self.name_lookup.get((season, w, cname))
                if hit is not None:
                    out.update(hit)
                    break
        out.update(self.static_lookup.get((season, cname), {}))
        return out

    def enrich(self, frame):
        base_meta = {
            "sources": list(dict.fromkeys(self.sources)),
            "errors": self.errors[-30:],
            "feature_count": len(self.feature_names),
            "dynamic_feature_count": len(self.dynamic_names),
            "identity": self.identity_diagnostics,
            "source_rows": self.source_rows,
            "cache": self.cache_diagnostics[-30:],
            "asof_policy": "week W receives only observations from weeks < W; latest leak-free prior snapshot is used across byes/source lag",
            "recent_form_alpha": self.RECENT_ALPHA,
            "lookup_keys": int(len(self.id_lookup) + len(self.name_lookup)),
            "static_team_seasons": int(len(self.static_lookup)),
        }
        if frame.empty:
            return frame.copy(), {
                **base_meta, "coverage": 0.0, "dynamic_coverage": 0.0,
                "pair_feature_coverage": 0.0, "dynamic_pair_feature_coverage": 0.0,
            }
        out = frame.copy()
        hits = dynamic_hits = 0
        names = list(dict.fromkeys(self.feature_names))
        dyn = set(self.dynamic_names)
        pair_cov, dyn_pair_cov = [], []
        for i, r in out.iterrows():
            hv = self._team(int(r.season), int(r.week), str(r.home_team), r.get("home_id", ""))
            av = self._team(int(r.season), int(r.week), str(r.away_team), r.get("away_id", ""))
            hits += int(bool(hv or av))
            dynamic_hits += int(any(k in hv or k in av for k in dyn))
            home_available = sum(pd.notna(hv.get(n, np.nan)) for n in names)
            away_available = sum(pd.notna(av.get(n, np.nan)) for n in names)
            pair_available = sum(pd.notna(hv.get(n, np.nan)) and pd.notna(av.get(n, np.nan)) for n in names)
            dyn_pair_available = sum(pd.notna(hv.get(n, np.nan)) and pd.notna(av.get(n, np.nan)) for n in dyn)
            denom, dyn_denom = max(1, len(names)), max(1, len(dyn))
            out.at[i, "advanced_home_coverage"] = home_available / denom
            out.at[i, "advanced_away_coverage"] = away_available / denom
            out.at[i, "advanced_pair_coverage"] = pair_available / denom
            out.at[i, "advanced_dynamic_pair_coverage"] = dyn_pair_available / dyn_denom
            pair_cov.append(pair_available / denom)
            dyn_pair_cov.append(dyn_pair_available / dyn_denom)
            for n in names:
                h, a = hv.get(n, np.nan), av.get(n, np.nan)
                out.at[i, "home_" + n] = h
                out.at[i, "away_" + n] = a
                if pd.notna(h) and pd.notna(a):
                    out.at[i, "diff_" + n] = float(h) - float(a)
                    out.at[i, "avg_" + n] = (float(h) + float(a)) / 2
                else:
                    out.at[i, "diff_" + n] = np.nan
                    out.at[i, "avg_" + n] = np.nan
        return out, {
            **base_meta,
            "coverage": hits / len(out),
            "dynamic_coverage": dynamic_hits / len(out),
            "pair_feature_coverage": float(np.mean(pair_cov)) if pair_cov else 0.0,
            "dynamic_pair_feature_coverage": float(np.mean(dyn_pair_cov)) if dyn_pair_cov else 0.0,
        }
