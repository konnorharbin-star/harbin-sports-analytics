from __future__ import annotations

import math
import re
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


def _cached(url, path):
    if path.exists() and path.stat().st_size > 100:
        return pd.read_csv(path, low_memory=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=40, headers={"User-Agent": "HarbinSportsAnalytics/7.2"})
    r.raise_for_status()
    path.write_bytes(r.content)
    return pd.read_csv(path, low_memory=False)


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
    """Return (numeric ESPN-team-id column, textual team-name column).

    SportsDataverse schemas have changed over time and `pos_team` has existed as both
    a team identifier and a human-readable team field in adjacent datasets. The loader
    chooses by both semantic name and observed value type instead of guessing.
    """
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
    """Leak-free pregame advanced efficiency plus roster priors, keyed by ESPN team ID/name."""

    KEYWORDS = (
        "epa", "success", "explos", "line_yard", "stuff", "power", "first_down",
        "scoring", "points_per", "sec_per", "plays_per", "havoc", "turnover",
        "penalty", "field_pos", "starting_fp", "yards_per",
    )
    BLOCK = (
        "game_id", "season", "week", "team", "pos_team", "opponent", "opp_team",
        "home", "away", "score", "points", "win", "result", "id",
    )

    def __init__(self, start_season, end_season, cache_dir="cache/advanced"):
        self.start_season = int(start_season)
        self.end_season = int(end_season)
        self.cache = Path(cache_dir)
        self.errors = []
        self.sources = []
        self.id_lookup = {}
        self.name_lookup = {}
        self.static_lookup = {}
        self.feature_names = []
        self.dynamic_names = []
        self.identity_diagnostics = {}
        self._build()

    def _load_adv(self):
        frames = []
        for season in range(self.start_season, self.end_season + 1):
            try:
                d = _cached(ADV_TEAM_URL.format(season=season), self.cache / f"adv_team_{season}.csv")
                if len(d):
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

    @staticmethod
    def _state_mean(st, metrics):
        return {m: (st[m][0] / st[m][1] if st[m][1] else np.nan) for m in metrics}

    def _store(self, season, week, team_id, team_name, values):
        if str(team_id):
            self.id_lookup[(int(season), int(week), str(team_id))] = dict(values)
        if str(team_name):
            self.name_lookup[(int(season), int(week), str(team_name))] = dict(values)

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

        # Some source seasons contain more than one split for the same team/week.
        wr = work.groupby(["season", "week", "team_id", "team_name"], as_index=False, dropna=False)[metrics].mean(numeric_only=True)
        wr = wr.sort_values(["season", "week", "team_id", "team_name"])
        prior = {}
        for season in sorted(wr.season.unique()):
            states = {}
            sub = wr[wr.season == season]
            for _, row in sub.iterrows():
                persistent = ("id", str(row.team_id)) if str(row.team_id) else ("name", str(row.team_name))
                st = states.setdefault(persistent, {m: [0.0, 0] for m in metrics})
                carry = prior.get(persistent, {})

                # Exact game-week lookup is strictly pregame: current-week metrics are
                # not incorporated until after this snapshot is stored.
                pre = {
                    m: (st[m][0] / st[m][1] if st[m][1] else (.62 * carry[m] if m in carry and pd.notna(carry[m]) else np.nan))
                    for m in metrics
                }
                self._store(season, int(row.week), row.team_id, row.team_name, pre)

                for m in metrics:
                    v = row.get(m)
                    if pd.notna(v) and math.isfinite(float(v)):
                        st[m][0] += float(v)
                        st[m][1] += 1

                # A future schedule week may not yet have a source row. Store the
                # just-completed cumulative state at week+1 so live week N uses data
                # through week N-1 without waiting for an N-row to appear upstream.
                post = self._state_mean(st, metrics)
                self._store(season, int(row.week) + 1, row.team_id, row.team_name, post)

            for key, st in states.items():
                prior[key] = {m: (v[0] / v[1] if v[1] else prior.get(key, {}).get(m, np.nan)) for m, v in st.items()}

        self.dynamic_names = metrics
        self.feature_names = list(metrics)

    def _load_static(self):
        files = {
            "talent": "cfb_matchup_roster_talent.csv",
            "returning": "cfb_matchup_returning_production.csv",
            "continuity": "cfb_matchup_coach_continuity.csv",
        }
        for label, fn in files.items():
            try:
                d = _cached(f"{RAW_BASE}/{fn}", self.cache / fn)
            except Exception as e:
                self.errors.append(f"{label}: {type(e).__name__}: {e}")
                continue
            sc = _col(d.columns, "season", "year")
            tc = _col(d.columns, "team", "school", "team_name")
            if sc is None or tc is None:
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
        self.feature_names = list(dict.fromkeys(self.feature_names + [k for v in self.static_lookup.values() for k in v]))

    def _team(self, season, week, team, team_id=""):
        out = {}
        season, week = int(season), int(week)
        cid = canon_id(team_id)
        cname = canon_team(team)

        # Exact lookup first. For byes or lagging upstream weekly files, walk backward
        # to the most recent leak-free snapshot instead of dropping all dynamics.
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
        }
        if frame.empty:
            return frame.copy(), {**base_meta, "coverage": 0.0, "dynamic_coverage": 0.0}

        out = frame.copy()
        hits = dynamic_hits = 0
        names = set(self.feature_names)
        for i, r in out.iterrows():
            hv = self._team(int(r.season), int(r.week), str(r.home_team), r.get("home_id", ""))
            av = self._team(int(r.season), int(r.week), str(r.away_team), r.get("away_id", ""))
            hits += int(bool(hv or av))
            dynamic_hits += int(any(k in hv or k in av for k in self.dynamic_names))
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
        }
