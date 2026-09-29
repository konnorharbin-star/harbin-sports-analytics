from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ADV_TEAM_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_adv_team/adv_team_{season}.csv"
RAW_BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-data/main/data"


def canon_team(value: object) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    aliases = {
        "northcarolinastate": "ncstate", "southernmethodist": "smu",
        "texaschristian": "tcu", "brighamyoung": "byu", "centralflorida": "ucf",
        "louisianastate": "lsu", "alabamabirmingham": "uab", "nevadalasvegas": "unlv",
        "texaselpaso": "utep", "texassanantonio": "utsa", "floridainternational": "fiu",
        "southernmississippi": "southernmiss", "miamiflorida": "miami",
    }
    return aliases.get(s, s)


def _read_csv_cached(url: str, path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 100:
        return pd.read_csv(path, low_memory=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30, headers={"User-Agent": "HarbinSportsAnalytics/4.0"})
    r.raise_for_status()
    path.write_bytes(r.content)
    return pd.read_csv(path, low_memory=False)


def _first_col(columns, names=(), contains=()):
    cmap = {str(c).lower(): c for c in columns}
    for n in names:
        if n.lower() in cmap:
            return cmap[n.lower()]
    for c in columns:
        lc = str(c).lower()
        if any(all(tok in lc for tok in group) for group in contains):
            return c
    return None


class AdvancedFeatureStore:
    """Leakage-safe advanced team features from SportsDataverse.

    Every value attached to a game is computed from games strictly before that
    game's week. Current-game advanced box data is never allowed into its own
    feature row. Week 1 may carry a shrunk prior-season mean.
    """

    METRICS = {
        "net_epa": (("off_epa",), ("def_epa",)),
        "net_pass_epa": (("off_pass_epa",), ("def_pass_epa",)),
        "net_rush_epa": (("off_rush_epa",), ("def_rush_epa",)),
        "net_success": (("off_success_rate",), ("def_success_rate",)),
        "net_pass_success": (("off_pass_success_rate",), ("def_pass_success_rate",)),
        "net_rush_success": (("off_rush_success_rate",), ("def_rush_success_rate",)),
        "net_finish": (("off_pts_per_scoring_opp",), ("def_pts_per_scoring_opp",)),
        "net_scoring_opp": (("off_scoring_opp_rate_oe",), ("def_scoring_opp_rate_oe",)),
        "net_third_down": (("off_3rd_down_pct",), ("def_3rd_down_pct",)),
    }
    SINGLE = {
        "pace": ("off_sec_per_play_mean", "off_sec_per_play_median"),
        "plays": ("off_plays_per_game", "plays_per_game"),
        "starting_fp": ("off_starting_fp",),
    }

    def __init__(self, start_season: int, end_season: int, cache_dir="cache/advanced"):
        self.start_season = int(start_season)
        self.end_season = int(end_season)
        self.cache = Path(cache_dir)
        self.errors: list[str] = []
        self.sources: list[str] = []
        self.lookup: dict[tuple[int, int, str], dict[str, float]] = {}
        self.static_lookup: dict[tuple[int, str], dict[str, float]] = {}
        self.feature_names: list[str] = []
        self._build()

    @staticmethod
    def _resolve(df: pd.DataFrame, candidate: str):
        if candidate in df.columns:
            return candidate
        lc = candidate.lower()
        for c in df.columns:
            if str(c).lower() == lc:
                return c
        return None

    def _load_adv(self) -> pd.DataFrame:
        frames = []
        for season in range(self.start_season, self.end_season + 1):
            try:
                url = ADV_TEAM_URL.format(season=season)
                df = _read_csv_cached(url, self.cache / f"adv_team_{season}.csv")
                if len(df):
                    frames.append(df)
            except Exception as exc:
                self.errors.append(f"adv_team {season}: {type(exc).__name__}: {exc}")
        if frames:
            self.sources.append("SportsDataverse espn_cfb_adv_team")
            return pd.concat(frames, ignore_index=True, sort=False)
        return pd.DataFrame()

    def _numeric_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for name, (offs, defs) in self.METRICS.items():
            oc = next((self._resolve(df, x) for x in offs if self._resolve(df, x)), None)
            dc = next((self._resolve(df, x) for x in defs if self._resolve(df, x)), None)
            if oc and dc:
                out[name] = pd.to_numeric(df[oc], errors="coerce") - pd.to_numeric(df[dc], errors="coerce")
        for name, candidates in self.SINGLE.items():
            c = next((self._resolve(df, x) for x in candidates if self._resolve(df, x)), None)
            if c:
                out[name] = pd.to_numeric(df[c], errors="coerce")
        return out

    def _build_asof(self, df: pd.DataFrame):
        if df.empty:
            return
        season_col = _first_col(df.columns, ("season", "year"))
        week_col = _first_col(df.columns, ("week",))
        team_col = _first_col(df.columns, ("team", "team_name", "school"))
        if not all((season_col, week_col, team_col)):
            self.errors.append("adv_team: missing season/week/team keys")
            return
        vals = self._numeric_metrics(df)
        if vals.empty:
            self.errors.append("adv_team: expected advanced numeric columns were not found")
            return
        work = pd.DataFrame({
            "season": pd.to_numeric(df[season_col], errors="coerce"),
            "week": pd.to_numeric(df[week_col], errors="coerce"),
            "team_key": df[team_col].map(canon_team),
        })
        work = pd.concat([work, vals], axis=1).dropna(subset=["season", "week"])
        work["season"] = work["season"].astype(int); work["week"] = work["week"].astype(int)
        metrics = list(vals.columns)
        week_rows = work.groupby(["season", "week", "team_key"], as_index=False)[metrics].mean(numeric_only=True)
        week_rows = week_rows.sort_values(["season", "week", "team_key"])
        prior_final: dict[str, dict[str, float]] = {}
        for season in sorted(week_rows["season"].unique()):
            sub = week_rows[week_rows["season"] == season]
            teams = sorted(sub["team_key"].dropna().unique())
            season_final = {}
            for team in teams:
                t = sub[sub["team_key"] == team].sort_values("week")
                sums = {m: 0.0 for m in metrics}; counts = {m: 0 for m in metrics}
                carry = prior_final.get(team, {})
                for _, row in t.iterrows():
                    pre = {}
                    for m in metrics:
                        if counts[m] > 0:
                            pre[m] = sums[m] / counts[m]
                        elif m in carry and math.isfinite(carry[m]):
                            pre[m] = 0.62 * carry[m]
                        else:
                            pre[m] = np.nan
                    self.lookup[(int(season), int(row.week), team)] = pre
                    for m in metrics:
                        v = row.get(m)
                        if pd.notna(v) and math.isfinite(float(v)):
                            sums[m] += float(v); counts[m] += 1
                season_final[team] = {m: sums[m] / counts[m] if counts[m] else carry.get(m, np.nan) for m in metrics}
            prior_final.update(season_final)
        self.feature_names = metrics

    def _load_static(self):
        files = {
            "talent": "cfb_matchup_roster_talent.csv",
            "returning": "cfb_matchup_returning_production.csv",
            "continuity": "cfb_matchup_coach_continuity.csv",
        }
        wanted_tokens = ("talent", "rtprod", "return", "cont")
        for label, filename in files.items():
            try:
                df = _read_csv_cached(f"{RAW_BASE}/{filename}", self.cache / filename)
            except Exception as exc:
                self.errors.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            sc = _first_col(df.columns, ("season", "year"))
            tc = _first_col(df.columns, ("team", "school", "team_name"))
            if not sc or not tc:
                continue
            numeric = []
            for c in df.columns:
                lc = str(c).lower()
                if c in (sc, tc) or not any(t in lc for t in wanted_tokens):
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() >= max(5, int(len(df) * .1)):
                    numeric.append(c)
            for _, row in df.iterrows():
                try: key = (int(row[sc]), canon_team(row[tc]))
                except Exception: continue
                dest = self.static_lookup.setdefault(key, {})
                for c in numeric:
                    v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                    if pd.notna(v): dest[f"prior_{re.sub(r'[^a-z0-9]+','_',str(c).lower()).strip('_')}"] = float(v)
            if numeric:
                self.sources.append(f"SportsDataverse {filename}")

    def _build(self):
        self._build_asof(self._load_adv())
        self._load_static()

    def _team_values(self, season: int, week: int, team: str) -> dict[str, float]:
        key = canon_team(team)
        out = dict(self.lookup.get((int(season), int(week), key), {}))
        out.update(self.static_lookup.get((int(season), key), {}))
        return out

    def enrich(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if frame.empty:
            return frame.copy(), {"coverage": 0.0, "sources": self.sources, "errors": self.errors}
        out = frame.copy()
        all_names = set(self.feature_names)
        for vals in self.static_lookup.values(): all_names.update(vals.keys())
        hits = 0
        for idx, row in out.iterrows():
            hv = self._team_values(int(row.season), int(row.week), str(row.home_team))
            av = self._team_values(int(row.season), int(row.week), str(row.away_team))
            if hv or av: hits += 1
            for name in all_names:
                h, a = hv.get(name, np.nan), av.get(name, np.nan)
                out.at[idx, f"home_{name}"] = h
                out.at[idx, f"away_{name}"] = a
                if pd.notna(h) and pd.notna(a):
                    out.at[idx, f"diff_{name}"] = float(h) - float(a)
                    out.at[idx, f"avg_{name}"] = (float(h) + float(a)) / 2
                else:
                    out.at[idx, f"diff_{name}"] = np.nan
                    out.at[idx, f"avg_{name}"] = np.nan
        coverage = hits / len(out) if len(out) else 0.0
        return out, {"coverage": coverage, "sources": list(dict.fromkeys(self.sources)), "errors": self.errors, "feature_count": len(all_names)}
