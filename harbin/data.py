from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

SCHEDULE_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_{season}.csv"
ODDS_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/csv/cfb_line_odds.csv.gz"


@dataclass
class Game:
    game_id: str
    season: int
    week: int
    date: str
    away_id: str
    away_team: str
    home_id: str
    home_team: str
    away_score: float | None
    home_score: float | None
    completed: bool
    neutral_site: bool
    provider: str | None = None
    away_ml: float | None = None
    home_ml: float | None = None
    home_spread: float | None = None
    market_total: float | None = None


def _num(v):
    try:
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def _bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes"}


def _canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class SportsDataVerseClient:
    """Free GitHub-friendly CFB schedule/results and market loader."""

    BOOK_PRIORITY = ("pinnacle", "circa", "draftkings", "espn", "fanduel", "fanatics", "bovada")

    def __init__(self):
        self._seasons: dict[int, pd.DataFrame] = {}
        self._odds: pd.DataFrame | None = None
        self.odds_columns: list[str] = []

    def season_frame(self, season: int) -> pd.DataFrame:
        if season not in self._seasons:
            df = pd.read_csv(SCHEDULE_URL.format(season=season), low_memory=False)
            if {"home_division", "away_division"}.issubset(df.columns):
                fbs = df["home_division"].astype(str).str.lower().eq("fbs") & df["away_division"].astype(str).str.lower().eq("fbs")
                if fbs.any():
                    df = df.loc[fbs].copy()
            self._seasons[season] = df
        return self._seasons[season].copy()

    def detect(self) -> tuple[int, int]:
        now = pd.Timestamp.now(tz="UTC")
        season = now.year if now.month >= 7 else now.year - 1
        try:
            df = self.season_frame(season)
        except Exception:
            season -= 1
            df = self.season_frame(season)
        regular = df[df["season_type"].astype(str).str.lower().isin({"regular", "2"})].copy()
        regular["_date"] = pd.to_datetime(regular["start_date"], utc=True, errors="coerce")
        completed = regular["completed"].map(_bool)
        upcoming = regular[(~completed) & (regular["_date"] >= now - pd.Timedelta(days=1))]
        if len(upcoming):
            return season, int(upcoming.sort_values("_date").iloc[0]["week"])
        done = regular[completed]
        return season, int(done["week"].max()) if len(done) else 1

    def _row_game(self, r) -> Game:
        def sid(v, fallback):
            try:
                return str(int(v)) if not pd.isna(v) else str(fallback)
            except Exception:
                return str(fallback)
        return Game(
            game_id=str(r.get("game_id", "")),
            season=int(r.get("season", 0)),
            week=int(r.get("week", 0)),
            date=str(r.get("start_date", "")),
            away_id=sid(r.get("away_id"), r.get("away_team", "away")),
            away_team=str(r.get("away_team", "Away")),
            home_id=sid(r.get("home_id"), r.get("home_team", "home")),
            home_team=str(r.get("home_team", "Home")),
            away_score=_num(r.get("away_points")),
            home_score=_num(r.get("home_points")),
            completed=_bool(r.get("completed", False)),
            neutral_site=_bool(r.get("neutral_site", False)),
        )

    def history(self, start: int, target_season: int, target_week: int) -> list[Game]:
        games: list[Game] = []
        for season in range(start, target_season + 1):
            try:
                df = self.season_frame(season)
            except Exception:
                continue
            df = df[df["completed"].map(_bool)].copy()
            if season == target_season:
                df = df[df["week"].astype(int) < int(target_week)]
            for _, r in df.iterrows():
                g = self._row_game(r)
                if g.home_score is not None and g.away_score is not None:
                    games.append(g)
        return sorted(games, key=lambda g: (g.season, g.date, g.game_id))

    def week(self, season: int, week: int) -> list[Game]:
        df = self.season_frame(season)
        reg = df["season_type"].astype(str).str.lower().isin({"regular", "2"})
        rows = df[(df["week"].astype(int) == int(week)) & reg]
        games = [self._row_game(r) for _, r in rows.iterrows()]
        return self._attach_odds(games)

    def _load_odds(self) -> pd.DataFrame:
        if self._odds is None:
            try:
                self._odds = pd.read_csv(ODDS_URL, compression="gzip", low_memory=False)
            except Exception:
                self._odds = pd.DataFrame()
            self.odds_columns = list(self._odds.columns)
        return self._odds

    @staticmethod
    def _col(columns, exact=(), contains=()):
        cmap = {_canon(c): c for c in columns}
        for e in exact:
            if _canon(e) in cmap:
                return cmap[_canon(e)]
        for c in columns:
            cc = _canon(c)
            if any(all(token in cc for token in group) for group in contains):
                return c
        return None

    def _attach_odds(self, games: list[Game]) -> list[Game]:
        odds = self._load_odds()
        if odds.empty:
            return games
        id_col = self._col(odds.columns, exact=("game_id", "id"), contains=(("game", "id"),))
        if not id_col:
            return games
        wanted = {str(g.game_id) for g in games}
        o = odds[odds[id_col].astype(str).isin(wanted)].copy()
        if o.empty:
            return games

        provider_col = self._col(o.columns, exact=("provider", "sportsbook", "book"), contains=(("provider",), ("book",)))
        time_col = self._col(o.columns, exact=("updated", "updated_at", "timestamp", "created_at"), contains=(("update",), ("timestamp",), ("created", "at")))
        hml_col = self._col(o.columns, exact=("home_moneyline", "home_money_line", "home_ml"), contains=(("home", "money", "line"), ("home", "ml")))
        aml_col = self._col(o.columns, exact=("away_moneyline", "away_money_line", "away_ml"), contains=(("away", "money", "line"), ("away", "ml")))
        hsp_col = self._col(o.columns, exact=("home_spread", "spread"), contains=(("home", "spread"),))
        total_col = self._col(o.columns, exact=("over_under", "overunder", "total"), contains=(("over", "under"), ("total",)))

        if provider_col:
            def book_rank(v):
                s = str(v).lower()
                for i, name in enumerate(self.BOOK_PRIORITY):
                    if name in s:
                        return i
                return len(self.BOOK_PRIORITY)
            o["_book_rank"] = o[provider_col].map(book_rank)
        else:
            o["_book_rank"] = 999
        if time_col:
            o["_ts"] = pd.to_datetime(o[time_col], utc=True, errors="coerce")
        else:
            o["_ts"] = pd.NaT
        o = o.sort_values(["_book_rank", "_ts"], ascending=[True, False])
        chosen = o.groupby(o[id_col].astype(str), sort=False).head(1)
        by_id = {str(r[id_col]): r for _, r in chosen.iterrows()}

        for g in games:
            r = by_id.get(str(g.game_id))
            if r is None:
                continue
            g.home_ml = _num(r.get(hml_col)) if hml_col else None
            g.away_ml = _num(r.get(aml_col)) if aml_col else None
            g.home_spread = _num(r.get(hsp_col)) if hsp_col else None
            g.market_total = _num(r.get(total_col)) if total_col else None
            if provider_col and not pd.isna(r.get(provider_col)):
                g.provider = str(r.get(provider_col))
        return games
