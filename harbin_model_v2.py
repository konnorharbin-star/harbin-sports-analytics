from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from harbin_model import (
    Game, RatingEngine, train, prediction_rows, render_html, render_png,
    REPLICA_SIGMA,
)

SCHEDULE_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_{season}.csv"
ODDS_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/csv/cfb_line_odds.csv.gz"


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "t", "yes"}


def _num(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


class SportsDataVerseClient:
    """GitHub-friendly CFB data client.

    Primary source is sportsdataverse/cfbfastR-data on raw.githubusercontent.com.
    This avoids ESPN blocking GitHub-hosted runners with HTTP 403.
    """

    def __init__(self):
        self._season_cache: dict[int, pd.DataFrame] = {}
        self._odds = None

    def season_frame(self, season: int) -> pd.DataFrame:
        if season not in self._season_cache:
            url = SCHEDULE_URL.format(season=season)
            df = pd.read_csv(url)
            # Keep FBS-v-FBS when division information is available.
            if {"home_division", "away_division"}.issubset(df.columns):
                mask = (
                    df["home_division"].astype(str).str.lower().eq("fbs")
                    & df["away_division"].astype(str).str.lower().eq("fbs")
                )
                if mask.any():
                    df = df.loc[mask].copy()
            self._season_cache[season] = df
        return self._season_cache[season].copy()

    def detect(self):
        now = pd.Timestamp.now(tz="UTC")
        season = now.year if now.month >= 7 else now.year - 1
        try:
            df = self.season_frame(season)
        except Exception:
            season -= 1
            df = self.season_frame(season)

        regular = df[df["season_type"].astype(str).str.lower().isin({"regular", "2"})].copy()
        regular["_date"] = pd.to_datetime(regular["start_date"], utc=True, errors="coerce")
        completed = regular["completed"].map(_as_bool)
        upcoming = regular[(~completed) & (regular["_date"] >= now - pd.Timedelta(days=1))]
        if len(upcoming):
            week = int(upcoming.sort_values("_date").iloc[0]["week"])
        else:
            done = regular[completed]
            week = int(done["week"].max()) if len(done) else 1
        return season, week, 2

    def _row_to_game(self, r) -> Game:
        st_raw = str(r.get("season_type", "regular")).lower()
        st = 3 if "post" in st_raw or st_raw == "3" else 2
        completed = _as_bool(r.get("completed", False))
        g = Game(
            str(r.get("game_id", "")),
            int(r.get("season", 0)),
            int(r.get("week", 0)),
            st,
            str(r.get("start_date", "")),
            str(int(r.get("away_id"))) if not pd.isna(r.get("away_id")) else str(r.get("away_team", "")),
            str(r.get("away_team", "Away")),
            str(int(r.get("home_id"))) if not pd.isna(r.get("home_id")) else str(r.get("home_team", "")),
            str(r.get("home_team", "Home")),
            _num(r.get("away_points")),
            _num(r.get("home_points")),
            completed,
            _as_bool(r.get("neutral_site", False)),
            "sportsdataverse",
            None,
            None,
            None,
            None,
        )
        return g

    def _load_odds(self):
        if self._odds is not None:
            return self._odds
        try:
            self._odds = pd.read_csv(ODDS_URL, compression="gzip", low_memory=False)
        except Exception:
            self._odds = pd.DataFrame()
        return self._odds

    @staticmethod
    def _pick_col(cols, candidates):
        lower = {c.lower(): c for c in cols}
        for name in candidates:
            if name.lower() in lower:
                return lower[name.lower()]
        for c in cols:
            lc = c.lower()
            if all(tok in lc for tok in candidates[0].lower().split("_")):
                return c
        return None

    def _attach_odds(self, games: list[Game]) -> list[Game]:
        odds = self._load_odds()
        if odds.empty or "game_id" not in odds.columns:
            return games

        wanted = {str(g.game_id) for g in games}
        o = odds[odds["game_id"].astype(str).isin(wanted)].copy()
        if o.empty:
            return games

        # Prefer the latest timestamped quote when the dataset contains repeated snapshots.
        ts_col = self._pick_col(o.columns, ["updated", "timestamp", "created_at", "date"])
        if ts_col:
            o["_ts"] = pd.to_datetime(o[ts_col], utc=True, errors="coerce")
            o = o.sort_values("_ts")
        o = o.groupby(o["game_id"].astype(str), as_index=False).tail(1)

        home_ml = self._pick_col(o.columns, ["home_moneyline", "home_money_line", "home_ml"])
        away_ml = self._pick_col(o.columns, ["away_moneyline", "away_money_line", "away_ml"])
        total = self._pick_col(o.columns, ["over_under", "overunder", "total"])
        spread = self._pick_col(o.columns, ["spread", "home_spread", "spread_line"])
        provider = self._pick_col(o.columns, ["provider", "sportsbook", "book"])

        by_id = {str(r["game_id"]): r for _, r in o.iterrows()}
        for g in games:
            r = by_id.get(str(g.game_id))
            if r is None:
                continue
            g.home_ml = _num(r.get(home_ml)) if home_ml else None
            g.away_ml = _num(r.get(away_ml)) if away_ml else None
            g.market_total = _num(r.get(total)) if total else None
            g.market_spread_home = _num(r.get(spread)) if spread else None
            if provider:
                v = r.get(provider)
                if not pd.isna(v):
                    g.provider = str(v)
        return games

    def week(self, season: int, week: int, st: int = 2, force: bool = False):
        df = self.season_frame(season)
        if st == 2:
            mask_type = df["season_type"].astype(str).str.lower().isin({"regular", "2"})
        else:
            mask_type = df["season_type"].astype(str).str.lower().isin({"postseason", "post", "3"})
        rows = df[(df["week"].astype(int) == int(week)) & mask_type]
        games = [self._row_to_game(r) for _, r in rows.iterrows()]
        return self._attach_odds(games)

    def history(self, start: int, end: int, target_season: int, target_week: int):
        out: list[Game] = []
        for season in range(start, end + 1):
            try:
                df = self.season_frame(season)
            except Exception:
                continue
            completed = df["completed"].map(_as_bool)
            df = df.loc[completed].copy()
            if season == target_season:
                df = df[df["week"].astype(int) < int(target_week)]
            for _, r in df.iterrows():
                g = self._row_to_game(r)
                if g.home_score is not None and g.away_score is not None:
                    out.append(g)
        return sorted(out, key=lambda g: (g.season, g.date, g.game_id))


def run(season=None, week=None, history_start=None, root="."):
    root = Path(root)
    out = root / "outputs"
    docs = root / "docs"
    hist = root / "history"
    out.mkdir(exist_ok=True)
    docs.mkdir(exist_ok=True)
    hist.mkdir(exist_ok=True)

    client = SportsDataVerseClient()
    if season is None or week is None:
        ds, dw, _ = client.detect()
        season = season or ds
        week = week or dw

    history_start = history_start or max(2018, season - 4)
    games = client.history(history_start, season, season, week)
    engine = RatingEngine()
    train_df = engine.train_frame(games)
    bundle = train(train_df)

    upcoming = [g for g in client.week(season, week, 2, force=True) if not g.completed]
    X = engine.upcoming(upcoming)
    pred = prediction_rows(upcoming, X, bundle)

    stamp = datetime.now(timezone.utc).astimezone().isoformat()
    base = f"cfb_model_{season}_week{week}"
    pred.to_csv(out / f"{base}.csv", index=False)
    (out / f"{base}.json").write_text(pred.to_json(orient="records", indent=2))

    meta = {
        "season": season,
        "week": week,
        "history_start": history_start,
        "historical_games": len(games),
        "training_rows": len(train_df),
        "upcoming_games": len(upcoming),
        "metrics": bundle["metrics"],
        "replica_margin_sigma": REPLICA_SIGMA,
        "calibrated_margin_sigma": bundle["margin_sigma"],
        "generated_at": stamp,
        "source": "sportsdataverse/cfbfastR-data GitHub datasets",
    }
    (out / f"{base}_metadata.json").write_text(json.dumps(meta, indent=2))

    date = stamp[:10]
    render_html(pred, out / f"{base}.html", week, date)
    pages = max(1, math.ceil(len(pred) / 14))
    for p in range(1, pages + 1):
        render_png(pred, out / f"{base}_page{p}.png", p, week, date)

    render_html(pred, docs / "index.html", week, date)
    (docs / "latest.json").write_text(pred.to_json(orient="records", indent=2))
    (docs / "metadata.json").write_text(json.dumps(meta, indent=2))
    (docs / ".nojekyll").write_text("")

    snap = pred.copy()
    if len(snap):
        snap.insert(0, "snapshot_at", stamp)
        hp = hist / "prediction_snapshots.csv"
        old = pd.read_csv(hp) if hp.exists() else pd.DataFrame()
        pd.concat([old, snap], ignore_index=True).to_csv(hp, index=False)

    mp = hist / "run_metadata.jsonl"
    mp.write_text((mp.read_text() if mp.exists() else "") + json.dumps(meta, separators=(",", ":")) + "\n")
    return pred, meta
