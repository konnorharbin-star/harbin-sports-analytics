from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import pandas as pd

from .data import Game


@dataclass
class TeamState:
    elo: float = 1500.0
    offense: float = 0.0
    defense: float = 0.0
    margin_form: float = 0.0
    total_form: float = 56.0
    win_rate: float = 0.5
    volatility: float = 14.0
    sos_elo: float = 1500.0
    games: int = 0

    def offseason_regress(self):
        self.elo = 1500 + 0.68 * (self.elo - 1500)
        self.offense *= 0.58; self.defense *= 0.58; self.margin_form *= 0.45
        self.total_form = 56 + 0.48 * (self.total_form - 56)
        self.win_rate = 0.5 + 0.45 * (self.win_rate - 0.5)
        self.volatility = 14 + 0.6 * (self.volatility - 14)
        self.sos_elo = 1500 + 0.5 * (self.sos_elo - 1500)
        self.games = 0


class OpponentAdjustedRatings:
    """Leakage-safe pregame ratings using only information available before kickoff."""
    HOME_POINTS = 2.4

    def __init__(self):
        self.state = defaultdict(TeamState); self.season = None; self.league_ppg = 28.0

    def _season(self, season):
        if self.season is None: self.season = season
        elif season != self.season:
            for s in self.state.values(): s.offseason_regress()
            self.league_ppg = 28 + 0.5 * (self.league_ppg - 28); self.season = season

    def features(self, g: Game) -> dict:
        self._season(g.season); a, h = self.state[g.away_id], self.state[g.home_id]
        hfa = 0.0 if g.neutral_site else self.HOME_POINTS
        bh = self.league_ppg + h.offense - a.defense + hfa; ba = self.league_ppg + a.offense - h.defense
        return {
            "game_id": g.game_id, "season": g.season, "week": g.week, "date": g.date,
            "away_id": str(g.away_id), "home_id": str(g.home_id),
            "away_team": g.away_team, "home_team": g.home_team,
            "neutral_site": int(g.neutral_site), "home_field": int(not g.neutral_site),
            "league_ppg": self.league_ppg, "baseline_home_points": bh, "baseline_away_points": ba,
            "baseline_margin": bh-ba, "baseline_total": bh+ba, "elo_diff_home": h.elo-a.elo,
            "offense_diff_home": h.offense-a.offense, "defense_diff_home": h.defense-a.defense,
            "net_eff_diff_home": (h.offense+h.defense)-(a.offense+a.defense),
            "margin_form_diff_home": h.margin_form-a.margin_form, "total_form_avg": (h.total_form+a.total_form)/2,
            "win_rate_diff_home": h.win_rate-a.win_rate, "volatility_avg": (h.volatility+a.volatility)/2,
            "sos_diff_home": h.sos_elo-a.sos_elo, "home_games": h.games, "away_games": a.games,
            "games_diff_home": h.games-a.games, "early_season": int(g.week <= 4),
        }

    def update(self, g: Game):
        self._season(g.season); a, h = self.state[g.away_id], self.state[g.home_id]
        hp, ap = float(g.home_score), float(g.away_score); hfa = 0.0 if g.neutral_site else self.HOME_POINTS
        pred_h = self.league_ppg + h.offense - a.defense + hfa; pred_a = self.league_ppg + a.offense - h.defense
        rh, ra = hp-pred_h, ap-pred_a; ae=.095
        h.offense += ae*rh; a.defense -= ae*rh; a.offense += ae*ra; h.defense -= ae*ra
        margin,total=hp-ap,hp+ap; af=.22; oh,oa=h.margin_form,a.margin_form
        h.margin_form=(1-af)*h.margin_form+af*margin; a.margin_form=(1-af)*a.margin_form-af*margin
        h.total_form=(1-af)*h.total_form+af*total; a.total_form=(1-af)*a.total_form+af*total
        h.volatility=(1-af)*h.volatility+af*abs(margin-oh); a.volatility=(1-af)*a.volatility+af*abs(-margin-oa)
        h.win_rate=(1-af)*h.win_rate+af*(1. if margin>0 else 0.); a.win_rate=(1-af)*a.win_rate+af*(1. if margin<0 else 0.)
        eh,ea=h.elo,a.elo; h.sos_elo=(1-af)*h.sos_elo+af*ea; a.sos_elo=(1-af)*a.sos_elo+af*eh
        exp=1/(1+10**(-((eh-ea)+(0 if g.neutral_site else 55))/400)); actual=1. if margin>0 else .5 if margin==0 else 0.
        delta=22*min(1.8,1+math.log1p(abs(margin))/5)*(actual-exp); h.elo+=delta; a.elo-=delta
        self.league_ppg=.997*self.league_ppg+.003*((hp+ap)/2); h.games+=1; a.games+=1

    def training_frame(self,games:list[Game])->pd.DataFrame:
        rows=[]
        for g in games:
            f=self.features(g); f["target_margin_home"]=float(g.home_score)-float(g.away_score); f["target_total"]=float(g.home_score)+float(g.away_score); rows.append(f); self.update(g)
        return pd.DataFrame(rows)

    def upcoming_frame(self,games:list[Game])->pd.DataFrame:
        return pd.DataFrame([self.features(g) for g in games])
