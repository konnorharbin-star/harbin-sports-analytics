from __future__ import annotations

import json, math, re, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
REPLICA_SIGMA = 16.41

@dataclass
class Game:
    game_id: str; season: int; week: int; season_type: int; date: str
    away_id: str; away_team: str; home_id: str; home_team: str
    away_score: float|None; home_score: float|None; completed: bool; neutral_site: bool
    provider: str|None=None; away_ml: float|None=None; home_ml: float|None=None
    market_spread_home: float|None=None; market_total: float|None=None

def _num(x):
    try: return None if x in (None, "") else float(x)
    except Exception: return None

class ESPNClient:
    def __init__(self, cache="cache/espn"):
        self.cache=Path(cache); self.cache.mkdir(parents=True,exist_ok=True)
        self.s=requests.Session(); self.s.headers.update({"User-Agent":"Mozilla/5.0 HarbinSportsAnalytics/1.0"})
    def fetch(self, season, week, st=2, force=False):
        p=self.cache/f"{season}_st{st}_w{week}.json"
        if p.exists() and not force: return json.loads(p.read_text())
        params={"limit":1000,"dates":season,"seasontype":st,"week":week,"groups":80}; err=None
        for i in range(4):
            try:
                r=self.s.get(ESPN,params=params,timeout=25); r.raise_for_status(); d=r.json(); p.write_text(json.dumps(d)); return d
            except Exception as e: err=e; time.sleep(1+i)
        raise RuntimeError(f"ESPN request failed: {err}")
    def current(self):
        r=self.s.get(ESPN,params={"limit":1000,"groups":80},timeout=25); r.raise_for_status(); return r.json()
    def detect(self):
        d=self.current(); ev=d.get("events") or []
        if ev:
            e=ev[0]; return int((e.get("season") or {}).get("year") or datetime.now().year), int((e.get("week") or {}).get("number") or 1), int((e.get("season") or {}).get("type") or 2)
        y=datetime.now().year; return (y if datetime.now().month>=7 else y-1),1,2
    def parse(self,e):
        comps=e.get("competitions") or []
        if not comps: return None
        c=comps[0]; teams={x.get("homeAway"):x for x in c.get("competitors") or []}
        if "home" not in teams or "away" not in teams: return None
        h,a=teams["home"],teams["away"]; ht,at=h.get("team") or {},a.get("team") or {}; status=((c.get("status") or {}).get("type") or {})
        odds=(c.get("odds") or [None])[0]; provider=away_ml=home_ml=spread=total=None
        if odds:
            provider=str((odds.get("provider") or {}).get("name") or "") or None; total=_num(odds.get("overUnder")); spread=_num(odds.get("spread")); ao,ho=odds.get("awayTeamOdds") or {},odds.get("homeTeamOdds") or {}; away_ml=_num(ao.get("moneyLine") or ao.get("moneyline")); home_ml=_num(ho.get("moneyLine") or ho.get("moneyline")); details=str(odds.get("details") or ""); m=re.match(r"^(.+?)\s+([+-]?\d+(?:\.\d+)?)$",details)
            if m:
                name,line=m.group(1).strip().lower(),_num(m.group(2)); hnames={str(ht.get(k) or "").lower() for k in ("abbreviation","displayName","shortDisplayName")}; anames={str(at.get(k) or "").lower() for k in ("abbreviation","displayName","shortDisplayName")}
                if line is not None:
                    if name in hnames: spread=float(line)
                    elif name in anames: spread=-float(line)
        so=e.get("season") or {}; wo=e.get("week") or {}
        return Game(str(e.get("id") or c.get("id") or ""), int(so.get("year") or 0), int(wo.get("number") or 0), int(so.get("type") or 2), str(e.get("date") or c.get("date") or ""), str(at.get("id") or ""), str(at.get("displayName") or at.get("name") or "Away"), str(ht.get("id") or ""), str(ht.get("displayName") or ht.get("name") or "Home"), _num(a.get("score")), _num(h.get("score")), bool(status.get("completed") or status.get("state")=="post"), bool(c.get("neutralSite")), provider, away_ml, home_ml, spread, total)
    def week(self,season,week,st=2,force=False): return [g for e in self.fetch(season,week,st,force).get("events") or [] if (g:=self.parse(e))]
    def history(self,start,end,target_season,target_week):
        out={}
        for y in range(start,end+1):
            for w in range(1,17):
                if y==target_season and w>=target_week: break
                try: gs=self.week(y,w,2)
                except Exception: continue
                for g in gs:
                    if g.completed and g.home_score is not None and g.away_score is not None: out[g.game_id]=g
            if y<target_season:
                for w in range(1,7):
                    try: gs=self.week(y,w,3)
                    except Exception: continue
                    for g in gs:
                        if g.completed and g.home_score is not None and g.away_score is not None: out[g.game_id]=g
        return sorted(out.values(),key=lambda g:(g.season,g.date,g.game_id))

class TeamState:
    def __init__(self): self.elo=1500.; self.pf=28.; self.pa=28.; self.margin=0.; self.total=56.; self.win=.5; self.games=0
    def regress(self): self.elo=.62*self.elo+.38*1500; self.pf=.5*self.pf+.5*28; self.pa=.5*self.pa+.5*28; self.margin*=.5; self.total=.5*self.total+.5*56; self.win=.5*self.win+.25; self.games=0

class RatingEngine:
    def __init__(self): self.s=defaultdict(TeamState); self.season=None
    def _newseason(self,y):
        if self.season is None: self.season=y
        elif y!=self.season:
            for st in self.s.values(): st.regress()
            self.season=y
    def feat(self,g):
        self._newseason(g.season); a,h=self.s[g.away_id],self.s[g.home_id]
        return {"game_id":g.game_id,"season":g.season,"week":g.week,"away_team":g.away_team,"home_team":g.home_team,"elo_diff_home":h.elo-a.elo,"pf_diff_home":h.pf-a.pf,"pa_diff_home":h.pa-a.pa,"margin_form_diff_home":h.margin-a.margin,"total_tendency":(h.total+a.total)/2,"win_diff_home":h.win-a.win,"games_diff_home":h.games-a.games,"home_games":h.games,"away_games":a.games,"neutral_site":int(g.neutral_site),"home_field":0 if g.neutral_site else 1}
    def update(self,g):
        a,h=self.s[g.away_id],self.s[g.home_id]; ap,hp=float(g.away_score),float(g.home_score); margin=hp-ap; total=hp+ap; exp_h=1/(1+10**(-((h.elo-a.elo)+(0 if g.neutral_site else 55))/400)); act=1 if margin>0 else .5 if margin==0 else 0; k=24; h.elo+=k*(act-exp_h); a.elo-=k*(act-exp_h); alpha=.24
        for st,pf,pa,m,w in ((h,hp,ap,margin,1 if margin>0 else 0),(a,ap,hp,-margin,1 if margin<0 else 0)):
            st.pf=(1-alpha)*st.pf+alpha*pf; st.pa=(1-alpha)*st.pa+alpha*pa; st.margin=(1-alpha)*st.margin+alpha*m; st.total=(1-alpha)*st.total+alpha*total; st.win=(1-alpha)*st.win+alpha*w; st.games+=1
    def train_frame(self,games):
        rows=[]
        for g in games:
            f=self.feat(g); f["target_margin_home"]=float(g.home_score)-float(g.away_score); f["target_total"]=float(g.home_score)+float(g.away_score); rows.append(f); self.update(g)
        return pd.DataFrame(rows)
    def upcoming(self,games): return pd.DataFrame([self.feat(g) for g in games])

NON={"game_id","season","week","away_team","home_team","target_margin_home","target_total"}
def features(df): return [c for c in df.columns if c not in NON and pd.api.types.is_numeric_dtype(df[c])]
def _fit(df,target,cols):
    X,y=df[cols],df[target]; ridge=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",Ridge(alpha=18.0))]); ridge.fit(X,y); boost=Pipeline([("imp",SimpleImputer(strategy="median")),("m",HistGradientBoostingRegressor(max_depth=3,learning_rate=.045,max_iter=260,l2_regularization=7,min_samples_leaf=25,random_state=17))]); boost.fit(X,y); return ridge,boost
def train(df):
    if len(df)<250: raise RuntimeError(f"Need at least 250 historical games; got {len(df)}")
    cols=features(df); d=df.sort_values(["season","week","game_id"]).reset_index(drop=True); cut=int(.8*len(d)); tr,va=d.iloc[:cut],d.iloc[cut:]; metrics={}; sig={}
    for t in ("target_margin_home","target_total"):
        r,b=_fit(tr,t,cols); p=.58*r.predict(va[cols])+.42*b.predict(va[cols]); res=va[t].to_numpy()-p; metrics[t+"_mae"]=float(mean_absolute_error(va[t],p)); metrics[t+"_rmse"]=float(mean_squared_error(va[t],p)**.5); sig[t]=float(max(6,np.std(res,ddof=1)))
    mr,mb=_fit(df,"target_margin_home",cols); trr,tb=_fit(df,"target_total",cols); return {"cols":cols,"mr":mr,"mb":mb,"tr":trr,"tb":tb,"metrics":metrics,"margin_sigma":sig["target_margin_home"],"total_sigma":sig["target_total"]}
def predict(bundle,X):
    z=X[bundle["cols"]]; m=.58*bundle["mr"].predict(z)+.42*bundle["mb"].predict(z); t=.58*bundle["tr"].predict(z)+.42*bundle["tb"].predict(z); return np.clip(m,-45,45),np.clip(t,28,90)
def cdf(x): return .5*(1+math.erf(x/math.sqrt(2)))
def implied(o): o=float(o); return abs(o)/(abs(o)+100) if o<0 else 100/(o+100)
def fair_american(p): return int(round(-100*p/(1-p))) if p>=.5 else int(round(100*(1-p)/p))
def roi(p,o): return p*(100/abs(o) if o<0 else o/100)-(1-p)
def ml_badge(p,o):
    e=100*(p-implied(o)); return ("STRONG" if e>=6 else "BET" if e>=3 else "LEAN" if e>=2 else ""),e
def spread_badge(m,sp):
    e=abs(float(m)-(-float(sp))); return ("STRONG" if e>=6 else "BET" if e>=4 else "LEAN" if e>=2 else ""),e
def total_badge(t,mt):
    e=abs(round(float(t))-float(mt)); return ("STRONG" if e>=7.5 else "BET" if e>=4.5 else "LEAN" if e>=2.5 else ""),e
def no_vig(a,h):
    pa,ph=implied(a),implied(h); s=pa+ph; return pa/s,ph/s

def prediction_rows(games,X,bundle):
    if X.empty: return pd.DataFrame()
    ms,ts=predict(bundle,X); market={g.game_id:g for g in games}; rows=[]
    for i,r in X.reset_index(drop=True).iterrows():
        g=market[str(r.game_id)]; m,t=float(ms[i]),float(ts[i]); ap=(t-m)/2; hp=(t+m)/2; winner=g.home_team if m>=0 else g.away_team; pwin=cdf(abs(m)/REPLICA_SIGMA); wp=int(round(100*pwin)); ml_odds=g.home_ml if m>=0 else g.away_ml; mtag=""; medge=mroi=fair=None
        if ml_odds is not None: mtag,medge=ml_badge(pwin,ml_odds); mroi=roi(pwin,ml_odds); fair=fair_american(pwin)
        spteam=spline=stag=sedge=None
        if g.market_spread_home is not None:
            home_edge=m-(-g.market_spread_home); spteam=g.home_team if home_edge>0 else g.away_team; spline=g.market_spread_home if home_edge>0 else -g.market_spread_home; stag,sedge=spread_badge(m,g.market_spread_home)
        tdir=ttag=tedge=None
        if g.market_total is not None: tdir="O" if t>g.market_total else "U"; ttag,tedge=total_badge(t,g.market_total)
        nv=None
        if g.away_ml is not None and g.home_ml is not None:
            pa,ph=no_vig(g.away_ml,g.home_ml); nv=ph if m>=0 else pa
        rows.append({"game_id":g.game_id,"season":g.season,"week":g.week,"date":g.date,"away_team":g.away_team,"home_team":g.home_team,"away_score":int(round(ap)),"home_score":int(round(hp)),"model_margin_home":m,"model_total":t,"winner":winner,"win_probability":pwin,"win_pct":wp,"provider":g.provider,"ml_team":winner,"ml_odds":ml_odds,"ml_badge":mtag,"ml_edge_pp":medge,"ml_est_roi":mroi,"ml_fair_odds":fair,"no_vig_market_prob_winner":nv,"spread_team":spteam,"spread_line":spline,"spread_badge":stag,"spread_edge_pts":sedge,"market_spread_home":g.market_spread_home,"market_total":g.market_total,"total_dir":tdir,"proj_total":int(round(t)),"total_badge":ttag,"total_edge_pts":tedge})
    return pd.DataFrame(rows)

STYLE="""body{margin:0;background:#101214;color:#f2f2f2;font-family:Arial,Helvetica,sans-serif}.wrap{padding:16px}.page{max-width:1288px;margin:auto}.top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px}h1{font-size:28px;margin:0 0 6px}.sub,.counter{color:#a9abad;font-size:15px}.counter{text-align:right;line-height:1.5}table{width:100%;border-collapse:collapse;table-layout:fixed}th{text-align:left;color:#85898d;font-size:11px;padding:8px 8px}td{border-top:1px solid #272a2e;padding:11px 8px;font-size:15px;height:28px}tbody tr:nth-child(even){background:#17191c}.at{color:#7d8084}.proj{font-weight:700}.bar{display:inline-block;height:7px;background:#4f86e2;border-radius:5px;margin-right:8px;vertical-align:middle}.market{color:#63676b}.market.active{color:#f2f2f2;font-weight:700}.badge{font-size:10px;padding:5px 8px;border-radius:5px;margin-left:6px}.strong{background:#62c76d;color:#103515}.bet{background:#214a2c;color:#61c66d}.lean{background:#483b1f;color:#e5b950}.projonly{font-size:12px;color:#96999d}.nolines{color:#63676b}.c1{width:23%}.c2{width:7%}.c3{width:9%}.c4{width:23%}.c5{width:23%}.c6{width:15%}.nav{text-align:center;padding:18px}.nav button{background:#23262a;color:#eee;border:1px solid #3a3d42;border-radius:6px;padding:7px 12px;margin:3px}b{color:#fff}"""
def _fmt_o(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    x=int(float(v)); return f"+{x}" if x>0 else str(x)
def _fmt_l(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    return f"{float(v):+g}"
def _badge(x): return "" if not x or str(x)=="nan" else f'<span class="badge {str(x).lower()}">{x}</span>'
def render_html(df,path,week,date):
    path=Path(path); pages=[]; n=max(1,math.ceil(len(df)/14))
    for p,start in enumerate(range(0,len(df),14),1):
        rows=[]
        for _,r in df.iloc[start:start+14].iterrows():
            aw,hm=str(r.away_team),str(r.home_team); a,h=int(r.away_score),int(r.home_score); matchup=(f"<b>{aw}</b>" if a>h else aw)+" <span class='at'>@</span> "+(f"<b>{hm}</b>" if h>a else hm); win=int(r.win_pct); bar=max(4,min(78,(win-50)*1.55)); ml=f"{r.ml_team} {_fmt_o(r.ml_odds)} {_badge(r.ml_badge)}"; sp="<span class='nolines'>NO LINE</span>" if pd.isna(r.spread_line) else f"{r.spread_team} {_fmt_l(r.spread_line)} {_badge(r.spread_badge)}"; tt=f"<span class='nolines'>NO LINE</span> <span class='projonly'>proj {int(r.proj_total)}</span>" if pd.isna(r.market_total) else f"{r.total_dir} {float(r.market_total):g} <span class='projonly'>proj {int(r.proj_total)}</span> {_badge(r.total_badge)}"; rows.append(f"<tr><td>{matchup}</td><td class='proj'>{a}&ndash;{h}</td><td><span class='bar' style='width:{bar}px'></span>{win}%</td><td class='market {'active' if r.ml_badge else ''}'>{ml}</td><td class='market {'active' if r.spread_badge else ''}'>{sp}</td><td class='market {'active' if r.total_badge else ''}'>{tt}</td></tr>")
        pages.append(f"<section class='page' id='page-{p}'><div class='top'><div><h1>CFB MODEL · WEEK {week} PICKS</h1><div class='sub'>Projected scores &amp; best bets · {date}</div></div><div class='counter'>Games {start+1}–{min(start+14,len(df))} of {len(df)}<br>{p} / {n}</div></div><table><colgroup><col class='c1'><col class='c2'><col class='c3'><col class='c4'><col class='c5'><col class='c6'></colgroup><thead><tr><th>MATCHUP (WINNER BOLD)</th><th>PROJ</th><th>WIN %</th><th>MONEYLINE</th><th>SPREAD</th><th>TOTAL</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>")
    if not pages: pages=[f"<section class='page'><h1>CFB MODEL · WEEK {week} PICKS</h1><div class='sub'>No upcoming games returned.</div></section>"]
    nav="".join(f"<button onclick='showPage({i})'>{i}</button>" for i in range(1,n+1)); doc=f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>{STYLE}</style></head><body><div class='wrap'>{''.join(pages)}</div><div class='nav'>{nav}</div><script>function showPage(n){{document.querySelectorAll('.page').forEach((x,i)=>x.style.display=i===n-1?'block':'none')}}showPage(1)</script></body></html>"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(doc)
def render_png(df,path,page,week,date):
    W,H=1320,690; img=Image.new("RGB",(W,H),(16,18,20)); d=ImageDraw.Draw(img)
    def font(n,b=False):
        p="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; return ImageFont.truetype(p,n) if Path(p).exists() else ImageFont.load_default()
    def text(x,y,s,n=15,b=False,fill=(242,242,242)): d.text((x,y),str(s),font=font(n,b),fill=fill)
    text(17,14,f"CFB MODEL · WEEK {week} PICKS",27,True); text(17,51,f"Projected scores & best bets · {date}",15,False,(169,171,173)); start=(page-1)*14; end=min(start+14,len(df)); text(1120,18,f"Games {start+1}–{end} of {len(df)}",15,False,(169,171,173)); text(1240,42,f"{page} / {max(1,math.ceil(len(df)/14))}",15,False,(169,171,173)); xs=[20,310,390,505,800,1095]
    for x,h in zip(xs,["MATCHUP (WINNER BOLD)","PROJ","WIN %","MONEYLINE","SPREAD","TOTAL"]): text(x,78,h,11,True,(133,137,141))
    for i,(_,r) in enumerate(df.iloc[start:end].iterrows()):
        y=100+i*42
        if i%2: d.rectangle((16,y,1304,y+41),fill=(23,25,28))
        d.line((16,y,1304,y),fill=(39,42,46)); a,h=int(r.away_score),int(r.home_score); aw,hm=str(r.away_team),str(r.home_team); text(20,y+11,f"{aw} @ {hm}",15,True); text(310,y+11,f"{a}–{h}",16,True); win=int(r.win_pct); bw=max(4,min(54,int((win-50)*1.1))); d.rounded_rectangle((390,y+18,390+bw,y+25),radius=4,fill=(79,134,226)); text(390+bw+10,y+10,f"{win}%"); text(505,y+11,f"{r.ml_team} {_fmt_o(r.ml_odds)} {r.ml_badge or ''}",14,bool(r.ml_badge)); text(800,y+11,"NO LINE" if pd.isna(r.spread_line) else f"{r.spread_team} {_fmt_l(r.spread_line)} {r.spread_badge or ''}",14,bool(r.spread_badge)); text(1095,y+11,"NO LINE" if pd.isna(r.market_total) else f"{r.total_dir} {float(r.market_total):g} {r.total_badge or ''}",14,bool(r.total_badge))
    Path(path).parent.mkdir(parents=True,exist_ok=True); img.save(path)
def run(season=None,week=None,history_start=None,root="."):
    root=Path(root); out=root/"outputs"; docs=root/"docs"; hist=root/"history"; out.mkdir(exist_ok=True); docs.mkdir(exist_ok=True); hist.mkdir(exist_ok=True); c=ESPNClient(root/"cache/espn")
    if season is None or week is None:
        ds,dw,_=c.detect(); season=season or ds; week=week or dw
    history_start=history_start or max(2018,season-4); games=c.history(history_start,season,season,week); eng=RatingEngine(); train_df=eng.train_frame(games); bundle=train(train_df); upcoming=[g for g in c.week(season,week,2,force=True) if not g.completed]; X=eng.upcoming(upcoming); pred=prediction_rows(upcoming,X,bundle); stamp=datetime.now().astimezone().isoformat(); base=f"cfb_model_{season}_week{week}"; pred.to_csv(out/f"{base}.csv",index=False); (out/f"{base}.json").write_text(pred.to_json(orient="records",indent=2)); meta={"season":season,"week":week,"history_start":history_start,"historical_games":len(games),"training_rows":len(train_df),"upcoming_games":len(upcoming),"metrics":bundle["metrics"],"replica_margin_sigma":REPLICA_SIGMA,"calibrated_margin_sigma":bundle["margin_sigma"],"generated_at":stamp,"source":"ESPN public scoreboard JSON"}; (out/f"{base}_metadata.json").write_text(json.dumps(meta,indent=2)); date=stamp[:10]; render_html(pred,out/f"{base}.html",week,date); pages=max(1,math.ceil(len(pred)/14)); [render_png(pred,out/f"{base}_page{p}.png",p,week,date) for p in range(1,pages+1)]; render_html(pred,docs/"index.html",week,date); (docs/"latest.json").write_text(pred.to_json(orient="records",indent=2)); (docs/"metadata.json").write_text(json.dumps(meta,indent=2)); (docs/".nojekyll").write_text(""); snap=pred.copy(); snap.insert(0,"snapshot_at",stamp); hp=hist/"prediction_snapshots.csv"; old=pd.read_csv(hp) if hp.exists() else pd.DataFrame(); pd.concat([old,snap],ignore_index=True).to_csv(hp,index=False); mp=hist/"run_metadata.jsonl"; mp.write_text((mp.read_text() if mp.exists() else "")+json.dumps(meta,separators=(",",":"))+"\n"); return pred,meta
