from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BG=(15,17,19); ALT=(22,24,27); GRID=(39,42,46); TEXT=(240,241,242); MUTED=(128,132,136); BLUE=(76,132,224)
GREEN=(95,196,104); DARK_GREEN=(31,70,42); AMBER=(72,59,30); AMBER_TXT=(227,181,73); WARN=(214,164,75)


def _font(size,bold=False):
    candidates=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def _fmt_odds(v):
    if v is None or pd.isna(v): return "—"
    x=int(float(v)); return f"+{x}" if x>0 else str(x)


def _fmt_line(v):
    if v is None or pd.isna(v): return "—"
    return f"{float(v):+g}"


def _badge_html(x):
    if not x or str(x)=="nan": return ""
    return f'<span class="badge {str(x).lower()}">{x}</span>'


def _market_status_short(status):
    if status.startswith("PROJECTION-ONLY"): return "Projection-only · no verified live lines"
    if status.startswith("PARTIAL"): return status.split(" — ")[0].title()
    if status.startswith("LIVE"): return "Live market data"
    return status


def render_html(df,path,week,updated_at,market_status):
    css="""*{box-sizing:border-box}body{margin:0;background:#0f1113;color:#f0f1f2;font-family:Inter,Arial,sans-serif}.shell{max-width:1320px;margin:auto;padding:14px}.page{display:none}.head{display:flex;justify-content:space-between;align-items:flex-start}.title{font-size:24px;font-weight:800}.sub,.counter{font-size:12px;color:#8b8e92}.status{font-size:10px;color:#aeb1b5;margin-top:3px}.status.warn{color:#d6a44b}.counter{text-align:right;line-height:1.45}table{width:100%;border-collapse:collapse;table-layout:fixed;margin-top:8px}th{font-size:9px;letter-spacing:.08em;color:#777c81;text-align:left;padding:7px 8px;border-bottom:1px solid #272a2e}td{font-size:12px;padding:8px;border-bottom:1px solid #272a2e;height:36px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}tbody tr:nth-child(even){background:#16181b}.winner{font-weight:800}.at{color:#777c81}.proj{font-weight:800}.bar{display:inline-block;height:6px;background:#4c84e0;border-radius:5px;margin-right:6px;vertical-align:middle}.quiet{color:#64686d}.active{color:#f0f1f2;font-weight:700}.projtot{font-size:10px;color:#96999d}.badge{font-size:8px;font-weight:800;border-radius:4px;padding:3px 5px;margin-left:4px}.strong{background:#5fc468;color:#0c2c12}.bet{background:#1f462a;color:#63c76d}.lean{background:#483b1e;color:#e3b549}.nav{text-align:center;padding:14px}.nav button{background:#202328;border:1px solid #373b40;color:#eee;border-radius:5px;padding:6px 10px;margin:2px}.c1{width:23%}.c2{width:7%}.c3{width:9%}.c4{width:23%}.c5{width:23%}.c6{width:15%}@media(max-width:900px){.shell{overflow-x:auto}.page{min-width:1100px}}"""
    pages=[]; n=max(1,math.ceil(len(df)/14)); warn_class="warn" if market_status.startswith("PROJECTION-ONLY") else ""
    for p in range(1,n+1):
        start=(p-1)*14; chunk=df.iloc[start:start+14]; rows=[]
        for _,r in chunk.iterrows():
            aw,hm,winner=str(r.away_team),str(r.home_team),str(r.winner)
            matchup=f'<span class="winner">{aw}</span>' if aw==winner else aw
            matchup+=' <span class="at">@</span> '
            matchup+=f'<span class="winner">{hm}</span>' if hm==winner else hm
            ml='<span class="quiet">NO LINE</span>' if pd.isna(r.ml_odds) else f'{r.ml_team} {_fmt_odds(r.ml_odds)} {_badge_html(r.ml_badge)}'
            sp='<span class="quiet">NO LINE</span>' if pd.isna(r.spread_line) else f'{r.spread_team} {_fmt_line(r.spread_line)} {_badge_html(r.spread_badge)}'
            tot=f'<span class="quiet">NO LINE</span> <span class="projtot">proj {int(r.proj_total)}</span>' if pd.isna(r.market_total) else f'{r.total_dir} {float(r.market_total):g} <span class="projtot">proj {int(r.proj_total)}</span> {_badge_html(r.total_badge)}'
            bar=max(4,min(52,(int(r.win_pct)-50)*1.15))
            rows.append(f'<tr><td>{matchup}</td><td class="proj">{int(r.away_score)}–{int(r.home_score)}</td><td><span class="bar" style="width:{bar}px"></span>{int(r.win_pct)}%</td><td class="{"active" if r.ml_badge else "quiet"}">{ml}</td><td class="{"active" if r.spread_badge else "quiet"}">{sp}</td><td class="{"active" if r.total_badge else "quiet"}">{tot}</td></tr>')
        game_range=f"Games {start+1}–{min(start+14,len(df))} of {len(df)}" if len(df) else "0 games"
        pages.append(f'<section class="page" id="p{p}"><div class="head"><div><div class="title">CFB MODEL · WEEK {week} PICKS</div><div class="sub">Projected scores &amp; best bets · Updated {updated_at}</div><div class="status {warn_class}">{_market_status_short(market_status)}</div></div><div class="counter">{game_range}<br>{p} / {n}</div></div><table><colgroup><col class="c1"><col class="c2"><col class="c3"><col class="c4"><col class="c5"><col class="c6"></colgroup><thead><tr><th>MATCHUP (WINNER BOLD)</th><th>PROJ</th><th>WIN %</th><th>MONEYLINE</th><th>SPREAD</th><th>TOTAL</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>')
    nav=''.join(f'<button onclick="show({i})">{i}</button>' for i in range(1,n+1))
    doc=f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CFB Model Week {week}</title><style>{css}</style></head><body><a href="operations.html" style="display:block;padding:12px;color:#9bc8ff">NFL + CFB paper performance and release blockers</a><div class="shell">{"".join(pages)}<div class="nav">{nav}</div></div><script>function show(n){{document.querySelectorAll(".page").forEach((e,i)=>e.style.display=i===n-1?"block":"none")}}show(1)</script></body></html>'
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(doc)


def render_png(df,path,page,week,updated_at,market_status):
    W,H=1320,690; im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    f_title=_font(22,True); f_sub=_font(11); f_status=_font(9,True); f_head=_font(9,True); f_row=_font(12); f_bold=_font(12,True); f_small=_font(9); f_badge=_font(8,True)
    d.text((17,13),f"CFB MODEL · WEEK {week} PICKS",font=f_title,fill=TEXT)
    d.text((17,43),f"Projected scores & best bets · Updated {updated_at}",font=f_sub,fill=MUTED)
    d.text((17,59),_market_status_short(market_status),font=f_status,fill=WARN if market_status.startswith("PROJECTION-ONLY") else MUTED)
    start=(page-1)*14; end=min(start+14,len(df)); n=max(1,math.ceil(len(df)/14))
    d.text((1138,17),f"Games {start+1 if len(df) else 0}–{end} of {len(df)}",font=f_sub,fill=MUTED); d.text((1260,39),f"{page} / {n}",font=f_sub,fill=MUTED)
    xs=[17,305,385,500,790,1075]
    for x,h in zip(xs,["MATCHUP (WINNER BOLD)","PROJ","WIN %","MONEYLINE","SPREAD","TOTAL"]): d.text((x,78),h,font=f_head,fill=(119,124,129))
    d.line((16,96,1304,96),fill=GRID)
    def badge(x,y,label):
        if not label or str(label)=="nan": return 0
        label=str(label); tw=d.textbbox((0,0),label,font=f_badge)[2]; w=tw+12
        fill=GREEN if label=="STRONG" else DARK_GREEN if label=="BET" else AMBER; txt=(12,45,17) if label=="STRONG" else GREEN if label=="BET" else AMBER_TXT
        d.rounded_rectangle((x,y-2,x+w,y+13),radius=4,fill=fill); d.text((x+6,y+1),label,font=f_badge,fill=txt); return w+5
    def txtw(s,font): return d.textbbox((0,0),str(s),font=font)[2]
    for j,(_,r) in enumerate(df.iloc[start:end].iterrows()):
        y=97+j*41
        if j%2: d.rectangle((16,y,1304,y+40),fill=ALT)
        d.line((16,y,1304,y),fill=GRID); ty=y+13; aw,hm,win=str(r.away_team),str(r.home_team),str(r.winner)
        x=17; fa=f_bold if aw==win else f_row; fh=f_bold if hm==win else f_row
        d.text((x,ty),aw,font=fa,fill=TEXT); x+=txtw(aw,fa); d.text((x,ty)," @ ",font=f_row,fill=MUTED); x+=txtw(" @ ",f_row); d.text((x,ty),hm,font=fh,fill=TEXT)
        d.text((305,ty),f"{int(r.away_score)}–{int(r.home_score)}",font=f_bold,fill=TEXT)
        pct=int(r.win_pct); bw=max(4,min(46,int((pct-50)*1.05))); d.rounded_rectangle((385,ty+6,385+bw,ty+11),radius=3,fill=BLUE); d.text((385+bw+7,ty-1),f"{pct}%",font=f_row,fill=TEXT)
        if pd.isna(r.ml_odds): d.text((500,ty),"NO LINE",font=f_small,fill=(96,100,105))
        else:
            s=f"{r.ml_team} {_fmt_odds(r.ml_odds)}"; active=bool(r.ml_badge); ff=f_bold if active else f_row; d.text((500,ty),s,font=ff,fill=TEXT if active else (96,100,105)); badge(500+txtw(s,ff)+6,ty,str(r.ml_badge or ""))
        if pd.isna(r.spread_line): d.text((790,ty),"NO LINE",font=f_small,fill=(96,100,105))
        else:
            s=f"{r.spread_team} {_fmt_line(r.spread_line)}"; active=bool(r.spread_badge); ff=f_bold if active else f_row; d.text((790,ty),s,font=ff,fill=TEXT if active else (96,100,105)); badge(790+txtw(s,ff)+6,ty,str(r.spread_badge or ""))
        if pd.isna(r.market_total): d.text((1075,ty),"NO LINE",font=f_small,fill=(96,100,105)); d.text((1118,ty),f"proj {int(r.proj_total)}",font=f_small,fill=(143,147,151))
        else:
            s=f"{r.total_dir} {float(r.market_total):g}"; active=bool(r.total_badge); ff=f_bold if active else f_row; d.text((1075,ty),s,font=ff,fill=TEXT if active else (96,100,105)); x2=1075+txtw(s,ff)+6; ps=f"proj {int(r.proj_total)}"; d.text((x2,ty+2),ps,font=f_small,fill=(143,147,151)); x2+=txtw(ps,f_small)+5; badge(x2,ty,str(r.total_badge or ""))
    Path(path).parent.mkdir(parents=True,exist_ok=True); im.save(path)
