#!/usr/bin/env python3
"""
Build a self-contained HTML viewer from annotations.csv (Plotly.js from a pinned CDN, thumbnails embedded as base64).

    python build_viewer.py --annotations <dir>/annotations.csv [--max-points 3000]

Explore tab: embedding scatter (colored by make / color / domain / correctness), hover to see the image and its
annotations in the side panel, filters for domain / make / color / confidence / needs-review.
Review tab: lowest-confidence rows as a thumbnail grid.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_crop, parse_bbox  # noqa: E402

PLOTLY_URL = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"


def thumb(path, domain, bbox, side):
    try:
        im = load_crop(path, domain, bbox, max_side=side)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def stratified(df, n, seed):
    """Up to n rows with coordinates, spread evenly over domain x make."""
    df = df.dropna(subset=["x", "y"])
    if len(df) <= n:
        return df
    rng = np.random.RandomState(seed)
    groups = [g.index.values.copy() for _, g in df.groupby(["domain", "gt_make"])]
    for g in groups:
        rng.shuffle(g)
    picked, i = [], 0
    while len(picked) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(picked) < n:
                picked.append(g[i])
        i += 1
    return df.loc[picked]


def record(r, with_thumb, side):
    def f(k, nd=3):
        v = r.get(k)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else (round(float(v), nd) if isinstance(v, (float, np.floating)) else v)
    d = {
        "id": int(r["image_id"]), "domain": r["domain"], "x": f("x", 4), "y": f("y", 4),
        "gt": r["gt_class"], "gt_make": r["gt_make"],
        "make": r["pred_make"], "make_conf": f("make_conf"),
        "model": r["pred_model"], "model_conf": f("pred_model_conf"), "top3": r["top3"],
        "color": f("color"), "color_conf": f("color_conf"),
        "ok_make": bool(r["correct_make"]), "ok_model": bool(r["correct_model"]), "review": bool(r["needs_review"]),
    }
    if with_thumb:
        d["img"] = thumb(r["path"], r["domain"], parse_bbox(r), side)
    return d


def summary(df):
    s = {"n": int(len(df)), "domains": {}}
    for dom, g in list(df.groupby("domain")) + [("all", df)]:
        e = {"n": int(len(g)), "make_acc": round(float(g["correct_make"].mean()), 4),
             "model_acc": round(float(g["correct_model"].mean()), 4),
             "mean_model_conf": round(float(g["pred_model_conf"].mean()), 4),
             "review": int(g["needs_review"].sum())}
        if "color" in g and g["color"].notna().any():
            e["colors"] = {k: int(v) for k, v in g["color"].value_counts().items()}
            e["mean_color_conf"] = round(float(g["color_conf"].mean()), 4)
        s["domains"][dom] = e
    return s


def build(df, out_path, max_points=3000, review_n=120, seed=42):
    df = df.reset_index(drop=True)
    if "color" not in df.columns:
        df["color"] = np.nan; df["color_conf"] = np.nan
    if "x" not in df.columns:
        df["x"] = np.nan; df["y"] = np.nan
    pts = stratified(df, max_points, seed)
    conf = df["pred_model_conf"].copy()
    if df["color_conf"].notna().any():
        conf = np.minimum(conf, df["color_conf"].fillna(1.0))
    rev = df.loc[conf.sort_values().index[:review_n]]

    payload = {
        "summary": summary(df),
        "points": [record(r, True, 144) for r in pts.to_dict("records")],
        "review": [record(r, True, 200) for r in rev.to_dict("records")],
        "has_color": bool(df["color"].notna().any()),
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__PLOTLY__", PLOTLY_URL).replace("__DATA__", data)
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    print(f"[viewer] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(payload['points']):,} points, "
          f"{len(payload['review'])} review rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-points", type=int, default=3000)
    ap.add_argument("--review-n", type=int, default=120)
    a = ap.parse_args()
    out = a.output or str(Path(a.annotations).with_name("viewer.html"))
    build(pd.read_csv(a.annotations), out, a.max_points, a.review_n)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vehicle Annotations</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--text:#1c1917;--muted:#78716c;--line:#e7e5e4;--accent:#2563eb;--bad:#dc2626;--good:#16a34a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--panel:#1f1e1d;--text:#f5f5f4;--muted:#a8a29e;--line:#34312f;--accent:#60a5fa;--bad:#f87171;--good:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
header{padding:14px 16px 4px}h1{font-size:18px;margin:0 0 6px}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:12px;color:var(--muted)}
.stat b{display:block;color:var(--text);font-size:14px}
nav{display:flex;gap:4px;padding:0 16px;border-bottom:1px solid var(--line)}
nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--muted);padding:8px 12px;font:inherit;cursor:pointer}
nav button.on{color:var(--text);border-color:var(--accent)}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px}
.bar label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:2px}
select,input[type=range]{font:inherit;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:3px}
.main{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:12px;padding:0 16px 16px}
@media (max-width:800px){.main{grid-template-columns:1fr}}
#plot{height:70vh;min-height:420px;background:var(--panel);border:1px solid var(--line);border-radius:8px}
#side{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;align-self:start;position:sticky;top:8px}
#side img{width:100%;border-radius:6px;background:var(--line)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin-top:8px;font-size:13px}.kv span:nth-child(odd){color:var(--muted)}
.ok{color:var(--good)}.no{color:var(--bad)}.muted{color:var(--muted)}
#review{display:none;padding:12px 16px;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px;font-size:12px}.card img{width:100%;border-radius:4px}
.hide{display:none}
</style></head><body>
<header><h1>Vehicle annotations</h1><div class="stats" id="stats"></div></header>
<nav><button id="t1" class="on">Explore</button><button id="t2">Review low-confidence</button></nav>
<div id="explore">
 <div class="bar">
  <label>Color by<select id="by"><option value="gt_make">Ground-truth make</option><option value="make">Predicted make</option><option value="color">Color</option><option value="domain">Domain</option><option value="ok_model">Model correct</option></select></label>
  <label>Domain<select id="fd"><option value="">all</option></select></label>
  <label>Make (predicted)<select id="fm"><option value="">all</option></select></label>
  <label id="lc">Color<select id="fc"><option value="">all</option></select></label>
  <label>Min model confidence <span id="cv">0.00</span><input id="fconf" type="range" min="0" max="1" step="0.05" value="0"></label>
  <label><span>Needs review</span><input id="frev" type="checkbox"></label>
  <span class="muted" id="count"></span>
 </div>
 <div class="main"><div id="plot"></div>
  <aside id="side"><div class="muted">Hover a point to see the image and its annotations.</div></aside></div>
</div>
<div id="review"></div>
<script src="__PLOTLY__"></script>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const COLORS={white:"#f5f5f4",black:"#1c1917",silver:"#c0c0c0",gray:"#78716c",red:"#dc2626",blue:"#2563eb",green:"#16a34a",yellow:"#facc15",orange:"#f97316",brown:"#78350f",beige:"#d6c3a1",gold:"#ca8a04",other:"#a855f7"};
const uniq=(a)=>[...new Set(a.filter(v=>v!=null))].sort();
function hue(i,n){return `hsl(${Math.round(360*i/Math.max(n,1))},62%,52%)`}
// summary
const S=D.summary.domains;$("stats").innerHTML=Object.entries(S).map(([k,v])=>`<div class="stat"><b>${k}: ${v.n.toLocaleString()}</b>make ${(100*v.make_acc).toFixed(1)}% · model ${(100*v.model_acc).toFixed(1)}% · conf ${v.mean_model_conf.toFixed(2)} · review ${v.review.toLocaleString()}${v.mean_color_conf!=null?` · color conf ${v.mean_color_conf.toFixed(2)}`:""}</div>`).join("");
// filters
for(const d of uniq(D.points.map(p=>p.domain)))$("fd").add(new Option(d,d));
for(const m of uniq(D.points.map(p=>p.make)))$("fm").add(new Option(m,m));
if(D.has_color){for(const c of uniq(D.points.map(p=>p.color)))$("fc").add(new Option(c,c))}else{$("lc").classList.add("hide");$("by").querySelector("[value=color]").remove()}
function filtered(){const d=$("fd").value,m=$("fm").value,c=$("fc").value,mc=+$("fconf").value,r=$("frev").checked;
 return D.points.filter(p=>p.x!=null&&(!d||p.domain==d)&&(!m||p.make==m)&&(!c||p.color==c)&&(p.model_conf>=mc)&&(!r||p.review))}
function draw(){const by=$("by").value,pts=filtered();$("cv").textContent=(+$("fconf").value).toFixed(2);
 $("count").textContent=`${pts.length.toLocaleString()} of ${D.points.length.toLocaleString()} points`;
 const key=p=>by=="ok_model"?(p.ok_model?"correct":"wrong"):p[by];
 const groups={};
 pts.forEach(p=>{const k=String(key(p));(groups[k]=groups[k]||[]).push(p)});
 const names=Object.keys(groups).sort();
 const traces=names.map((k,i)=>{const g=groups[k];
  return {type:"scattergl",mode:"markers",name:k,x:g.map(p=>p.x),y:g.map(p=>p.y),customdata:g.map(p=>p.id),hoverinfo:"none",
   marker:{size:6,color:by=="ok_model"?(k=="correct"?"#16a34a":"#dc2626"):by=="color"?COLORS[k]||"#999":hue(i,names.length),line:{width:by=="color"?0.6:0,color:"#888"}}}});
 const dark=matchMedia("(prefers-color-scheme: dark)").matches;const fg=dark?"#f5f5f4":"#1c1917";
 Plotly.react("plot",traces,{margin:{l:10,r:10,t:10,b:10},xaxis:{visible:false},yaxis:{visible:false},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:fg},legend:{itemsizing:"constant"},hovermode:"closest",dragmode:"pan"},{responsive:true,scrollZoom:true,displaylogo:false});
 $("plot").removeAllListeners&&$("plot").removeAllListeners("plotly_hover");
 $("plot").on("plotly_hover",e=>show(byId[e.points[0].customdata]));}
const byId={};D.points.forEach(p=>byId[p.id]=p);D.review.forEach(p=>byId[p.id]=byId[p.id]||p);
function mark(ok){return `<span class="${ok?"ok":"no"}">${ok?"✓":"✗"}</span>`}
function show(p){$("side").innerHTML=`${p.img?`<img src="${p.img}">`:""}<div class="kv">
 <span>domain</span><span>${p.domain}</span>
 <span>true class</span><span>${p.gt}</span>
 <span>make</span><span>${p.make} ${mark(p.ok_make)} <span class="muted">${(100*p.make_conf).toFixed(0)}%</span></span>
 <span>model</span><span>${p.model} ${mark(p.ok_model)} <span class="muted">${(100*p.model_conf).toFixed(0)}%</span></span>
 ${p.color?`<span>color</span><span>${p.color} <span class="muted">${(100*p.color_conf).toFixed(0)}%</span></span>`:""}
 <span>top 3</span><span class="muted">${p.top3.split("|").join("<br>")}</span>
 ${p.review?`<span>flag</span><span class="no">needs review</span>`:""}</div>`}
["by","fd","fm","fc","fconf","frev"].forEach(id=>$(id).addEventListener("input",draw));
// review tab
$("review").innerHTML=D.review.map(p=>`<div class="card">${p.img?`<img src="${p.img}">`:""}<div><b>${p.make}</b> ${mark(p.ok_make)} ${(100*p.make_conf).toFixed(0)}%<br>${p.model} ${mark(p.ok_model)} ${(100*p.model_conf).toFixed(0)}%${p.color?`<br>color: ${p.color} ${(100*p.color_conf).toFixed(0)}%`:""}<br><span class="muted">true: ${p.gt}</span></div></div>`).join("");
function tab(n){$("explore").style.display=n==1?"":"none";$("review").style.display=n==2?"grid":"none";$("t1").classList.toggle("on",n==1);$("t2").classList.toggle("on",n==2);if(n==1)Plotly.Plots.resize("plot")}
$("t1").onclick=()=>tab(1);$("t2").onclick=()=>tab(2);
draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
