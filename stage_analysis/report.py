"""
Self-contained interactive HTML report (no server, no external assets).

Tabs: Stocks (choose stage view, theme, sector), Themes, Sectors, ETFs.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from .scanner import VIEWS, ScanResult

STOCK_COLS = [
    "ticker", "substage", "transition_score", "weeks_in_stage", "close", "pct_from_ma30w",
    "ma30w_slope_pct", "tr_dist_to_pivot_pct", "tr_pivot", "mansfield_rs", "RS_1M", "RS_3M",
    "adr20_pct", "vol30", "tma_daily", "tma_weekly", "consol_days", "consol_weeks",
    "bk_squeeze_daily", "bk_squeeze_weekly", "obv_rising_20d", "macd_v", "pmo_rising",
    "variant1", "variant2", "passes_filters", "instrument", "sector", "themes",
    "1W", "1M", "3M", "6M", "YTD", "1Y", "stage",
]
LABELS = {
    "ticker": "Ticker", "substage": "Stage", "transition_score": "1→2 Score",
    "weeks_in_stage": "Wks in Stage", "close": "Close", "pct_from_ma30w": "% vs 30W",
    "ma30w_slope_pct": "30W Slope %", "tr_dist_to_pivot_pct": "% to Pivot", "tr_pivot": "Pivot",
    "mansfield_rs": "Mansfield RS", "RS_1M": "RS 1M", "RS_3M": "RS 3M", "adr20_pct": "ADR20 %",
    "vol30": "Vol(30)", "tma_daily": "TMA Daily", "tma_weekly": "TMA Weekly",
    "consol_days": "Consol D", "consol_weeks": "Consol W", "bk_squeeze_daily": "Sqz D",
    "bk_squeeze_weekly": "Sqz W", "obv_rising_20d": "OBV↑", "macd_v": "MACD-V",
    "pmo_rising": "PMO↑", "variant1": "V1", "variant2": "V2", "passes_filters": "Filters",
    "instrument": "Vehicle", "sector": "Sector", "themes": "Themes",
}
GROUP_COLS = ["N", "1W", "1M", "3M", "6M", "YTD", "1Y", "RS_1M", "RS_3M", "Rotation", "Score",
              "%S1", "%S2", "%S3", "%S4", "%Above30W", "1->2 Cands"]


def _records(df: pd.DataFrame, cols, index_name: str):
    if df is None or df.empty:
        return []
    d = df.reset_index().rename(columns={"index": index_name})
    keep = [c for c in dict.fromkeys([index_name] + list(cols)) if c in d.columns]
    out = []
    for row in d[keep].to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            clean[k] = v
        out.append(clean)
    return out


def build_html(res: ScanResult, title: str = "Stage Analysis Scanner", default_view: str = "1to2") -> str:
    payload = {
        "asOf": res.as_of.strftime("%Y-%m-%d"),
        "views": {k: v[0] for k, v in VIEWS.items()},
        "viewOrder": list(VIEWS),
        "defaultView": default_view,
        "labels": LABELS,
        "stocks": _records(res.stocks, STOCK_COLS, "ticker"),
        "themes": _records(res.theme_baskets, GROUP_COLS, "Group"),
        "sectors": _records(res.sector_baskets, GROUP_COLS, "Group"),
        "sectorEtfs": _records(res.sector_etfs, ["Group"] + GROUP_COLS, "ETF"),
        "themeEtfs": _records(res.theme_etfs, ["Group"] + GROUP_COLS, "ETF"),
    }
    data = json.dumps(payload).replace("</", "<\\/")
    return TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


def write_html(res: ScanResult, path: str, **kw) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(res, **kw))
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--accent:#2f5bd3;
--pos:#1a7f43;--neg:#c0392b;--pos-bg:rgba(26,127,67,.14);--neg-bg:rgba(192,57,43,.14);
--s1:#8a6d00;--s2:#1a7f43;--s3:#b35c00;--s4:#c0392b;}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#a19f97;
--line:#33322e;--accent:#7c9cff;--pos:#4cc483;--neg:#ff7a6b;--pos-bg:rgba(76,196,131,.16);
--neg-bg:rgba(255,122,107,.16);--s1:#e0bf3a;--s2:#4cc483;--s3:#ffa24d;--s4:#ff7a6b;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:20px 16px 8px;max-width:1500px;margin:auto}h1{font-size:20px;margin:0}
.sub{color:var(--muted);font-size:13px}main{max-width:1500px;margin:auto;padding:0 16px 40px}
.tabs,.views{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);
padding:6px 12px;border-radius:999px;cursor:pointer}button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
select,input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:8px 0 12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer}
.card b{display:block;font-size:20px}.card span{color:var(--muted);font-size:12px}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}
th{position:sticky;top:0;background:var(--panel);cursor:pointer;font-size:12px;color:var(--muted);user-select:none}
td.l,th.l{text-align:left}tr:hover td{background:rgba(127,127,127,.06)}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.st1{color:var(--s1);border:1px solid var(--s1)}.st2{color:var(--s2);border:1px solid var(--s2)}
.st3{color:var(--s3);border:1px solid var(--s3)}.st4{color:var(--s4);border:1px solid var(--s4)}
.hot{background:var(--pos-bg)}.cold{background:var(--neg-bg)}.desc{color:var(--muted);margin:4px 0 8px}
.note{color:var(--muted);font-size:12px;margin-top:10px}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub" id="asof"></div></header>
<main>
<div class="tabs" id="tabs"></div>
<section id="pane"></section>
<p class="note">Stages: Weinstein, weekly bars, 30-week SMA. 1→2 score (0-100) = base length, MA flattening,
price vs MA, distance to pivot, 10/20/40W convergence, BB/KC squeeze, OBV, Mansfield RS, volume expansion.
V1/V2 = your TMA lateral-consolidation variants with ADR/volume/price filters. Not investment advice.</p>
</main>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const L = D.labels;
let tab = "stocks", view = D.defaultView, sortKey = null, sortDir = -1, theme = "", sector = "", q = "";
$("#asof").textContent = "Data as of " + D.asOf + " · " + D.stocks.length + " stocks scanned";

const fmt = (k, v) => {
  if (v === null || v === undefined || v === "") return "";
  if (typeof v === "boolean") return v ? "✓" : "";
  if (typeof v !== "number") return v;
  if (k === "vol30") return v >= 1e6 ? (v/1e6).toFixed(1)+"M" : (v/1e3).toFixed(0)+"K";
  if (["weeks_in_stage","consol_days","consol_weeks","N","1->2 Cands"].includes(k)) return v.toFixed(0);
  return v.toFixed(Math.abs(v) >= 100 ? 0 : (Math.abs(v) >= 10 ? 1 : 2));
};
const signed = k => /^(1W|1M|3M|6M|YTD|1Y|RS_.*|pct_from_ma30w|ma30w_slope_pct|mansfield_rs|macd_v)$/.test(k);

const VIEWFN = {
  "1to2": r => r.substage === "1->2 Transition" || r.substage === "2A Breakout",
  "1": r => r.stage === 1, "2": r => r.stage === 2, "2a": r => (r.substage||"").startsWith("2A"),
  "3": r => r.stage === 3, "4": r => r.stage === 4, "v1": r => r.variant1, "v2": r => r.variant2,
  "all": r => true,
};

function table(rows, cols, leftCols, cellClass) {
  if (!rows.length) return '<div class="desc" style="padding:12px">No rows.</div>';
  const sorted = sortKey && cols.includes(sortKey) ? [...rows].sort((a,b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
    return (x > y ? 1 : -1) * sortDir; }) : rows;
  let h = '<div class="wrap"><table><thead><tr>' + cols.map(c =>
    `<th class="${leftCols.includes(c)?"l":""}" data-k="${c}">${L[c]||c}${sortKey===c?(sortDir>0?" ▲":" ▼"):""}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of sorted) {
    h += "<tr>" + cols.map(c => {
      const v = r[c]; let cls = leftCols.includes(c) ? "l" : "";
      if (signed(c) && typeof v === "number") cls += v > 0 ? " pos" : (v < 0 ? " neg" : "");
      if (cellClass) cls += " " + (cellClass(c, v, r) || "");
      let txt = fmt(c, v);
      if (c === "substage" && v) txt = `<span class="tag st${r.stage}">${v}</span>`;
      if (c === "Rotation" && v) txt = `<span class="tag st${{Leading:2,Improving:1,Weakening:3,Lagging:4}[v]}">${v}</span>`;
      return `<td class="${cls}">${txt}</td>`; }).join("") + "</tr>";
  }
  return h + "</tbody></table></div>";
}

function bindSort() {
  document.querySelectorAll("th[data-k]").forEach(th => th.onclick = () => {
    const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : -1; sortKey = k; render(); });
}

function stocksPane() {
  const themes = [...new Set(D.stocks.flatMap(r => (r.themes||"").split(", ").filter(Boolean)))].sort();
  const sectors = [...new Set(D.stocks.map(r => r.sector))].sort();
  const base = D.stocks.filter(r => (!theme || (r.themes||"").split(", ").includes(theme))
                                && (!sector || r.sector === sector)
                                && (!q || r.ticker.toLowerCase().includes(q.toLowerCase())));
  const counts = Object.fromEntries(Object.keys(VIEWFN).map(k => [k, base.filter(VIEWFN[k]).length]));
  const rows = base.filter(VIEWFN[view]);
  if (!sortKey) { sortKey = (view === "1to2" || view === "1") ? "transition_score" : "RS_3M"; sortDir = -1; }
  const cols = ["ticker","substage","transition_score","weeks_in_stage","close","pct_from_ma30w","ma30w_slope_pct",
    "tr_dist_to_pivot_pct","mansfield_rs","RS_1M","RS_3M","adr20_pct","vol30","tma_daily","tma_weekly",
    "consol_days","consol_weeks","bk_squeeze_daily","bk_squeeze_weekly","obv_rising_20d","pmo_rising",
    "variant1","variant2","passes_filters","instrument","sector","themes"];
  return `<div class="cards">${D.viewOrder.map(k => [k, D.views[k]]).map(([k,d]) =>
      `<div class="card" data-v="${k}" style="${k===view?"outline:2px solid var(--accent)":""}"><b>${counts[k]}</b><span>${d}</span></div>`).join("")}</div>
    <div class="bar">
      <select id="th"><option value="">All themes</option>${themes.map(t=>`<option ${t===theme?"selected":""}>${t}</option>`).join("")}</select>
      <select id="se"><option value="">All sectors</option>${sectors.map(t=>`<option ${t===sector?"selected":""}>${t}</option>`).join("")}</select>
      <input id="q" placeholder="Ticker…" value="${q}">
    </div>
    <div class="desc">${D.views[view]} — ${rows.length} stocks</div>` +
    table(rows, cols, ["ticker","substage","tma_daily","tma_weekly","instrument","sector","themes"]);
}

const heat = (c, v) => (/^(1W|1M|3M|6M|YTD|1Y)$/.test(c) && typeof v === "number")
  ? (v > 0 ? "hot" : (v < 0 ? "cold" : "")) : "";

function groupPane(rows, keyCol, title, extra) {
  const cols = [keyCol, ...(extra||[]), "N","1W","1M","3M","6M","YTD","1Y","RS_1M","RS_3M","Rotation","Score",
                "%S1","%S2","%S3","%S4","%Above30W","1->2 Cands"].filter(c => rows.some(r => r[c] !== undefined));
  if (!sortKey || !cols.includes(sortKey)) { sortKey = "Score"; sortDir = -1; }
  return `<div class="desc">${title}</div>` + table(rows, cols, [keyCol, "Group", "Rotation"], heat);
}

const TABS = {
  stocks: ["Stocks by stage", stocksPane],
  themes: ["Themes", () => groupPane(D.themes, "Group",
    "Thematic baskets — median member return, RS vs SPY, rotation quadrant and stage breadth. Click a theme to see its stocks.")],
  sectors: ["Sectors", () => groupPane(D.sectors, "Group",
    "Sector baskets built from scanned stocks — median return and stage breadth.")],
  etfs: ["Sector & Theme ETFs", () => groupPane(D.sectorEtfs, "ETF", "Sector SPDR ETFs vs SPY", ["Group"])
        + "<br>" + groupPane(D.themeEtfs, "ETF", "Industry / thematic ETFs vs SPY", ["Group"])],
};

function render() {
  $("#tabs").innerHTML = Object.entries(TABS).map(([k,[n]]) => `<button class="${k===tab?"on":""}" data-t="${k}">${n}</button>`).join("");
  $("#pane").innerHTML = TABS[tab][1]();
  document.querySelectorAll("[data-t]").forEach(b => b.onclick = () => { tab = b.dataset.t; sortKey = null; render(); });
  document.querySelectorAll(".card[data-v]").forEach(c => c.onclick = () => { view = c.dataset.v; sortKey = null; render(); });
  const th = $("#th"), se = $("#se"), qi = $("#q");
  if (th) th.onchange = () => { theme = th.value; render(); };
  if (se) se.onchange = () => { sector = se.value; render(); };
  if (qi) qi.oninput = () => { q = qi.value; render(); const n = $("#q"); n.focus(); n.setSelectionRange(q.length, q.length); };
  if (tab === "themes") document.querySelectorAll("#pane tbody tr").forEach(tr => {
    tr.style.cursor = "pointer"; tr.onclick = () => { theme = tr.cells[0].textContent; tab = "stocks"; view = "all"; sortKey = null; render(); }; });
  bindSort();
}
render();
</script></body></html>
"""
