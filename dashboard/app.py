"""ATC decision-support dashboard: airport capacity (AAR) and delay-risk forecasts, 3-12 h ahead.

Replay mode: pick any "issue time" in the held-out 2025 test period.  The forecasts shown are exactly what the
models would have produced at that time (trained only on data before 2025); switch on "reveal outcome" to
compare with what happened.   Run:  streamlit run dashboard/app.py
"""
import json
import os
import sys
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import advisory, config as C  # noqa: E402
from src import train_delay as TD  # noqa: E402

st.set_page_config(page_title="Capacity & Risk", layout="wide", page_icon="✈")

# ------------------------------------------------------------------ look: graphite glass, one blue accent
ACCENT, OK, AMBER, STOP, MUTED = "#4c9dff", "#5fd38d", "#ffb547", "#ff6b5e", "#8a929c"
TEXT2 = "rgba(240,244,250,.74)"
GRID = "rgba(255,255,255,.07)"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --sand:#f5f7fa; --sand-2:rgba(240,244,250,.74); --sand-3:rgba(232,238,246,.5); --accent:#4c9dff; --ok:#5fd38d;
        --amber:#ffb547; --stop:#ff6b5e; --glass:rgba(255,255,255,.075); --line:rgba(255,255,255,.13);
        --mono:"IBM Plex Mono",ui-monospace,"SF Mono",monospace;
        --body:"SF Pro Text",-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif; }
html, body, [class*="css"], .stApp, .stMarkdown, button, input { font-family: var(--body); letter-spacing:-.005em; }
.stApp { background:
    radial-gradient(38% 46% at 82% 8%, rgba(214,228,245,.20), transparent 70%),
    radial-gradient(30% 26% at 8% 0%, rgba(190,205,225,.12), transparent 70%),
    radial-gradient(70% 40% at 50% 110%, rgba(76,157,255,.10), transparent 70%), #23272d; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1320px; }
[data-testid="stSidebar"] { background: rgba(40,44,51,.72); backdrop-filter: blur(40px); border-right:1px solid var(--line); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: var(--sand-2); }
.mono { font-family: var(--mono); font-feature-settings:"zero"; letter-spacing:0; }

.topbar { display:flex; flex-wrap:wrap; align-items:center; gap:6px 18px; white-space:nowrap; color:var(--sand-2); font-size:14px; margin-bottom:22px; }
.topbar .brand { color:var(--sand); font-weight:600; font-size:16px; display:flex; align-items:center; gap:10px; }
.topbar .ring { width:14px; height:14px; border-radius:50%; border:2px solid var(--accent); box-shadow:0 0 12px rgba(76,157,255,.6); }
.topbar .dot { width:7px; height:7px; border-radius:50%; background:var(--ok); display:inline-block; margin-right:6px; }
.topbar .right { margin-left:auto; display:flex; gap:22px; }

.hero h1 { font-size:44px; font-weight:650; letter-spacing:-.02em; margin:0 0 4px; color:var(--sand); }
.hero p { color:var(--sand-2); margin:0 0 18px; font-size:16px; }

.glass { background:var(--glass); border:1px solid var(--line); border-radius:22px; backdrop-filter:blur(40px);
         box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 20px 50px rgba(0,0,0,.18); }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin:4px 0 18px; }
.tile { padding:14px 18px 12px; min-width:min(150px,100%); flex:1; }
.tile .num { font-size:34px; font-weight:600; line-height:1.05; color:var(--sand); letter-spacing:-.02em; }
.unit { font-size:14px; color:var(--sand-3); margin-left:4px; font-weight:400; }
.tile .lbl { color:var(--sand-3); font-size:13px; margin-top:4px; }
.tile.warn .num { color:var(--amber); } .tile.stop .num { color:var(--stop); } .tile.good .num { color:var(--ok); }

.alert { display:flex; gap:14px; align-items:flex-start; padding:16px 20px; margin:0 0 16px; }
.alert .bar { width:4px; align-self:stretch; border-radius:4px; }
.alert .t { font-weight:600; color:var(--sand); } .alert .s { color:var(--sand-2); font-size:14px; }
.alert.severe .bar { background:var(--stop); box-shadow:0 0 14px rgba(255,107,94,.6); }
.alert.moderate .bar { background:var(--amber); box-shadow:0 0 14px rgba(255,181,71,.5); }
.alert.clear .bar { background:var(--ok); box-shadow:0 0 14px rgba(95,211,141,.45); }
.alert ul { margin:6px 0 0 0; padding-left:18px; color:var(--sand-2); font-size:14px; }

.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(360px,100%),1fr)); gap:14px; margin-bottom:14px; }
.card { padding:18px 22px 16px; }
.card .ctitle { color:var(--sand-2); font-size:14px; margin-bottom:6px; }
.card .big { font-size:46px; font-weight:600; letter-spacing:-.025em; color:var(--sand); line-height:1.05; margin-bottom:10px; }
.card .big .unit { font-size:20px; }
.crow { display:grid; grid-template-columns:minmax(80px,150px) 1fr 56px; gap:12px; align-items:center; font-size:13px; color:var(--sand-2); margin:6px 0; }
.crow .track { height:6px; border-radius:6px; background:rgba(255,255,255,.08); overflow:hidden; }
.crow .fill { height:100%; border-radius:6px; background:#9aa3ae; }
.crow .fill.ok { background:var(--ok); } .crow .fill.accent { background:var(--accent); }
.crow .fill.stop { background:var(--stop); } .crow .fill.amber { background:var(--amber); }
.crow .v { text-align:right; color:var(--sand); }
.card .note { color:var(--sand-3); font-size:12.5px; margin-top:10px; }
.section { color:var(--sand); font-weight:600; font-size:15px; margin:10px 0 6px; }
.caption { color:var(--sand-3); font-size:13px; }

/* tabs as a floating pill bar */
[data-testid="stTabs"] [role="tablist"] { gap:4px; background:rgba(58,63,72,.62); border:1px solid var(--line) !important;
    border-radius:999px; padding:5px; width:fit-content; margin:6px auto 18px; backdrop-filter:blur(40px); box-shadow:0 12px 30px rgba(0,0,0,.25); }
[data-testid="stTabs"] [role="tablist"] > :not([role="tab"]) { display:none; }
[data-testid="stTab"] { border-radius:999px; padding:8px 18px; height:auto; color:var(--sand-2); background:transparent;
    border:none !important; box-shadow:none !important; }
[data-testid="stTab"] p { font-size:14px; }
[data-testid="stTab"][aria-selected="true"] { background:rgba(255,255,255,.14); color:var(--sand); }
[data-testid="stTab"]:hover { color:var(--sand); }
[data-testid="stTab"] .react-aria-SelectionIndicator { display:none; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background:var(--glass); border:1px solid var(--line); border-radius:22px; padding:10px 12px; }
</style>
""", unsafe_allow_html=True)

pio.templates["glass"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2, size=12),
    colorway=[ACCENT, OK, AMBER, STOP, "#c5ccd5"], title=dict(font=dict(color="#f5f7fa", size=14), x=0.01),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(255,255,255,.14)"),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(255,255,255,.14)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=44, b=16, l=10, r=10),
    hoverlabel=dict(bgcolor="#3a3f47", bordercolor="rgba(255,255,255,.2)", font=dict(color="#f5f7fa"))))
pio.templates.default = "plotly_dark+glass"


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def chart(fig, where=st):
    """Streamlit injects its theme colours into Plotly, so set the glass backgrounds on the figure itself."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    where.plotly_chart(fig, width="stretch", theme=None)


def tile(value, unit, label, tone=""):
    return f'<div class="glass tile {tone}"><div class="num">{value}<span class="unit mono">{unit}</span></div><div class="lbl">{label}</div></div>'


def compare(title, value, unit, rows, note=""):
    """Big headline number plus one bar per model/baseline (bar length relative to the largest value)."""
    rows = [r for r in rows if pd.notna(r[1])]
    top = max((r[1] for r in rows), default=1) or 1
    bars = "".join(f'<div class="crow"><span>{escape(lbl)}</span><div class="track"><div class="fill {tone}" '
                   f'style="width:{max(2, 100 * v / top):.0f}%"></div></div><span class="v mono">{shown}</span></div>'
                   for lbl, v, shown, tone in rows)
    return (f'<div class="glass card"><div class="ctitle">{title}</div><div class="big">{value}<span class="unit mono">{unit}</span>'
            f'</div>{bars}<div class="note">{note}</div></div>')


def alert(kind, title, sub="", items=()):
    li = "".join(f"<li>{escape(i)}</li>" for i in items)
    return (f'<div class="glass alert {kind}"><div class="bar"></div><div><div class="t">{title}</div>'
            f'<div class="s">{sub}</div>{f"<ul>{li}</ul>" if li else ""}</div></div>')


NAMES = {
    "carrier": "airline", "other_airport": "route", "airport": "hub", "direction": "arrival/departure",
    "hour_local": "time of day", "hour_sin": "time of day", "hour_cos": "time of day", "doy_sin": "season", "doy_cos": "season",
    "month": "month", "dow": "day of week", "distance_mi": "distance", "sched_load_sample": "schedule density",
    "rec_dep_mean_24h": "hub departure delays (24 h)", "rec_arr_mean_24h": "inbound delays (24 h)",
    "rec_dep_mean_6h": "hub departure delays (6 h)", "rec_arr_mean_6h": "inbound delays (6 h)",
    "rec_dep_severe_24h": "recent severe departure delays", "rec_arr_severe_24h": "recent severe arrival delays",
    "aar_ratio_min_6h": "recent capacity dip", "aar_ratio_min_12h": "recent capacity dip", "aar_ratio_now": "current capacity",
    "aar_ratio_mean_24h": "capacity last 24 h", "taf_aar_ratio": "TAF-forecast capacity", "taf_aar_worst_ratio": "TAF worst-case capacity",
    "taf_aar": "TAF-forecast capacity", "taf_aar_worst": "TAF worst-case capacity", "temp_c": "temperature",
    "bts_scheduled_departures": "airport departures scheduled", "bts_scheduled_arrivals": "airport arrivals scheduled",
    "bts_departures_previous_3h": "airport departures last 3 h", "bts_arrivals_previous_3h": "airport arrivals last 3 h",
    "bts_severe_departure_rate_previous_3h": "airport severe delays last 3 h",
}


def api_key():
    """ANTHROPIC_API_KEY from the environment, else from a git-ignored .env (repo root or jho/aerocast_data)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    for env in (ROOT / ".env", ROOT / "jho" / "aerocast_data" / ".env"):
        if not key and env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
    return key or None


def claude_explain(context):
    """Plain-English rephrasing of numbers already on screen. The LLM never forecasts; the models do."""
    body = {"model": "claude-haiku-5-5", "max_tokens": 600,
            "system": "You explain an airport capacity and delay-risk outlook to an operations duty manager. Use ONLY the "
                      "numbers given, do not invent or change any. Do not give orders; the human decides. Plain English, "
                      "at most 4 sentences, no em dashes.",
            "messages": [{"role": "user", "content": json.dumps(context, default=str)}]}
    r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=30,
                      headers={"x-api-key": api_key(), "anthropic-version": "2023-06-01", "content-type": "application/json"})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text").strip()


def nice(f):
    if f in NAMES:
        return NAMES[f]
    if f.startswith("taf_"):
        return "TAF: " + f[4:].replace("_", " ")
    return f.replace("_", " ")


@st.cache_data(show_spinner=False)
def load():
    hourly = pd.read_csv(C.PROC / "hourly_weather.csv", parse_dates=["hour_utc"])
    aar = pd.read_parquet(C.PROC / "aar_predictions.parquet")
    aar = aar[aar["split"] == "test"]
    dl = pd.read_parquet(C.PROC / "delay_test_predictions.parquet")
    rep = {p.stem: pd.read_csv(p) for p in C.REPORTS.glob("*.csv")}
    return hourly, aar, dl, rep


@st.cache_resource(show_spinner="Loading delay models…")
def load_delay_models():
    t = TD.prep(pd.read_parquet(C.PROC / "delay_table.parquet"))
    t = t[t["split"] == "test"].set_index(["flight_id", "direction", "airport", "lead_h"], drop=False)
    models = {l: joblib.load(C.MODELS / f"delay_lead{l}.joblib") for l in range(3, 13)}
    return t, models


hourly, aar_pred, dl_pred, rep = load()
icaos = sorted(aar_pred["airport"].unique())

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### Replay controls")
qp = st.query_params          # deep link, e.g. ?airport=KDFW&t0=2025-02-18T17
airport = st.sidebar.selectbox("Airport", icaos, index=icaos.index(qp["airport"]) if qp.get("airport") in icaos else 0,
                               format_func=lambda a: f"{C.AIRPORTS[a]['iata']} ({a})")
cfg = C.AIRPORTS[airport]
h_ap = hourly[hourly["airport"] == airport].set_index("hour_utc")
test_h = h_ap[(h_ap.index >= pd.Timestamp("2025-01-01", tz="UTC")) & h_ap["aar_ratio"].notna()]
worst = test_h.groupby(test_h.index.floor("D"))["aar_ratio"].idxmin()
worst = sorted(worst, key=lambda t: test_h.loc[t, "aar_ratio"])[:8]
labels = {t: f"{t:%d %b %H:%MZ} — actual AAR dipped to {test_h.loc[t, 'aar']:.0f}/h" for t in worst}
jump = st.sidebar.selectbox("Jump to a notable weather event", ["(choose)"] + worst, format_func=lambda t: t if t == "(choose)" else labels[t])
default = (jump - pd.Timedelta(hours=6)) if jump != "(choose)" else pd.Timestamp("2025-03-15 06:00", tz="UTC")
if jump == "(choose)" and qp.get("t0"):
    try:
        default = pd.Timestamp(qp["t0"]).tz_localize("UTC")
    except (ValueError, TypeError):
        pass
default = min(max(default, pd.Timestamp("2025-01-01 00:00", tz="UTC")), pd.Timestamp("2025-07-31 11:00", tz="UTC"))
d = st.sidebar.date_input("Forecast issue date (UTC)", value=default.date(), min_value=pd.Timestamp("2025-01-01").date(),
                          max_value=pd.Timestamp("2025-07-31").date())
hr = st.sidebar.selectbox("Issue hour (UTC)", list(range(24)), index=int(default.hour))
t0 = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=hr)
reveal = st.sidebar.toggle("Reveal what actually happened", value=True)
sens = st.sidebar.radio("Alert sensitivity", ["Balanced", "Cautious"],
                        help="Balanced alerts on the forecast (3 h ahead on 2025 data: 71% of alerts were real, 69% of reductions caught). "
                             "Cautious alerts on the 10% low bound (47% of alerts real, 90% of reductions caught).")
thr = st.sidebar.slider("Flag flights at risk ≥", 0.10, 0.50, 0.25, 0.05)
st.sidebar.caption("Replay of the held-out 2025 period. Models were trained on data before 2025 only.")

# ------------------------------------------------------------------ data for the selected issue time
fc = aar_pred[(aar_pred["airport"] == airport) & (aar_pred["issue_utc"] == t0)].sort_values("hour_utc").copy()
if len(fc) < 10:
    st.warning("No forecast for this issue time (outside the test window). Pick a date between 2025-01-02 and 2025-07-31.")
    st.stop()
fc["pred_aar"], fc["lo_aar"], fc["hi_aar"] = (fc[c] * fc["aar_vfr"] for c in ("pred_ratio", "lo_ratio", "hi_ratio"))
fc["taf_aar_val"] = fc["taf_aar_ratio"] * fc["aar_vfr"]
fc["persist_aar"] = fc["persist_ratio"] * fc["aar_vfr"]
fc["actual_aar"] = fc["y_ratio"] * fc["aar_vfr"]
wins = advisory.capacity_windows(fc, "pred_ratio" if sens == "Balanced" else "lo_ratio")

# flights whose scheduled time is 3-12 h after the issue time (nearest-hour lead model)
d_all = dl_pred[dl_pred["airport"] == airport].copy()
d_all["h_ahead"] = (d_all["event_utc"] - t0) / pd.Timedelta(hours=1)
fl = d_all[(d_all["h_ahead"] >= 2.5) & (d_all["h_ahead"] < 12.5) & (d_all["lead_h"] == d_all["h_ahead"].round().clip(3, 12))].copy()
fl["risk_level"] = advisory.risk_level(fl["p"])
n_high = int((fl["p"] >= thr).sum())

# ------------------------------------------------------------------ header + alert
n_flights = dl_pred[dl_pred["lead_h"] == 3]["flight_id"].nunique()
html(f'''<div class="topbar"><span class="brand"><span class="ring"></span>Capacity &amp; Risk</span>
  <span><span class="dot"></span>METAR · TAF · BTS · {n_flights:,} test flights</span>
  <span class="right"><span>Replay of held-out 2025</span><span class="mono">{t0:%Y-%m-%d %H:%MZ}</span></span></div>''')
html(f'''<div class="hero"><h1>{cfg["iata"]} outlook</h1>
  <p>Issued {t0:%d %b %Y %H:%MZ} · local {t0.tz_convert(cfg["tz"]):%H:%M %Z} · arrival capacity and severe-delay risk for the next 3 to 12 hours.</p></div>''')

prev = h_ap.loc[t0 - pd.Timedelta(hours=1)] if (t0 - pd.Timedelta(hours=1)) in h_ap.index else None
cur_aar = float(prev["aar"]) if prev is not None and pd.notna(prev["aar"]) else np.nan
min_row = fc.loc[fc["pred_aar"].idxmin()]
lo_txt = f"could be as low as {min_row['lo_aar']:.0f}/h"
if wins and any(w["severity"] == "severe" for w in wins):
    html(alert("severe", f"Severe capacity reduction expected around {min_row['hour_utc']:%H:%MZ}",
               f"AAR ≈ {min_row['pred_aar']:.0f}/h ({lo_txt}), {100 * (1 - min_row['pred_ratio']):.0f}% below best case."))
elif wins:
    html(alert("moderate", f"Capacity reduction expected around {min_row['hour_utc']:%H:%MZ}",
               f"Lowest forecast AAR ≈ {min_row['pred_aar']:.0f}/h ({lo_txt})."))
else:
    html(alert("clear", "No capacity reduction forecast in the next 3–12 h", "Routine monitoring; re-check at the next METAR or TAF update."))

cat = str(prev["flight_category"]) if prev is not None and pd.notna(prev["flight_category"]) else "n/a"
low_tone = "stop" if min_row["pred_ratio"] <= advisory.SEVERE else "warn" if min_row["pred_ratio"] <= advisory.MODERATE else ""
html('<div class="tiles">' + "".join([
    tile(cat, "", "Last published weather", "good" if cat == "VFR" else "warn" if cat != "n/a" else ""),
    tile(f"{cur_aar:.0f}" if np.isfinite(cur_aar) else "–", f"/h of {cfg['aar']['VFR']}", "AAR now (weather-derived)"),
    tile(f"{min_row['pred_aar']:.0f}", f"/h  {min_row['lo_aar']:.0f}–{min_row['hi_aar']:.0f}", "Lowest forecast AAR, 80% range", low_tone),
    tile(f"{len(fl)}", "flights", "Sampled flights in window"),
    tile(f"{n_high}", f"{n_high / max(len(fl), 1):.0%}", f"Flights at risk ≥ {int(thr * 100)}%", "warn" if n_high else ""),
]) + "</div>")

tab_f, tab_d, tab_h, tab_r, tab_e, tab_s = st.tabs(["Capacity", "Delay risk", "History", "Recommendations", "Evaluation", "Storm hours"])

# ------------------------------------------------------------------ tab: capacity forecast
with tab_f:
    hist = h_ap.loc[t0 - pd.Timedelta(hours=24): t0 - pd.Timedelta(hours=1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["aar"], name="Observed (weather-derived)", line=dict(color="#c5ccd5", width=2)))
    fig.add_trace(go.Scatter(x=pd.concat([fc["hour_utc"], fc["hour_utc"][::-1]]), y=pd.concat([fc["hi_aar"], fc["lo_aar"][::-1]]),
                             fill="toself", fillcolor="rgba(76,157,255,0.18)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["pred_aar"], name="Model forecast", line=dict(color=ACCENT, width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["taf_aar_val"], name="TAF-implied", line=dict(color=OK, dash="dot")))
    fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["persist_aar"], name="Persistence (naive)", line=dict(color=MUTED, dash="dot")))
    if reveal:
        fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["actual_aar"], name="What actually happened", line=dict(color=STOP, width=2)))
    for frac, label in ((advisory.MODERATE, "moderate reduction (80%)"), (advisory.SEVERE, "severe reduction (60%)")):
        fig.add_hline(y=frac * cfg["aar"]["VFR"], line_dash="dash", line_color=AMBER if frac > 0.7 else STOP, line_width=1,
                      annotation_text=label, annotation_position="top left", annotation_font_color=TEXT2)
    fig.add_vline(x=t0, line_color="rgba(255,255,255,.5)", line_width=1)
    fig.update_layout(title="Arrival acceptance rate, last 24 h and next 12 h", height=470, yaxis_title="arrivals / hour",
                      legend=dict(orientation="h", y=-0.14, x=0), margin=dict(b=70), hovermode="x unified")
    chart(fig)
    show = fc[["hour_utc", "lead_h", "pred_aar", "lo_aar", "hi_aar", "taf_aar_val", "persist_aar"] + (["actual_aar"] if reveal else [])].copy()
    show["hour_utc"] = show["hour_utc"].dt.strftime("%H:%MZ")
    show.columns = ["Hour (UTC)", "Lead (h)", "Forecast AAR", "Low (10%)", "High (90%)", "TAF-implied", "Persistence"] + (["Actual"] if reveal else [])
    st.dataframe(show.round(0).astype({c: int for c in show.columns if c not in ("Hour (UTC)",)}), hide_index=True, width="stretch")
    html('<p class="caption">AAR is derived from FAA facility-called rates by flight category, reduced for crosswind, tailwind, '
         'thunderstorms and winter weather. The 80% range comes from quantile models.</p>')

# ------------------------------------------------------------------ tab: delay risk
with tab_d:
    if fl.empty:
        html(alert("clear", "No sampled flights are scheduled in this window."))
    else:
        t, models = load_delay_models()
        fl = fl.sort_values("event_utc").reset_index(drop=True)
        drivers = []
        for lead, g in fl.groupby("lead_h"):
            key = list(zip(g["flight_id"], g["direction"], g["airport"], g["lead_h"]))
            bundle = models[int(lead)]
            contrib = TD.explain_bundle(bundle, t.loc[key]).to_numpy()      # log-odds vs the average flight
            feats = list(TD.explain_bundle(bundle, t.loc[key[:1]]).columns)
            for i, idx in enumerate(g.index):
                order = np.argsort(-contrib[i])[:3]
                drivers.append((idx, "; ".join(f"{nice(feats[j])} (+{contrib[i, j]:.2f})" for j in order if contrib[i, j] > 0.05) or "no strong driver"))
        fl["Main risk drivers"] = pd.Series(dict(drivers))
        hourly_risk = fl.assign(h=fl["event_utc"].dt.floor("h")).groupby("h").agg(mean_risk=("p", "mean"), n=("p", "size"), high=("p", lambda s: int((s >= thr).sum())))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly_risk.index, y=hourly_risk["mean_risk"] * 100, name="Mean severe-delay risk (%)",
                             marker=dict(color=[STOP if v >= thr else ACCENT for v in hourly_risk["mean_risk"]], cornerradius=6),
                             text=[f"{n} flt" for n in hourly_risk["n"]], textposition="outside", textfont=dict(color=TEXT2)))
        fig.add_hline(y=float(dl_pred["severe_delay"].mean() * 100), line_dash="dot", line_color=MUTED,
                      annotation_text="2025 average risk", annotation_position="top left", annotation_font_color=TEXT2)
        fig.update_layout(title="Severe-delay risk by scheduled hour", height=340, yaxis_title="mean P(delay > 45 min), %")
        chart(fig)
        tbl = fl[["event_utc", "direction", "other_airport", "carrier", "p", "risk_level", "Main risk drivers"] + (["delay_min"] if reveal else [])].copy()
        tbl["event_utc"] = tbl["event_utc"].dt.strftime("%d %b %H:%MZ")
        tbl["p"] = (tbl["p"] * 100).round(0).astype(int)
        tbl["direction"] = tbl["direction"].map({"dep": "departure", "arr": "arrival"})
        tbl.columns = ["Scheduled", "Type", "Other airport", "Airline", "Risk %", "Level", "Main risk drivers"] + (["Actual delay (min)"] if reveal else [])
        st.dataframe(tbl.sort_values("Risk %", ascending=False), hide_index=True, width="stretch",
                     column_config={"Risk %": st.column_config.ProgressColumn("Risk", format="%d%%", min_value=0, max_value=100)})
        html('<p class="caption">Risk = probability of a delay over 45 minutes (Platt-calibrated only where that lowered held-out 2024 error). '
             'The flight file is a sample of real traffic, so counts are for the sampled flights only; compare risk rates, not raw counts, between airports.</p>')

# ------------------------------------------------------------------ tab: history
with tab_h:
    c1, c2 = st.columns(2)
    past = h_ap.loc[t0 - pd.Timedelta(days=30): t0]
    daily = past.groupby(past.index.floor("D")).agg(mean_ratio=("aar_ratio", "mean"), low_hours=("aar_ratio", lambda s: int((s <= advisory.MODERATE).sum())))
    f1 = go.Figure(go.Bar(x=daily.index, y=daily["low_hours"], marker=dict(color=ACCENT, cornerradius=5)))
    f1.update_layout(title="Hours per day with AAR ≥20% below best case (last 30 days)", height=320)
    chart(f1, c1)
    dd = dl_pred[(dl_pred["airport"] == airport) & (dl_pred["lead_h"] == 3) & (dl_pred["event_utc"] <= t0) & (dl_pred["event_utc"] >= t0 - pd.Timedelta(days=30))]
    dday = dd.groupby(dd["event_utc"].dt.floor("D")).agg(pred=("p", "mean"), obs=("severe_delay", "mean"), n=("p", "size"))
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=dday.index, y=dday["obs"] * 100, name="Observed severe-delay share", mode="lines+markers", line=dict(color=STOP)))
    f2.add_trace(go.Scatter(x=dday.index, y=dday["pred"] * 100, name="Model risk (3 h ahead)", mode="lines", line=dict(color=ACCENT, width=3)))
    f2.update_layout(title="Severe-delay share per day, sampled flights (last 30 days)", height=320, yaxis_title="%", legend=dict(orientation="h", y=-0.2))
    chart(f2, c2)
    local = test_h.index.tz_convert(cfg["tz"])
    grid = (test_h["aar_ratio"] <= advisory.MODERATE).groupby([local.dayofweek, local.hour]).mean().unstack() * 100
    f3 = go.Figure(go.Heatmap(z=grid.values, x=grid.columns, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], xgap=3, ygap=3,
                              colorscale=[[0, "rgba(255,255,255,0.06)"], [1, ACCENT]], colorbar=dict(title="% hours", outlinewidth=0),
                              hovertemplate="%{y} %{x}:00 local · %{z:.0f}% of hours reduced<extra></extra>"))
    f3.update_layout(title="When is capacity reduced? Share of 2025 hours ≥20% below best case, by local weekday and hour",
                     xaxis=dict(title="local hour", dtick=2, showgrid=False), yaxis=dict(autorange="reversed", showgrid=False), height=330)
    chart(f3)

# ------------------------------------------------------------------ tab: recommendations
with tab_r:
    if not wins:
        html(alert("clear", "No capacity-reduction window forecast", "Routine monitoring: re-run at the next METAR / TAF update."))
    for w in wins:
        html(alert(w["severity"], f"{w['severity'].upper()} window {w['start']:%H:%MZ}–{w['end']:%H:%MZ}", items=advisory.advice(w, n_high)))
    html('<p class="caption">Recommendations are rule-based and transparent (src/advisory.py); the machine learning supplies only the forecasts. A qualified person decides.</p>')
    if api_key() and st.button("Explain this outlook in plain English (Claude)"):
        ctx = {"airport": cfg["iata"], "issued_utc": f"{t0:%Y-%m-%d %H:%MZ}", "last_weather_category": cat,
               "lowest_forecast_aar_per_h": round(float(min_row["pred_aar"])), "at_hour_utc": f"{min_row['hour_utc']:%H:%MZ}",
               "aar_80pct_range": [round(float(min_row["lo_aar"])), round(float(min_row["hi_aar"]))],
               "best_case_aar_per_h": cfg["aar"]["VFR"], "flights_in_window": len(fl), "flights_at_risk": n_high,
               "risk_threshold": thr, "windows": [{"severity": w["severity"], "start": f"{w['start']:%H:%MZ}",
                                                    "end": f"{w['end']:%H:%MZ}", "advice": advisory.advice(w, n_high)} for w in wins]}
        try:
            with st.spinner("Asking Claude…"):
                html(alert("clear", "Plain-English summary (Claude, explanation only)", escape(claude_explain(ctx))))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Claude summary unavailable ({type(e).__name__}). The forecasts and rules above do not depend on it.")

# ------------------------------------------------------------------ tab: evidence
with tab_e:
    lead = st.segmented_control("Lead time", [3, 6, 9, 12], default=3, format_func=lambda h: f"+{h} h") or 3
    am = rep["aar_metrics"]
    a = am[(am.scope == "all") & (am.lead_h == lead)].set_index("model")
    dm = rep["delay_metrics"]
    dsel = dm[(dm.scope == "all") & (dm.lead_h == lead)]
    dv = lambda m, variant="calibrated": (dsel[(dsel.model == m) & (dsel.variant == variant)]["pr_auc"].tolist() or [np.nan])[0]
    full, chance, lg, sched = dv(TD.FINAL), dv("constant_base_rate", "raw"), dv("lightgbm_comparison", "raw"), dv("A_schedule")
    band = a.loc["lightgbm_80pct_band", "band_coverage"] if "lightgbm_80pct_band" in a.index else np.nan
    aar_names = (("lightgbm", "Our model", "ok"), ("taf_implied", "TAF-implied", ""), ("persistence", "Persistence", ""), ("climatology", "Climatology", ""))
    cards = [
        compare("Capacity forecast error (RMSE)", f"{a.loc['lightgbm', 'rmse_aar']:.1f}", "/h",
                [(nm, a.loc[m, "rmse_aar"], f"{a.loc[m, 'rmse_aar']:.1f}", tone) for m, nm, tone in aar_names],
                "Arrivals per hour, unseen 2025 data. Shorter is better."),
        compare("Spotting a ≥20% capacity cut (PR-AUC)", f"{a.loc['lightgbm', 'event_pr_auc']:.2f}", "",
                [(nm, a.loc[m, "event_pr_auc"], f"{a.loc[m, 'event_pr_auc']:.2f}", tone) for m, nm, tone in aar_names],
                f"Cuts occur in {a.loc['lightgbm', 'event_rate']:.0%} of hours; a random guess scores that."),
        compare("Severe-delay ranking (PR-AUC)", f"{full:.2f}", f"vs {chance:.2f} chance",
                [("Full model", full, f"{full:.3f}", "ok"), ("Schedule only", sched, f"{sched:.3f}", ""),
                 ("LightGBM", lg, f"{lg:.3f}", ""), ("Chance", chance, f"{chance:.3f}", "")],
                "Schedule-only is reported at +3 h and +12 h."),
        compare("80% forecast range: share of outcomes inside", f"{band:.0%}", "of hours",
                [("Achieved", band, f"{band:.0%}", "ok" if band >= 0.8 else "amber"), ("Promised", 0.8, "80%", "")],
                "Measured on 2025. Above 80% means the range is a little wide, which is the safe side."),
    ]
    html('<div class="grid2">' + "".join(cards) + "</div>")

    if lead in TD.TUNE_LEADS:
        steps = [("A_schedule", "Schedule only"), ("B_plus_recent_hub", "+ recent hub delays"), ("C_plus_metar", "+ METAR weather"),
                 ("D_plus_taf", "+ TAF forecast"), (TD.FINAL, "+ BTS airport traffic")]
        ladder = [(nm, dv(m), f"{dv(m):.3f}", "ok" if m == TD.FINAL else "accent") for m, nm in steps]
        boot = rep["delay_bootstrap"].set_index("lead_h").loc[lead]
        traffic = rep.get("aar_vs_traffic")
        cards = [compare("Delay risk: gain over schedule-only, and what each source adds", f"+{boot.pr_auc_gain_mean:.3f}", "PR-AUC",
                         ladder, f"Day-block bootstrap 95% interval {boot.ci_low:+.3f} to {boot.ci_high:+.3f}.")]
        if traffic is not None:
            tr = traffic[traffic.airport == airport]
            bands = ["AAR below 70%", "AAR 70–85%", "AAR 85–95%", "AAR 95%+"]
            cards.append(compare(f"Does the AAR rule match real {cfg['iata']} traffic? Landings in the lowest vs highest band",
                                 f"−{100 * (tr.actual_to_scheduled.iloc[-1] - tr.actual_to_scheduled.iloc[0]):.0f}", "pts",
                                 [(bands[i], v, f"{v:.0%}", "accent") for i, v in enumerate(tr.actual_to_scheduled)],
                                 f"Busy hours only: share of scheduled BTS arrivals that landed, by rule AAR band. "
                                 f"Rank correlation {tr.spearman_aar_vs_completion.iloc[0]:.2f}: the direction holds, the effect is modest."))
        html('<div class="grid2">' + "".join(cards) + "</div>")

    rel_key = f"delay_reliability_lead{lead}"
    c1, c2 = st.columns(2)
    if rel_key in rep:
        r = rep[rel_key]
        f5 = go.Figure()
        f5.add_trace(go.Scatter(x=[0, r.mean_pred.max()], y=[0, r.mean_pred.max()], name="perfect", line=dict(color=MUTED, dash="dot")))
        f5.add_trace(go.Scatter(x=r.mean_pred, y=r.observed, name="model", mode="lines+markers", line=dict(color=ACCENT, width=3)))
        f5.update_layout(title=f"Is the risk well calibrated? (+{lead} h, 10 risk groups)", xaxis_title="predicted", yaxis_title="observed",
                         height=340, xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=dict(x=0.02, y=0.98))
        chart(f5, c1)
        fcases = rep[f"delay_failure_cases_lead{lead}"]
        fx = fcases[["kind", "airport", "carrier", "event_utc", "delay_min", "p"]].head(10).copy()
        fx["kind"] = fx["kind"].map({"missed_severe_delay": "missed", "false_alarm": "false alarm"})
        fx["event_utc"] = pd.to_datetime(fx["event_utc"]).dt.strftime("%d %b %H:%MZ")
        fx["p"] = (fx["p"] * 100).round(0)
        fx.columns = ["Failure", "Hub", "Airline", "Scheduled", "Delay (min)", "Risk %"]
        c2.markdown('<div class="section">Where the delay model fails</div>', unsafe_allow_html=True)
        c2.dataframe(fx, hide_index=True, width="stretch", height=300)
    html(alert("moderate", "Known limits", items=[
        "The delay model cannot anticipate multi-hour disruptions with no weather precursor (e.g. maintenance).",
        "In the very worst capacity drops the AAR model is no more accurate than the TAF-implied rate.",
        "AAR labels are a weather-derived proxy of FAA called rates, not logged facility data.",
        "The flight file is a ~3% sample of traffic; BTS covers reporting carriers only."]))

# ------------------------------------------------------------------ tab: storm hours (full-BTS pipeline, jho/aerocast_data)
with tab_s:
    sa, sd, fb = rep.get("jho_storm_hours_arrivals"), rep.get("jho_storm_hours_severe_delay"), rep.get("jho_fullbts_severe_metrics")
    if sa is None or sd is None:
        html(alert("moderate", "Storm-hour reports not found", "Run jho/aerocast_data/run_all.py and copy its reports into reports/jho_*.csv."))
    else:
        html(alert("clear", "How well do the forecasts hold up in the hours the challenge is about?",
                   "Storm hour = a NOAA Storm Event (thunderstorm wind, hail, tornado, flash flood) within 50 km overlaps the target hour. "
                   "NOAA logs are compiled after the event, so they only label 2025 test hours; they are never a model input. "
                   "Target here is actual BTS arrivals per hour from the full archive (3.26 M flights), not the weather-derived AAR."))
        lead_s = st.segmented_control("Lead time", [3, 6, 9, 12], default=3, format_func=lambda h: f"+{h} h", key="storm_lead") or 3
        a_row = sa[(sa.airport == cfg["iata"]) & (sa.horizon_h == lead_s)].iloc[0]
        rows = [("Our model", a_row.rmse_model, f"{a_row.rmse_model:.1f}", "ok"), ("Climatology", a_row.rmse_climatology, f"{a_row.rmse_climatology:.1f}", ""),
                ("Schedule", a_row.rmse_schedule, f"{a_row.rmse_schedule:.1f}", ""), ("Persistence", a_row.rmse_persistence, f"{a_row.rmse_persistence:.1f}", "")]
        cards = [compare(f"Arrivals per hour in storm hours (RMSE, {int(a_row.storm_hours)} hours)", f"{a_row.rmse_model:.1f}", "/h", rows,
                         f"Normal hours: {a_row.rmse_model_all_hours:.1f}/h. Errors are about {a_row.rmse_model / a_row.rmse_model_all_hours:.1f}x larger in storms. Shorter is better."),
                 compare("90% forecast range in storm hours: share of outcomes inside", f"{a_row.interval_coverage:.0%}", "of hours",
                         [("Storm hours", a_row.interval_coverage, f"{a_row.interval_coverage:.0%}", "stop"),
                          ("All hours", a_row.coverage_all_hours, f"{a_row.coverage_all_hours:.0%}", "ok"), ("Promised", 0.9, "90%", "")],
                         "Failure case: in storms the range is far too narrow. Read storm-hour forecasts as a warning, not an exact count.")]
        for direction in ("arrival", "departure"):
            r = sd[(sd.airport == cfg["iata"]) & (sd.direction == direction) & (sd.horizon_h == lead_s)]
            if len(r):
                r = r.iloc[0]
                tone = "ok" if r.verdict == "model better" else "amber" if r.verdict == "no clear difference" else "stop"
                cards.append(compare(f"Severe-delay ranking in storm hours, {direction}s (PR-AUC)", f"{r.pr_auc_model:.2f}", f"vs {r.pr_auc_climatology:.2f} climatology",
                                     [("Our model", r.pr_auc_model, f"{r.pr_auc_model:.3f}", tone), ("Climatology", r.pr_auc_climatology, f"{r.pr_auc_climatology:.3f}", ""),
                                      ("Severe share", r.severe_rate, f"{r.severe_rate:.0%}", ""), ("Mean predicted", r.mean_predicted_p, f"{r.mean_predicted_p:.0%}", "amber")],
                                     f"{int(r.flights)} flights in {int(r.storm_hours)} storm hours. 95% bootstrap interval for the gain "
                                     f"{r.diff_ci95_low:+.3f} to {r.diff_ci95_high:+.3f}: {r.verdict}. The model under-states storm risk."))
        html('<div class="grid2">' + "".join(cards) + "</div>")

        c1, c2 = st.columns(2)
        sap = sa[sa.airport == cfg["iata"]].sort_values("horizon_h")
        f6 = go.Figure()
        f6.add_trace(go.Bar(x=[f"+{h} h" for h in sap.horizon_h], y=sap.rmse_model_all_hours, name="All hours", marker=dict(color=ACCENT, cornerradius=5)))
        f6.add_trace(go.Bar(x=[f"+{h} h" for h in sap.horizon_h], y=sap.rmse_model, name="Storm hours", marker=dict(color=STOP, cornerradius=5)))
        f6.add_trace(go.Scatter(x=[f"+{h} h" for h in sap.horizon_h], y=sap.rmse_climatology, name="Climatology, storm hours", mode="lines+markers", line=dict(color=MUTED, dash="dot")))
        f6.update_layout(title=f"{cfg['iata']} arrivals forecast error by lead time", yaxis_title="RMSE, arrivals / hour", barmode="group",
                         height=340, legend=dict(orientation="h", y=-0.2))
        chart(f6, c1)
        sdp = sd[sd.airport == cfg["iata"]].copy()
        sdp["label"] = sdp.direction.str[:3] + " +" + sdp.horizon_h.astype(str) + " h"
        f7 = go.Figure()
        f7.add_trace(go.Scatter(x=sdp.label, y=sdp.pr_auc_model - sdp.pr_auc_climatology, mode="markers", name="gain over climatology",
                                marker=dict(color=ACCENT, size=10),
                                error_y=dict(type="data", symmetric=False, array=sdp.diff_ci95_high - (sdp.pr_auc_model - sdp.pr_auc_climatology),
                                             arrayminus=(sdp.pr_auc_model - sdp.pr_auc_climatology) - sdp.diff_ci95_low, color=TEXT2)))
        f7.add_hline(y=0, line_color=MUTED, line_dash="dot")
        f7.update_layout(title="Storm-hour severe-delay gain over climatology, with 95% interval", yaxis_title="PR-AUC gain", height=340, showlegend=False)
        chart(f7, c2)

        if fb is not None:
            fbs = fb[(fb.airport == cfg["iata"]) & (fb.horizon_h == lead_s) & (fb.get("subset", "all_hours") == "all_hours")]
            if len(fbs):
                html('<div class="section">Same classifier on the full BTS archive, all 2025 hours</div>')
                show = fbs[["direction", "n", "base_rate", "pr_auc_model", "pr_auc_climatology", "brier_model", "brier_climatology", "precision", "recall"]].copy()
                show.columns = ["Type", "Test flights", "Severe share", "PR-AUC model", "PR-AUC climatology", "Brier model", "Brier climatology", "Precision", "Recall"]
                st.dataframe(show.round(3), hide_index=True, width="stretch")
                html('<p class="caption">LightGBM trained on 2023-01 to 2024-06 BTS flights, threshold chosen on 2024 H2, scored once on 2025. '
                     'About 45,000 test flights per airport, direction and lead, so the intervals are tight.</p>')
        html(alert("moderate", "Known limits in storm hours", items=[
            f"Few storm hours: {int(sap.storm_hours.iloc[0])} at {cfg['iata']} in the 2025 test period, so storm results are less certain than the overall ones.",
            "In storms the model's probabilities run well below the observed severe-delay share; treat any rise in risk as significant.",
            "A TAF thunderstorm-timing feature (within 2 h of the target hour) was tested and rejected on 2024 validation data."]))
