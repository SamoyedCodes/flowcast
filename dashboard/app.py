"""ATC decision-support dashboard: airport capacity (AAR) and delay-risk forecasts, 3-12 h ahead.

Replay mode: pick any "issue time" in the held-out 2025 test period.  The forecasts shown are exactly what the
models would have produced at that time (trained only on data before 2025); switch on "reveal outcome" to
compare with what happened.   Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import advisory, config as C  # noqa: E402
from src import train_delay as TD  # noqa: E402

st.set_page_config(page_title="Airport Capacity & Delay Risk", layout="wide", page_icon="✈")

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
}


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
st.sidebar.title("✈ Capacity & Risk")
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
    st.warning("No forecast for this issue time (outside the test window). Pick a date between 2025-01-01 and 2025-07-31.")
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
st.title(f"{cfg['iata']} — capacity & delay-risk outlook")
st.caption(f"Issued {t0:%Y-%m-%d %H:%MZ} · local {t0.tz_convert(cfg['tz']):%H:%M %Z} · forecasts for +3 h … +12 h")

prev = h_ap.loc[t0 - pd.Timedelta(hours=1)] if (t0 - pd.Timedelta(hours=1)) in h_ap.index else None
cur_aar = float(prev["aar"]) if prev is not None and pd.notna(prev["aar"]) else np.nan
min_row = fc.loc[fc["pred_aar"].idxmin()]
lo_txt = f"(could be as low as {min_row['lo_aar']:.0f}/h)"
if wins and any(w["severity"] == "severe" for w in wins):
    st.error(f"SEVERE capacity reduction expected: AAR ≈ {min_row['pred_aar']:.0f}/h {lo_txt}, "
             f"{100 * (1 - min_row['pred_ratio']):.0f}% below best case, around {min_row['hour_utc']:%H:%MZ}.")
elif wins:
    st.warning(f"Capacity reduction expected: lowest forecast AAR ≈ {min_row['pred_aar']:.0f}/h {lo_txt} around {min_row['hour_utc']:%H:%MZ}.")
else:
    st.success("No capacity reduction forecast in the next 3–12 h.")

k = st.columns(5)
k[0].metric("Last published weather", str(prev["flight_category"]) if prev is not None else "n/a")
k[1].metric("AAR now (weather-derived)", f"{cur_aar:.0f}/h" if np.isfinite(cur_aar) else "n/a", f"best case {cfg['aar']['VFR']}/h", delta_color="off")
k[2].metric("Lowest forecast AAR", f"{min_row['pred_aar']:.0f}/h", f"range {min_row['lo_aar']:.0f}–{min_row['hi_aar']:.0f}", delta_color="off")
k[3].metric("Sampled flights in window", len(fl))
k[4].metric(f"Flights at risk ≥ {int(thr * 100)}%", n_high, f"{(n_high / max(len(fl), 1)):.0%} of window", delta_color="off")

tab_f, tab_d, tab_h, tab_r, tab_e = st.tabs(["Capacity forecast", "Delay risk", "History & trends", "Recommendations", "How good is it?"])

# ------------------------------------------------------------------ tab: capacity forecast
with tab_f:
    hist = h_ap.loc[t0 - pd.Timedelta(hours=24): t0 - pd.Timedelta(hours=1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["aar"], name="Observed (weather-derived)", line=dict(color="#555", width=2)))
    fig.add_trace(go.Scatter(x=pd.concat([fc["hour_utc"], fc["hour_utc"][::-1]]), y=pd.concat([fc["hi_aar"], fc["lo_aar"][::-1]]),
                             fill="toself", fillcolor="rgba(31,119,180,0.18)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["pred_aar"], name="Model forecast", line=dict(color="#1f77b4", width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["taf_aar_val"], name="TAF-implied", line=dict(color="#2ca02c", dash="dot")))
    fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["persist_aar"], name="Persistence (naive)", line=dict(color="#aaaaaa", dash="dot")))
    if reveal:
        fig.add_trace(go.Scatter(x=fc["hour_utc"], y=fc["actual_aar"], name="What actually happened", line=dict(color="#d62728", width=2)))
    for frac, label in ((advisory.MODERATE, "moderate reduction (80%)"), (advisory.SEVERE, "severe reduction (60%)")):
        fig.add_hline(y=frac * cfg["aar"]["VFR"], line_dash="dash", line_color="orange" if frac > 0.7 else "red",
                      annotation_text=label, annotation_position="bottom right")
    fig.add_vline(x=t0, line_color="black", line_width=1)
    fig.update_layout(height=430, yaxis_title="Arrival acceptance rate (arrivals / hour)", margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    show = fc[["hour_utc", "lead_h", "pred_aar", "lo_aar", "hi_aar", "taf_aar_val", "persist_aar"] + (["actual_aar"] if reveal else [])].copy()
    show["hour_utc"] = show["hour_utc"].dt.strftime("%H:%MZ")
    show.columns = ["Hour (UTC)", "Lead (h)", "Forecast AAR", "Low (10%)", "High (90%)", "TAF-implied", "Persistence"] + (["Actual"] if reveal else [])
    st.dataframe(show.round(0).astype({c: int for c in show.columns if c not in ("Hour (UTC)",)}), hide_index=True, width="stretch")
    st.caption("AAR is derived from FAA facility-called rates by flight category, reduced for crosswind, tailwind, thunderstorms and "
               "winter weather (see Methods). The 80% range comes from quantile models.")

# ------------------------------------------------------------------ tab: delay risk
with tab_d:
    if fl.empty:
        st.info("No sampled flights are scheduled in this window.")
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
                             marker_color=["#d62728" if v >= thr else "#1f77b4" for v in hourly_risk["mean_risk"]],
                             text=[f"{n} flt" for n in hourly_risk["n"]], textposition="outside"))
        fig.add_hline(y=float(dl_pred["severe_delay"].mean() * 100), line_dash="dot",
                      annotation_text="2025 average risk", annotation_position="top left")
        fig.update_layout(height=330, yaxis_title="Mean P(delay > 45 min), %", margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        tbl = fl[["event_utc", "direction", "other_airport", "carrier", "p", "risk_level", "Main risk drivers"] + (["delay_min"] if reveal else [])].copy()
        tbl["event_utc"] = tbl["event_utc"].dt.strftime("%d %b %H:%MZ")
        tbl["p"] = (tbl["p"] * 100).round(0).astype(int)
        tbl["direction"] = tbl["direction"].map({"dep": "departure", "arr": "arrival"})
        tbl.columns = ["Scheduled", "Type", "Other airport", "Airline", "Risk %", "Level", "Main risk drivers"] + (["Actual delay (min)"] if reveal else [])
        st.dataframe(tbl.sort_values("Risk %", ascending=False), hide_index=True, width="stretch")
        st.caption("Risk = calibrated probability of a delay over 45 minutes. The flight file is a sample of real traffic, so counts are for the "
                   "sampled flights only; compare risk rates, not raw counts, between airports.")

# ------------------------------------------------------------------ tab: history
with tab_h:
    c1, c2 = st.columns(2)
    past = h_ap.loc[t0 - pd.Timedelta(days=30): t0]
    daily = past.groupby(past.index.floor("D")).agg(mean_ratio=("aar_ratio", "mean"), low_hours=("aar_ratio", lambda s: int((s <= advisory.MODERATE).sum())))
    f1 = go.Figure(go.Bar(x=daily.index, y=daily["low_hours"], marker_color="#1f77b4"))
    f1.update_layout(title="Hours per day with AAR ≥20% below best case (last 30 days)", height=320, margin=dict(t=50, b=10))
    c1.plotly_chart(f1, width="stretch")
    dd = dl_pred[(dl_pred["airport"] == airport) & (dl_pred["lead_h"] == 3) & (dl_pred["event_utc"] <= t0) & (dl_pred["event_utc"] >= t0 - pd.Timedelta(days=30))]
    dday = dd.groupby(dd["event_utc"].dt.floor("D")).agg(pred=("p", "mean"), obs=("severe_delay", "mean"), n=("p", "size"))
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=dday.index, y=dday["obs"] * 100, name="Observed severe-delay share", mode="lines+markers", line=dict(color="#d62728")))
    f2.add_trace(go.Scatter(x=dday.index, y=dday["pred"] * 100, name="Model risk (3 h ahead)", mode="lines", line=dict(color="#1f77b4")))
    f2.update_layout(title="Severe-delay share per day, sampled flights (last 30 days)", height=320, margin=dict(t=50, b=10), yaxis_title="%", legend=dict(orientation="h", y=-0.2))
    c2.plotly_chart(f2, width="stretch")
    allh = test_h.assign(hour_local=test_h.index.tz_convert(cfg["tz"]).hour)
    by_hour = allh.groupby("hour_local")["aar_ratio"].apply(lambda s: (s <= advisory.MODERATE).mean() * 100)
    f3 = go.Figure(go.Bar(x=by_hour.index, y=by_hour.values, marker_color="#1f77b4"))
    f3.update_layout(title="How often is capacity reduced, by local hour (2025)?", xaxis_title="Local hour", yaxis_title="% of hours", height=300, margin=dict(t=50, b=10))
    st.plotly_chart(f3, width="stretch")

# ------------------------------------------------------------------ tab: recommendations
with tab_r:
    if not wins:
        st.success("No capacity-reduction window forecast. Routine monitoring: re-run at the next METAR / TAF update.")
    for w in wins:
        box = st.error if w["severity"] == "severe" else st.warning
        lines = advisory.advice(w, n_high)
        box(f"**{w['severity'].upper()} window {w['start']:%H:%MZ}–{w['end']:%H:%MZ}**")
        for line in lines:
            st.markdown(f"- {line}")
    st.caption("Recommendations are rule-based and transparent (src/advisory.py); the machine learning supplies only the forecasts. A qualified person decides.")

# ------------------------------------------------------------------ tab: evidence
with tab_e:
    am = rep["aar_metrics"]
    a = am[(am.scope == "all") & am.rmse_aar.notna()].pivot(index="lead_h", columns="model", values="rmse_aar")
    f4 = go.Figure()
    for m, nm in (("lightgbm", "Model"), ("taf_implied", "TAF-implied"), ("climatology", "Climatology"), ("persistence", "Persistence")):
        f4.add_trace(go.Scatter(x=a.index, y=a[m], name=nm, mode="lines+markers"))
    f4.update_layout(title="AAR forecast error on unseen 2025 data (RMSE, arrivals/hour; lower is better)", xaxis_title="Lead time (h)", height=330, margin=dict(t=50, b=10))
    st.plotly_chart(f4, width="stretch")
    dm = rep["delay_metrics"]
    dd_ = dm[(dm.scope == "all") & ((dm.variant == "calibrated") | dm.model.isin(["lightgbm_comparison", "constant_base_rate"]))]
    p = dd_.pivot_table(index="model", columns="lead_h", values="pr_auc")
    st.markdown("**Delay classifier — PR-AUC on unseen 2025 data** (chance level = share of severe delays, about 13%)")
    st.dataframe(p.round(3), width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("**Is the risk well calibrated? (3 h ahead)**")
    c1.dataframe(rep["delay_reliability_lead3"].round(3), hide_index=True)
    c2.markdown("**Cases where the system fails (3 h ahead)**")
    fcases = rep["delay_failure_cases_lead3"]
    c2.dataframe(fcases[["kind", "airport", "carrier", "event_utc", "delay_min", "p"]].head(10).round(2), hide_index=True)
    st.info("Known limits: the delay model cannot anticipate multi-hour disruptions with no weather precursor (e.g. maintenance); "
            "in the very worst capacity drops the AAR model is no more accurate than the TAF-implied rate; "
            "AAR labels are a weather-derived proxy of the FAA called rates, not logged facility data; the flight file is a ~3% sample of traffic.")
