"""
AeroCast decision-support dashboard for an airport operations duty manager (advisory only).

  streamlit run dashboard.py

Reads the outputs of build_datasets.py and train_models.py. Numbers come only from the statistical
models and baselines; recommendations come from fixed, visible rules. The optional Claude summary
only rephrases the drivers already on screen (key read from ANTHROPIC_API_KEY, never stored).
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from common import DATA, MANIFEST, PROC, ROOT, load_airports

REPORTS, MODELS = DATA / "reports", DATA / "models"

# Rule thresholds. Each one is shown next to the value that triggered it.
RULES = {
    "shortfall_ratio": (0.85, "team assumption: predicted arrivals below 85% of scheduled"),
    "crosswind_kt": (15.0, "FAA Order 8400.9 (1981): crosswind incl. gusts <= 15 kt on clear, dry runways"),
    "tailwind_kt": (5.0, "FAA JO 7110.65 3-5-1: runway most nearly into the wind once wind >= 5 kt"),
}
COST_PER_MIN = 98.41  # USD per aircraft block minute, A4A 2025 (airlines.org/dataset/u-s-passenger-carrier-delay-costs)


@st.cache_data
def load():
    a = pd.read_parquet(PROC / "aar_predictions.parquet")
    s = pd.read_parquet(PROC / "severe_predictions.parquet")
    info = json.loads((REPORTS / "run_info.json").read_text())
    mets = {n: pd.read_csv(REPORTS / f"{n}.csv") for n in ("aar_metrics", "severe_metrics", "severe_calibration")
            if (REPORTS / f"{n}.csv").exists()}
    return a, s, info, mets


def model_version():
    h = hashlib.sha256()
    for f in sorted(MODELS.glob("*.txt")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def recommendations(row, flights_hi):
    """Deterministic rules -> list of (level, text, evidence)."""
    out = []
    sched, pred, hi = row.sched_arrivals, row.pred, row.pred_hi
    thr, src = RULES["shortfall_ratio"]
    if sched >= 5 and hi < thr * sched:
        out.append(("high", "Arrival shortfall likely even at the top of the interval. Review an arrival-rate reduction "
                    "with ATC/TMU now, before aircraft are airborne.",
                    f"interval top {hi:.1f} < {thr:.0%} x scheduled {sched:.0f} ({src})"))
    elif sched >= 5 and pred < thr * sched:
        out.append(("medium", "Possible arrival shortfall. Re-check at the next TAF issue before acting.",
                    f"point forecast {pred:.1f} < {thr:.0%} x scheduled {sched:.0f}; interval top {hi:.1f} does not confirm"))
    xw, xsrc = RULES["crosswind_kt"]
    if pd.notna(row.get("obs_xwind_gust_kt")) and row.obs_xwind_gust_kt > xw:
        out.append(("medium", "Crosswind on the assumed runway heading is above the runway-use limit. Check runway "
                    "configuration options.", f"gust crosswind {row.obs_xwind_gust_kt:.0f} kt > {xw:.0f} kt ({xsrc})"))
    tw, tsrc = RULES["tailwind_kt"]
    if pd.notna(row.get("obs_tailwind_on_assumed_hdg_kt")) and row.obs_tailwind_on_assumed_hdg_kt > tw:
        out.append(("medium", "Tailwind on the assumed heading: expect a flow change to the reciprocal runway and a "
                    "short throughput dip during the change.",
                    f"tailwind {row.obs_tailwind_on_assumed_hdg_kt:.0f} kt > {tw:.0f} kt ({tsrc})"))
    ts_obs = row.get("obs_ts_in_hour", 0) == 1
    ts_taf = row.get("taf_ts", 0) == 1 or row.get("taf_tempo_or_prob_ts", 0) == 1
    if ts_obs or ts_taf:
        out.append(("medium", "Thunderstorm signal. Plan for ramp lightning holds and taxi-out queues.",
                    ("METAR TS at t0" if ts_obs else "") + (" + " if ts_obs and ts_taf else "") + ("TAF TS for target hour" if ts_taf else "")))
    if row.get("obs_flight_cat") in ("IFR", "LIFR") or row.get("taf_flight_cat") in ("IFR", "LIFR"):
        out.append(("medium", "Instrument conditions. Arrival capacity is typically lower in IMC.",
                    f"METAR {row.get('obs_flight_cat')}, TAF {row.get('taf_flight_cat', 'n/a')} "
                    "(ATL: ~250 ops/h visual vs ~180 low-IMC, FAA ATL Capacity Profile 2018)"))
    if flights_hi:
        out.append(("info", f"{flights_hi} flight(s) in this hour are above the severe-delay alert threshold (see table).",
                    "classifier probability >= validation-chosen threshold"))
    if not out:
        out.append(("ok", "No rule triggered. Keep the normal plan.", "all rule checks below threshold"))
    return out


def api_key():
    """ANTHROPIC_API_KEY from the environment, else from a local, git-ignored .env next to this file."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    return key or None


def claude_summary(context):
    key = api_key()
    if not key:
        return None
    body = {"model": "claude-haiku-5-5", "max_tokens": 1500,
            "system": "You explain airport capacity forecasts to an operations duty manager. Use ONLY the numbers "
                      "given. Do not invent, round differently or add numbers. Do not give orders; the human decides. "
                      "Plain English, max 4 sentences, no em dashes.",
            "messages": [{"role": "user", "content": json.dumps(context, default=str)}]}
    r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=30,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    r.raise_for_status()
    d = r.json()
    txt = "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text").strip()
    if not txt:
        raise RuntimeError(f"empty reply (stop_reason={d.get('stop_reason')})")
    return txt


def main():
    st.set_page_config(page_title="AeroCast", layout="wide")
    st.title("AeroCast: arrival capacity and severe-delay risk, 3 to 12 h ahead")
    st.caption("Advisory decision support for the airport operations duty manager. It does not issue ATC instructions; "
               "the qualified person decides.")
    a, s, info, mets = load()
    test = a[(a.split == "test") & a.pred.notna()]

    # data validity guard: a sparse flight sample cannot measure capacity
    hd_mean = test[test.high_demand_hour == 1].actual_arrivals.mean()
    if hd_mean < 20:
        st.error(f"Data check failed: high-demand hours average only {hd_mean:.1f} arrivals/h. This looks like the "
                 "0.5% organiser sample, not full BTS data, so AAR numbers here are NOT valid capacity estimates. "
                 "Run run_all.py to rebuild from full BTS data.")

    with st.sidebar:
        aps = sorted(a.airport.unique())
        ap = st.selectbox("Airport", aps)
        H = st.select_slider("Forecast horizon (h)", options=sorted(a.horizon_h.unique()), value=6)
        sub = test[(test.airport == ap) & (test.horizon_h == H)].sort_values("target_hour_utc")
        tz = load_airports([ap])[ap]["tz"]
        hours = sub.target_hour_utc.tolist()
        default = int(sub.reset_index(drop=True).eval("sched_arrivals - pred").idxmax()) if len(sub) else 0
        target = st.selectbox("Target hour (UTC, test period)", hours, index=default,
                              format_func=lambda t: f"{t:%Y-%m-%d %H:%MZ}  ({t.tz_localize('UTC').tz_convert(tz):%H:%M} local)")
        use_llm = st.checkbox("Claude plain-English summary", value=False,
                              help="Needs ANTHROPIC_API_KEY in the environment or in aerocast_data/.env. Explains on-screen drivers only.")

    row = sub[sub.target_hour_utc == target].iloc[0]
    fl = s[(s.airport == ap) & (s.horizon_h == H) & (s.event_utc.dt.floor("h") == target)].copy()
    thr = info["severe_threshold"]
    fl["alert"] = fl.p_severe >= thr

    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Historical trend", "Model evaluation", "Provenance & limits"])
    with tab1:
        st.markdown(f"**Forecast made at t0 = {row.t0_utc:%Y-%m-%d %H:%MZ}** for target hour "
                    f"**{target:%H:%MZ}** ({H} h ahead). Only data known at t0 is used.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted arrivals", f"{row.pred:.1f}", help="LightGBM; AAR proxy = actual arrivals/h (BTS carriers)")
        c2.metric("90% interval", f"{row.pred_lo:.0f} to {row.pred_hi:.0f}", help="split-conformal on validation residuals")
        c3.metric("Scheduled arrivals", f"{row.sched_arrivals:.0f}",
                  delta=f"{(row.pred / row.sched_arrivals - 1):+.0%} vs schedule" if row.sched_arrivals else None)
        c4.metric("Flights above severe-risk threshold", f"{int(fl.alert.sum())} / {len(fl)}", help=f"threshold {thr:.2f}")

        st.subheader("Recommendations (rule-based, advisory)")
        recs = recommendations(row, int(fl.alert.sum()))
        icon = {"high": "🔴", "medium": "🟠", "info": "🔵", "ok": "🟢"}
        for lvl, text, ev in recs:
            st.markdown(f"{icon[lvl]} {text}  \n<small>Evidence: {ev}</small>", unsafe_allow_html=True)

        st.subheader("Weather at t0 (latest METAR, and TAF for the target hour if available)")
        wx = {"Flight category": row.get("obs_flight_cat"), "Wind dir / speed (kt)": f"{row.get('obs_drct')} / {row.get('obs_sknt')}",
              "Gust (kt)": row.get("obs_gust_kt"), "Crosswind on assumed hdg (kt)": row.get("obs_xwind_kt"),
              "Gust crosswind (kt)": row.get("obs_xwind_gust_kt"), "Headwind (+) / tailwind (-) (kt)": row.get("obs_hwind_kt"),
              "Visibility (mi)": row.get("obs_vsby"), "Ceiling (ft)": row.get("obs_ceiling_ft"),
              "Thunder in last hour": row.get("obs_ts_in_hour"), "Pressure tendency 3 h (inHg)": row.get("obs_pres_tend_3h"),
              "Obs age (h)": row.get("obs_obs_age_h"), "TAF category": row.get("taf_flight_cat", "no TAF loaded"),
              "TAF thunder (prevailing / TEMPO-PROB)": f"{row.get('taf_ts', 'n/a')} / {row.get('taf_tempo_or_prob_ts', 'n/a')}",
              "TAF age (h)": row.get("taf_age_h", "n/a")}
        st.dataframe(pd.DataFrame({"value": [str(round(v, 1)) if isinstance(v, float) else str(v) for v in wx.values()]},
                                  index=list(wx.keys())), width="stretch")
        hdg = load_airports([ap])[ap].get("runway_heading_deg")
        st.caption(f"Wind components use the challenge's assumed runway heading for {ap}: {hdg} deg.")

        st.subheader("Flights in the target hour")
        st.dataframe(fl.sort_values("p_severe", ascending=False)[
            ["direction", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "event_utc", "p_severe", "base_climatology", "alert", "y_severe"]
        ].rename(columns={"base_climatology": "climatology_rate", "y_severe": "actual_severe (hindsight)"}), width="stretch")

        if use_llm:
            ctx = {"airport": ap, "horizon_h": H, "predicted_arrivals": round(row.pred, 1),
                   "interval": [round(row.pred_lo, 1), round(row.pred_hi, 1)], "scheduled": row.sched_arrivals,
                   "rules_triggered": [t for _, t, _ in recs], "evidence": [e for _, _, e in recs]}
            try:
                txt = claude_summary(ctx)
                st.info(txt if txt else "ANTHROPIC_API_KEY not set, so no summary. All numbers above are unaffected.")
            except Exception as e:  # noqa: BLE001
                st.warning(f"Claude summary unavailable ({e}). The forecast and rules above do not depend on it.")

    with tab2:
        win = a[(a.airport == ap) & (a.horizon_h == H) & a.pred.notna()
                & a.target_hour_utc.between(target - pd.Timedelta(hours=36), target + pd.Timedelta(hours=12))].sort_values("target_hour_utc")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=win.target_hour_utc, y=win.pred_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=win.target_hour_utc, y=win.pred_lo, fill="tonexty", line=dict(width=0), name="90% interval",
                                 fillcolor="rgba(31,119,180,0.2)"))
        fig.add_trace(go.Scatter(x=win.target_hour_utc, y=win.pred, name="predicted", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=win.target_hour_utc, y=win.sched_arrivals, name="scheduled", line=dict(dash="dot", color="grey")))
        fig.add_trace(go.Scatter(x=win.target_hour_utc, y=win.actual_arrivals, name="actual (hindsight)", line=dict(color="black")))
        fig.add_vline(x=target, line_dash="dash", line_color="red")
        fig.update_layout(height=380, yaxis_title="arrivals per hour", xaxis_title="UTC", margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        hist = a[(a.airport == ap) & (a.horizon_h == H)].copy()
        hist["date"] = hist.target_hour_utc.dt.date
        daily = hist.groupby("date").agg(actual=("actual_arrivals", "sum"), scheduled=("sched_arrivals", "sum")).reset_index()
        daily["completion_ratio"] = daily.actual / daily.scheduled.replace(0, np.nan)
        st.markdown("**Daily arrival completion ratio (actual / scheduled)**, normalised so airports of different size compare fairly")
        st.line_chart(daily.set_index("date").completion_ratio)

    with tab3:
        if "aar_metrics" in mets:
            m = mets["aar_metrics"]
            st.markdown("**AAR proxy, test set 2025.** RMSE in arrivals/hour; nRMSE divides by the airport's own 95th percentile.")
            st.dataframe(m[m.airport == ap].round(3), width="stretch")
            if "storm_hours" in set(m.subset):
                sm = m[(m.airport == ap) & (m.subset == "storm_hours")]
                st.markdown("**Storm hours only** (a NOAA Storm Event within 50 km overlaps the target hour; NOAA is used "
                            "only to label hours after the fact, never as an input). This is the case the challenge asks about.")
                st.dataframe(sm[["horizon_h", "n", "rmse_model", "rmse_climatology", "rmse_schedule", "rmse_persistence",
                                 "interval_coverage", "model_beats_climatology"]].round(3), width="stretch")
                st.caption("Errors are much larger in storm hours and the 90% interval covers far less than 90%. "
                           "Treat storm-hour forecasts as a warning signal, not a precise number.")
            bad = m[~m.model_beats_climatology]
            st.markdown("**Where the model does not beat climatology (failure cases):**")
            st.dataframe(bad.round(3) if len(bad) else pd.DataFrame({"note": ["none in this run"]}), width="stretch")
        if "severe_metrics" in mets:
            sv = mets["severe_metrics"]
            st.markdown("**Severe delay (>45 min, cancelled or diverted), test set 2025**")
            st.dataframe(sv[sv.airport == ap].round(3), width="stretch")
            if "subset" in sv:
                worse = sv[(sv.airport == ap) & (sv.subset == "storm_hours") & (sv.pr_auc_model < sv.pr_auc_climatology)]
                if len(worse):
                    st.markdown("**Storm hours where the classifier ranks flights worse than climatology (failure case):**")
                    st.dataframe(worse[["direction", "horizon_h", "n", "pr_auc_model", "pr_auc_climatology"]].round(3), width="stretch")
        if "severe_calibration" in mets:
            c = mets["severe_calibration"]
            fig = go.Figure([go.Scatter(x=c.mean_pred, y=c.obs_rate, mode="markers+lines", name="model"),
                             go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot"), name="perfect")])
            fig.update_layout(height=320, xaxis_title="predicted probability", yaxis_title="observed rate", margin=dict(t=20))
            st.plotly_chart(fig, width="stretch")
        worst = test[test.airport == ap].assign(err=lambda d: d.actual_arrivals - d.pred).sort_values("err").head(10)
        st.markdown("**Largest over-predictions on test (the model said capacity was fine and it was not):**")
        st.dataframe(worst[["target_hour_utc", "horizon_h", "sched_arrivals", "pred", "pred_lo", "actual_arrivals", "err",
                            "obs_flight_cat", "obs_ts_in_hour"]].round(1), width="stretch")

    with tab4:
        st.markdown(f"- Model version: `{model_version()}`; built {datetime.fromtimestamp((REPORTS / 'run_info.json').stat().st_mtime, timezone.utc):%Y-%m-%d %H:%MZ}")
        st.markdown(f"- Severe-delay alert threshold {thr:.2f}, chosen on validation (2024 H2) and frozen for test.")
        st.markdown(f"- Delay cost reference for impact estimates: USD {COST_PER_MIN}/aircraft block minute (A4A, 2025).")
        st.markdown("""**Limitations**
- The AAR here is a proxy: actual arrivals per hour of BTS reporting carriers. It is not the FAA-called Airport Arrival Rate; cargo and GA are missing.
- In quiet hours arrivals measure demand, not capacity. Capacity is scored on high-demand hours.
- Wind components use the challenge's single assumed heading per airport, not the real runway configuration.
- NOAA Storm Events are used only to evaluate, never as inputs, because they are compiled after the event.""")
        if MANIFEST.exists():
            st.markdown("**Source files (data/manifest.jsonl)**")
            st.dataframe(pd.read_json(MANIFEST, lines=True), width="stretch")


if __name__ == "__main__":
    main()
