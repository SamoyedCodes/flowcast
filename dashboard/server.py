"""Capacity & Risk web dashboard: a small JSON API over the pipeline outputs, plus the page in dashboard/web/.
Replays the held-out 2025 period exactly like dashboard/app.py: the forecasts are what the models trained on data
before 2025 would have said at the chosen issue time.   Run:  python dashboard/server.py   ->  http://localhost:8502
"""
import json
import math
import sys
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import advisory, config as C  # noqa: E402
from src import train_delay as TD  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
PORT = 8502
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


def nice(f):
    if f in NAMES:
        return NAMES[f]
    return "TAF: " + f[4:].replace("_", " ") if f.startswith("taf_") else f.replace("_", " ")


def num(x, nd=1):
    """JSON-safe number: NaN/inf/None -> None."""
    if x is None or isinstance(x, str):
        return x
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def ts(t):
    return None if pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%MZ")


# ------------------------------------------------------------------ data (loaded once)
HOURLY = pd.read_csv(C.PROC / "hourly_weather.csv", parse_dates=["hour_utc"])
AAR = pd.read_parquet(C.PROC / "aar_predictions.parquet").query("split == 'test'")
DL = pd.read_parquet(C.PROC / "delay_test_predictions.parquet")
REP = {p.stem: pd.read_csv(p) for p in C.REPORTS.glob("*.csv")}
H = {a: g.set_index("hour_utc").sort_index() for a, g in HOURLY.groupby("airport")}
AP = pd.read_csv(C.EXTERNAL / "airports.dat", header=None, usecols=[1, 2, 4, 5, 6, 7], names=["name", "city", "iata", "icao", "lat", "lon"])
AP = AP[AP["iata"].str.len() == 3].drop_duplicates("iata").set_index("iata")
TEST0, TEST1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-31 11:00", tz="UTC")


def bearing(a, b):
    """Initial great-circle bearing from airport a to b, degrees from north (None if either is unknown)."""
    if a not in AP.index or b not in AP.index:
        return None
    la1, lo1, la2, lo2 = map(math.radians, (AP.at[a, "lat"], AP.at[a, "lon"], AP.at[b, "lat"], AP.at[b, "lon"]))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return round(math.degrees(math.atan2(y, x)) % 360)


@lru_cache(maxsize=1)
def delay_models():
    t = TD.prep(pd.read_parquet(C.PROC / "delay_table.parquet"))
    t = t[t["split"] == "test"].set_index(["flight_id", "direction", "airport", "lead_h"], drop=False)
    return t, {lead: joblib.load(C.MODELS / f"delay_lead{lead}.joblib") for lead in range(3, 13)}


def drivers(fl):
    """Top three features pushing each flight's risk above the average flight (log-odds contributions)."""
    t, models = delay_models()
    out = {}
    for lead, g in fl.groupby("lead_h"):
        contrib = TD.explain_bundle(models[int(lead)], t.loc[list(zip(g["flight_id"], g["direction"], g["airport"], g["lead_h"]))])
        feats, vals = list(contrib.columns), contrib.to_numpy()
        for i, idx in enumerate(g.index):
            out[idx] = [[nice(feats[j]), round(float(vals[i, j]), 2)] for j in np.argsort(-vals[i])[:3] if vals[i, j] > 0.05]
    return out


# ------------------------------------------------------------------ endpoints
def meta():
    airports, events = [], []
    for icao, cfg in C.AIRPORTS.items():
        iata = cfg["iata"]
        airports.append({"icao": icao, "iata": iata, "name": AP.at[iata, "name"] if iata in AP.index else iata,
                         "tz": cfg["tz"], "runway_heading": cfg["runway_heading_deg"], "aar": cfg["aar"]})
        test = H[icao].loc[TEST0:][lambda d: d["aar_ratio"].notna()]
        worst = test.groupby(test.index.floor("D"))["aar_ratio"].idxmin()
        for t in sorted(worst, key=lambda t: test.at[t, "aar_ratio"])[:5]:
            events.append({"icao": icao, "iata": iata, "t0": ts(max(t - pd.Timedelta(hours=6), TEST0)), "at": ts(t),
                           "aar": num(test.at[t, "aar"], 0), "ratio": num(test.at[t, "aar_ratio"], 2)})
    events.sort(key=lambda e: e["ratio"])
    return {"airports": airports, "events": events, "start": ts(TEST0), "end": ts(TEST1),
            "flights": int(DL.loc[DL["lead_h"] == 3, "flight_id"].nunique()), "hours": int(AAR["issue_utc"].nunique()),
            "base_rate": num(DL["severe_delay"].mean(), 4), "default": {"icao": "KATL", "t0": "2025-03-15T06:00Z"}}


def weather_row(r):
    if r is None:
        return None
    return {"hour": ts(r.name), "category": r["flight_category"] if isinstance(r["flight_category"], str) else None,
            "aar": num(r["aar"], 0), "ratio": num(r["aar_ratio"], 3), "wind_dir": num(r["wind_dir_deg"], 0),
            "wind_kt": num(r["wind_kt"], 0), "gust_kt": num(r["gust_kt"], 0), "crosswind_kt": num(r["crosswind_kt"], 0),
            "headwind_kt": num(r["headwind_kt"], 0), "ceiling_ft": num(r["ceiling_ft"], 0), "vis_mi": num(r["vis_mi"], 1),
            "wx": r["wxcodes"] if isinstance(r["wxcodes"], str) else "", "thunder": bool(r["thunder"] == 1),
            "winter": bool(r["winter"] == 1), "temp_c": num(r["temp_c"], 0)}


def outlook(icao, t0, sens, thr):
    cfg, h_ap = C.AIRPORTS[icao], H[icao]
    fc = AAR[(AAR["airport"] == icao) & (AAR["issue_utc"] == t0)].sort_values("hour_utc").copy()
    if len(fc) < 10:
        return None
    basis = "pred_ratio" if sens == "balanced" else "lo_ratio"
    wins = advisory.capacity_windows(fc, basis)

    d_all = DL[DL["airport"] == icao].copy()
    d_all["h_ahead"] = (d_all["event_utc"] - t0) / pd.Timedelta(hours=1)
    fl = d_all[(d_all["h_ahead"] >= 2.5) & (d_all["h_ahead"] < 12.5) & (d_all["lead_h"] == d_all["h_ahead"].round().clip(3, 12))]
    fl = fl.sort_values("p", ascending=False)
    n_high = int((fl["p"] >= thr).sum())
    why = drivers(fl) if len(fl) else {}

    prev_t = t0 - pd.Timedelta(hours=1)
    now = weather_row(h_ap.loc[prev_t]) if prev_t in h_ap.index else None
    lo_row = fc.loc[fc["pred_ratio"].idxmin()]
    v = cfg["aar"]["VFR"]
    if any(w["severity"] == "severe" for w in wins):
        verdict = {"kind": "severe", "title": f"Severe capacity reduction expected around {lo_row['hour_utc']:%H:%MZ}",
                   "sub": f"AAR about {lo_row['pred_ratio'] * v:.0f}/h, could be as low as {lo_row['lo_ratio'] * v:.0f}/h, "
                          f"{100 * (1 - lo_row['pred_ratio']):.0f}% below best case."}
    elif wins:
        verdict = {"kind": "moderate", "title": f"Capacity reduction expected around {lo_row['hour_utc']:%H:%MZ}",
                   "sub": f"Lowest forecast AAR about {lo_row['pred_ratio'] * v:.0f}/h, could be as low as {lo_row['lo_ratio'] * v:.0f}/h."}
    else:
        verdict = {"kind": "clear", "title": "No capacity reduction forecast in the next 3 to 12 hours",
                   "sub": "Routine monitoring. Check again at the next METAR or TAF update."}

    def sev(r):
        return "severe" if r <= advisory.SEVERE else "moderate" if r <= advisory.MODERATE else "clear"

    hist = h_ap.loc[t0 - pd.Timedelta(hours=24): prev_t]
    return {
        "icao": icao, "iata": cfg["iata"], "t0": ts(t0), "local": t0.tz_convert(cfg["tz"]).strftime("%H:%M %Z"),
        "aar_vfr": v, "now": now, "verdict": verdict, "n_flights": len(fl), "n_high": n_high, "thr": thr, "sens": sens,
        "hist": [{"hour": ts(t), "aar": num(a, 0)} for t, a in hist["aar"].items()],
        "hours": [{"hour": ts(r.hour_utc), "lead": int(r.lead_h), "pred": num(r.pred_ratio * v, 0), "lo": num(r.lo_ratio * v, 0),
                   "hi": num(r.hi_ratio * v, 0), "taf": num(r.taf_aar_ratio * v, 0), "persist": num(r.persist_ratio * v, 0),
                   "actual": num(r.y_ratio * v, 0), "ratio": num(getattr(r, basis), 3), "sev": sev(getattr(r, basis)),
                   "hazards": advisory.hazards_for_hour(r._asdict())} for r in fc.itertuples()],
        "windows": [{"start": ts(w["start"]), "end": ts(w["end"]), "severity": w["severity"], "min_aar": num(w["min_aar"], 0),
                     "lo": num(w["min_aar_low"], 0), "hi": num(w["min_aar_high"], 0), "pct": num(w["pct_reduction"], 0),
                     "confidence": w["confidence"], "hazards": [advisory.HAZARD_TEXT[z] for z in w["hazards"]],
                     "advice": advisory.advice(w, n_high)} for w in wins],
        "flights": [{"id": int(r.flight_id), "dir": r.direction, "other": r.other_airport, "carrier": r.carrier,
                     "t": ts(r.event_utc), "h": num(r.h_ahead, 1), "p": num(r.p, 3), "level": str(advisory.risk_level([r.p])[0]),
                     "drivers": why.get(r.Index, []), "delay_min": num(r.delay_min, 0), "severe": int(r.severe_delay),
                     "bearing": bearing(cfg["iata"], r.other_airport)} for r in fl.itertuples()],
    }


def history(icao, t0):
    cfg, h_ap = C.AIRPORTS[icao], H[icao]
    past = h_ap.loc[t0 - pd.Timedelta(days=30): t0]
    daily = past.groupby(past.index.floor("D")).agg(avg=("aar_ratio", "mean"), low=("aar_ratio", lambda s: int((s <= advisory.MODERATE).sum())))
    dd = DL[(DL["airport"] == icao) & (DL["lead_h"] == 3) & (DL["event_utc"] <= t0) & (DL["event_utc"] >= t0 - pd.Timedelta(days=30))]
    dday = dd.groupby(dd["event_utc"].dt.floor("D")).agg(pred=("p", "mean"), obs=("severe_delay", "mean"), n=("p", "size"))
    test = h_ap.loc[TEST0:][lambda d: d["aar_ratio"].notna()]
    local = test.index.tz_convert(cfg["tz"])
    grid = (test["aar_ratio"] <= advisory.MODERATE).groupby([local.dayofweek, local.hour]).mean().unstack()
    return {"icao": icao, "iata": cfg["iata"], "t0": ts(t0),
            "daily": [{"day": ts(d), "low": int(r.low), "mean": num(r.avg, 3)} for d, r in daily.iterrows()],
            "delay": [{"day": ts(d), "pred": num(r.pred, 3), "obs": num(r.obs, 3), "n": int(r.n)} for d, r in dday.iterrows()],
            "heat": [[num(v, 3) for v in row] for row in grid.reindex(index=range(7), columns=range(24)).to_numpy()]}


def evaluation(icao, lead):
    am, dm = REP["aar_metrics"], REP["delay_metrics"]
    a = am[(am.scope == "all") & (am.lead_h == lead)].set_index("model")
    dsel = dm[(dm.scope == "all") & (dm.lead_h == lead)]

    def dv(m, variant="calibrated"):
        return num((dsel[(dsel.model == m) & (dsel.variant == variant)]["pr_auc"].tolist() or [np.nan])[0], 3)

    aar = [{"model": m, "label": lbl, "rmse": num(a.at[m, "rmse_aar"], 1), "pr_auc": num(a.at[m, "event_pr_auc"], 2)}
           for m, lbl in (("lightgbm", "Our model"), ("taf_implied", "TAF-implied"), ("persistence", "Persistence"), ("climatology", "Climatology"))
           if m in a.index]
    out = {"lead": lead, "aar": aar, "event_rate": num(a.at["lightgbm", "event_rate"], 3),
           "band": num(a.at["lightgbm_80pct_band", "band_coverage"], 3) if "lightgbm_80pct_band" in a.index else None,
           "delay": {"full": dv(TD.FINAL), "sched": dv("A_schedule"), "lightgbm": dv("lightgbm_comparison", "raw"),
                     "chance": dv("constant_base_rate", "raw")}, "tune": lead in TD.TUNE_LEADS}
    if out["tune"]:
        steps = [("A_schedule", "Schedule only"), ("B_plus_recent_hub", "+ recent hub delays"), ("C_plus_metar", "+ METAR weather"),
                 ("D_plus_taf", "+ TAF forecast"), (TD.FINAL, "+ BTS airport traffic")]
        boot = REP["delay_bootstrap"].set_index("lead_h").loc[lead]
        out["ladder"] = [{"label": lbl, "pr_auc": dv(m), "final": m == TD.FINAL} for m, lbl in steps]
        out["boot"] = {"gain": num(boot.pr_auc_gain_mean, 3), "lo": num(boot.ci_low, 3), "hi": num(boot.ci_high, 3)}
        rel = REP.get(f"delay_reliability_lead{lead}")
        out["reliability"] = [{"pred": num(r.mean_pred, 4), "obs": num(r.observed, 4), "n": int(r.n)} for r in rel.itertuples()] if rel is not None else []
        fc = REP.get(f"delay_failure_cases_lead{lead}")
        out["failures"] = [] if fc is None else [
            {"kind": "missed" if r.kind == "missed_severe_delay" else "false alarm", "hub": C.AIRPORTS.get(r.airport, {}).get("iata", r.airport),
             "dir": r.direction, "other": r.other_airport, "carrier": r.carrier, "t": ts(pd.Timestamp(r.event_utc)),
             "delay_min": num(r.delay_min, 0), "p": num(r.p, 3)} for r in fc.head(8).itertuples()]
    tr = REP.get("aar_vs_traffic")
    if tr is not None:
        tr = tr[tr.airport == icao]
        out["traffic"] = {"bands": [{"label": lbl, "share": num(s, 3), "hours": int(n)} for lbl, s, n in
                                    zip(["AAR below 70%", "AAR 70–85%", "AAR 85–95%", "AAR 95%+"], tr.actual_to_scheduled, tr.busy_hours)],
                          "spearman": num(tr.spearman_aar_vs_completion.iloc[0], 2) if len(tr) else None}
    return out


# ------------------------------------------------------------------ server
def parse_t0(s):
    t = pd.Timestamp((s or "").rstrip("Z"))
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return min(max(t.floor("h"), TEST0), TEST1)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def log_message(self, *a):
        pass

    def send_json(self, code, body):
        data = json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return super().do_GET()
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        icao = q.get("airport", "KATL")
        if icao not in C.AIRPORTS:
            return self.send_json(400, {"error": f"unknown airport {icao}"})
        try:
            if u.path == "/api/meta":
                return self.send_json(200, meta_cached())
            if u.path == "/api/outlook":
                t0 = parse_t0(q.get("t0"))
                body = outlook(icao, t0, "cautious" if q.get("sens") == "cautious" else "balanced", float(q.get("thr", 0.25)))
                return self.send_json(200, body) if body else self.send_json(404, {"error": f"No forecast issued at {ts(t0)}."})
            if u.path == "/api/history":
                return self.send_json(200, history(icao, parse_t0(q.get("t0"))))
            if u.path == "/api/eval":
                return self.send_json(200, evaluation(icao, int(q.get("lead", 3))))
        except (ValueError, KeyError) as e:
            return self.send_json(400, {"error": str(e)})
        self.send_json(404, {"error": "no such endpoint"})


meta_cached = lru_cache(maxsize=1)(meta)

if __name__ == "__main__":
    print(f"Capacity & Risk on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
