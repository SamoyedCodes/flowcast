"""
Build leakage-safe modelling tables from the pulled public data.

Outputs (data/processed/):
  metar_hourly.parquet        hourly as-of METAR features per airport (UTC hour = last obs in (H-1h, H])
  airport_hourly.parquet      per airport x UTC hour: scheduled & actual arrivals/departures, delay shares
  aar_horizon.parquet         per airport x target hour x horizon (3/6/9/12 h): features known at
                              t0 = target - horizon, plus targets (actual arrivals = AAR proxy, throughput ratio)
  flights_horizon.parquet     per flight x horizon: features known at t0, target = severe delay (>45 min)

Leakage rules enforced in code:
  * weather features use only METARs with valid <= t0
  * TAF features use only TAFs issued <= t0
  * airport-state features use only flights whose ACTUAL arrival time <= t0
  * inbound aircraft delay is used only if that previous leg actually landed <= t0
  * Storm Events are never features (post-event records); they are evaluation labels only
Split: chronological (train < 2024-07-01 <= val < 2025-01-01 <= test).

Usage: python build_datasets.py --airports ATL DFW --horizons 3 6 9 12 --flight-sample 0.25
"""
import argparse
import re

import numpy as np
import pandas as pd

from common import PROC, load_airports, to_utc_naive

SPLIT_VAL, SPLIT_TEST = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01")


# ---------------------------------------------------------------- METAR features
def ceiling_ft(row_c, row_l):
    """Lowest BKN/OVC/VV layer height in ft; NaN if none (unlimited)."""
    best = np.nan
    for c, l in zip(row_c, row_l):
        if isinstance(c, str) and c.strip() in ("BKN", "OVC", "VV") and pd.notna(l):
            best = l if np.isnan(best) else min(best, l)
    return best


def flight_category(ceil, vis):
    """FAA/AWC: LIFR ceil<500 or vis<1; IFR ceil<1000 or vis<3; MVFR ceil<=3000 or vis<=5; else VFR."""
    c = np.where(np.isnan(ceil), 1e9, ceil)
    v = np.where(np.isnan(vis), 1e9, vis)
    cat = np.select([(c < 500) | (v < 1), (c < 1000) | (v < 3), (c <= 3000) | (v <= 5)],
                    ["LIFR", "IFR", "MVFR"], "VFR")
    return np.where(np.isnan(ceil) & np.isnan(vis), None, cat)


def wind_components(speed_kt, dir_deg, heading_deg):
    """Crosswind = |V sin(theta-h)|; headwind on heading h = V cos(theta-h) (negative = tailwind).
    Direction is where the wind blows FROM (METAR convention), degrees true; runway heading
    treated as true per challenge assumption. Calm wind (0 kt) -> 0 components; missing/VRB direction -> NaN."""
    th = np.deg2rad(dir_deg - heading_deg)
    xw = np.abs(speed_kt * np.sin(th))
    hw = speed_kt * np.cos(th)
    calm = speed_kt == 0
    return np.where(calm, 0.0, xw), np.where(calm, 0.0, hw)


def metar_features(metar, airports):
    m = metar.copy()
    m["valid"] = to_utc_naive(m["valid"])
    for c in ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby", "alti", "mslp", "p01i",
              "skyl1", "skyl2", "skyl3", "skyl4"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["ceiling_ft"] = [ceiling_ft(c, l) for c, l in zip(m[["skyc1", "skyc2", "skyc3", "skyc4"]].values,
                                                         m[["skyl1", "skyl2", "skyl3", "skyl4"]].values)]
    m["flight_cat"] = flight_category(m.ceiling_ft.values, m.vsby.values)
    wx = m["wxcodes"].fillna("").astype(str)
    m["wx_ts"] = wx.str.contains(r"TS").astype(int)
    m["wx_fz"] = wx.str.contains(r"FZ").astype(int)
    m["wx_sn"] = wx.str.contains(r"SN").astype(int)
    m["wx_fg"] = wx.str.contains(r"FG|BR").astype(int)
    m["wx_heavy"] = wx.str.contains(r"\+").astype(int)
    m["temp_c"] = (m.tmpf - 32) * 5 / 9
    m["spread_c"] = (m.tmpf - m.dwpf) * 5 / 9
    m["gust_kt"] = m.gust
    out = []
    for code, g in m.groupby("station"):
        h = airports[code].get("runway_heading_deg")
        g = g.sort_values("valid").copy()
        if h is not None:
            g["xwind_kt"], g["hwind_kt"] = wind_components(g.sknt, g.drct, h)
            g["xwind_gust_kt"], _ = wind_components(g.gust.fillna(g.sknt), g.drct, h)
            g["tailwind_on_assumed_hdg_kt"] = np.clip(-g.hwind_kt, 0, None)
            g["best_end_headwind_kt"] = np.abs(g.hwind_kt)
            g["flow_toward_assumed_hdg"] = np.where(g.hwind_kt >= 0, 1, 0)  # 1 = land on h, 0 = on h+180
        # hourly as-of: last report in (H-1h, H]
        g["hour"] = g.valid.dt.ceil("h")
        keep = ["temp_c", "spread_c", "relh", "sknt", "gust_kt", "drct", "vsby", "ceiling_ft",
                "alti", "mslp", "p01i", "wx_ts", "wx_fz", "wx_sn", "wx_fg", "wx_heavy", "flight_cat",
                "xwind_kt", "hwind_kt", "xwind_gust_kt", "tailwind_on_assumed_hdg_kt",
                "best_end_headwind_kt", "flow_toward_assumed_hdg"]
        keep = [k for k in keep if k in g]
        agg = g.groupby("hour")[keep].last()
        # any thunder / max gust within the hour (still <= H)
        agg["ts_in_hour"] = g.groupby("hour").wx_ts.max()
        agg["max_gust_in_hour_kt"] = g.groupby("hour").gust_kt.max()
        agg = agg.reindex(pd.date_range(agg.index.min(), agg.index.max(), freq="h"))
        agg["obs_missing"] = agg.temp_c.isna().astype(int)
        last_obs = pd.Series(np.where(agg.temp_c.notna(), agg.index, pd.NaT), index=agg.index).ffill(limit=2)
        # forward-fill at most 2 h (past-only) and record how old the carried observation is
        # fill only hours with no report at all; a report without a gust group means no gust
        filled = agg.where(agg.obs_missing == 0, agg.ffill(limit=2))
        filled["obs_missing"] = agg.obs_missing
        filled["obs_age_h"] = (pd.Series(filled.index, index=filled.index) - pd.to_datetime(last_obs)).dt.total_seconds() / 3600
        filled["pres_tend_3h"] = filled.alti - filled.alti.shift(3)
        filled["pres_tend_6h"] = filled.alti - filled.alti.shift(6)
        if "flow_toward_assumed_hdg" in filled:
            fl = filled.flow_toward_assumed_hdg
            filled["flow_change_3h"] = (fl != fl.shift(3)).where(fl.notna() & fl.shift(3).notna()).astype(float)
        filled["airport"] = code
        out.append(filled.rename_axis("hour_utc").reset_index())
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- TAF features
def _tokens(x):
    return re.findall(r"[A-Z]{2,3}", str(x)) if pd.notna(x) else []


def _nums(x):
    return [float(v) for v in re.findall(r"\d+(?:\.\d+)?", str(x))] if pd.notna(x) else []


def taf_hourly(taf):
    """Expand each TAF issuance to hourly prevailing conditions + TEMPO/PROB flags."""
    t = taf.copy()
    for c in ("valid", "fx_valid", "fx_valid_end"):
        t[c] = to_utc_naive(t[c])
    ft = t.get("ftype", pd.Series("", index=t.index)).fillna("").astype(str).str.upper()
    t["is_temp_group"] = (t.get("is_tempo", False).fillna(False).astype(bool)
                          | ft.str.contains("TEMPO|PROB")).astype(int)
    t["ceil_ft"] = [min([l for c, l in zip(_tokens(sc), _nums(sl)) if c in ("BKN", "OVC", "VV")], default=np.nan)
                    for sc, sl in zip(t.skyc, t.skyl)]
    t["ts"] = t.presentwx.fillna("").astype(str).str.contains("TS").astype(int)
    t = t.sort_values(["airport", "valid", "fx_valid"])
    prev = t[t.is_temp_group == 0].copy()
    nxt = prev.groupby(["airport", "valid"]).fx_valid.shift(-1)
    prev["end"] = prev.fx_valid_end.fillna(nxt).fillna(prev.fx_valid + pd.Timedelta(hours=30))
    prev = prev[prev.end > prev.fx_valid]
    prev["hour_utc"] = [list(pd.date_range(a.ceil("h"), b, freq="h", inclusive="left")) for a, b in zip(prev.fx_valid, prev.end)]
    pe = prev.explode("hour_utc").dropna(subset=["hour_utc"])
    # if prevailing groups overlap, the later-starting group (e.g. FM) governs
    pe = pe.sort_values("fx_valid").groupby(["airport", "valid", "hour_utc"]).tail(1)
    pe = pe.rename(columns={"valid": "taf_issued_utc", "sknt": "taf_sknt", "gust": "taf_gust", "drct": "taf_drct",
                            "visibility": "taf_vis_mi", "ceil_ft": "taf_ceil_ft", "ts": "taf_ts"})
    pe = pe[["airport", "taf_issued_utc", "hour_utc", "taf_sknt", "taf_gust", "taf_drct", "taf_vis_mi", "taf_ceil_ft", "taf_ts"]]
    tm = t[(t.is_temp_group == 1) & t.fx_valid_end.notna()].copy()
    if len(tm):
        tm["hour_utc"] = [list(pd.date_range(a.floor("h"), b, freq="h", inclusive="left")) for a, b in zip(tm.fx_valid, tm.fx_valid_end)]
        te = tm.explode("hour_utc").dropna(subset=["hour_utc"])
        te = te.groupby(["airport", "valid", "hour_utc"]).agg(taf_tempo_or_prob_ts=("ts", "max"),
                                                                taf_tempo_min_vis=("visibility", "min"),
                                                                taf_tempo_min_ceil=("ceil_ft", "min")).reset_index()
        te = te.rename(columns={"valid": "taf_issued_utc"})
        out = pe.merge(te, on=["airport", "taf_issued_utc", "hour_utc"], how="left")
    else:
        out = pe.assign(taf_tempo_or_prob_ts=0, taf_tempo_min_vis=np.nan, taf_tempo_min_ceil=np.nan)
    out["taf_tempo_or_prob_ts"] = out.taf_tempo_or_prob_ts.fillna(0).astype(int)
    out["hour_utc"] = pd.to_datetime(out.hour_utc)
    for c in ("taf_sknt", "taf_gust", "taf_drct", "taf_vis_mi", "taf_ceil_ft"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["taf_flight_cat"] = flight_category(out.taf_ceil_ft.values.astype(float), out.taf_vis_mi.values.astype(float))
    return out


def attach_taf(q, tafh):
    """q has airport, target_hour_utc, t0_utc. Take the latest TAF issued <= t0 that covers target hour."""
    q = q.copy()
    r = tafh.rename(columns={"hour_utc": "target_hour_utc"}).copy()
    for d in (q, r):
        for c in ("t0_utc", "target_hour_utc", "taf_issued_utc"):
            if c in d:
                d[c] = to_utc_naive(d[c]).astype("datetime64[ns]")
    q = q.sort_values("t0_utc")
    r = r.sort_values("taf_issued_utc")
    m = pd.merge_asof(q, r, left_on="t0_utc", right_on="taf_issued_utc",
                      by=["airport", "target_hour_utc"], direction="backward",
                      tolerance=pd.Timedelta(hours=12))
    m["taf_age_h"] = (m.t0_utc - m.taf_issued_utc).dt.total_seconds() / 3600
    assert (m.taf_issued_utc.isna() | (m.taf_issued_utc <= m.t0_utc)).all(), "TAF leakage"
    return m


# ---------------------------------------------------------------- flights & airport hourly
def flight_times(f, airports_tz):
    import airportsdata
    tz = {k: v["tz"] for k, v in airportsdata.load("IATA").items()}
    tz.update({"PBI": "America/New_York"})
    tz.update(airports_tz)
    f = f.copy()
    date = pd.to_datetime(f.FL_DATE)
    hhmm = f.CRS_DEP_TIME.astype(int)
    local = date + pd.to_timedelta((hhmm // 100) * 60 + hhmm % 100, unit="m")
    dep = pd.Series(pd.NaT, index=f.index, dtype="datetime64[ns]")
    for a, idx in f.groupby("ORIGIN").groups.items():
        dep.loc[idx] = (local.loc[idx].dt.tz_localize(tz[a], ambiguous="NaT", nonexistent="shift_forward")
                        .dt.tz_convert("UTC").dt.tz_localize(None))
    f["sched_dep_utc"] = dep
    f["sched_arr_utc"] = dep + pd.to_timedelta(f.CRS_ELAPSED_TIME, unit="m")
    flown = (f.CANCELLED == 0) & (f.DIVERTED == 0)
    f["act_dep_utc"] = (f.sched_dep_utc + pd.to_timedelta(f.DEP_DELAY, unit="m")).where(f.CANCELLED == 0)
    f["act_arr_utc"] = (f.sched_arr_utc + pd.to_timedelta(f.ARR_DELAY, unit="m")).where(flown)
    f["severe_arr"] = ((f.ARR_DELAY > 45) | (f.CANCELLED == 1) | (f.DIVERTED == 1)).astype(int)
    f["severe_dep"] = ((f.DEP_DELAY > 45) | (f.CANCELLED == 1)).astype(int)
    return f


def airport_hourly(f, airports):
    out = []
    for code in airports:
        arr = f[f.DEST == code]
        dep = f[f.ORIGIN == code]
        s_arr = arr.groupby(arr.sched_arr_utc.dt.floor("h")).size().rename("sched_arrivals")
        a_arr = arr.dropna(subset=["act_arr_utc"]).groupby(arr.act_arr_utc.dt.floor("h")).size().rename("actual_arrivals")
        s_dep = dep.groupby(dep.sched_dep_utc.dt.floor("h")).size().rename("sched_departures")
        a_dep = dep.dropna(subset=["act_dep_utc"]).groupby(dep.act_dep_utc.dt.floor("h")).size().rename("actual_departures")
        sev = arr.groupby(arr.sched_arr_utc.dt.floor("h")).severe_arr.mean().rename("severe_arr_share")
        cxl = arr.groupby(arr.sched_arr_utc.dt.floor("h")).CANCELLED.sum().rename("arr_cancelled")
        h = pd.concat([s_arr, a_arr, s_dep, a_dep, sev, cxl], axis=1, sort=True)
        h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="h")).fillna(
            {"sched_arrivals": 0, "actual_arrivals": 0, "sched_departures": 0, "actual_departures": 0, "arr_cancelled": 0})
        h["throughput_ratio"] = (h.actual_arrivals / h.sched_arrivals).where(h.sched_arrivals >= 5)
        # demand-capped hours: scheduled demand in the top 30% for the airport -> arrivals closer to capacity
        h["high_demand_hour"] = (h.sched_arrivals >= h.sched_arrivals[h.sched_arrivals > 0].quantile(0.7)).astype(int)
        # normalised by the airport's own typical peak (95th pct of hourly arrivals) -> comparable across airports
        h["arrivals_norm"] = h.actual_arrivals / h.actual_arrivals.quantile(0.95)
        h["airport"] = code
        out.append(h.rename_axis("hour_utc").reset_index())
    return pd.concat(out, ignore_index=True)


def state_at_t0(f, code, t0):
    """Arrivals at `code` that actually landed in (t0-2h, t0]: count, mean delay, share >45 min.
    Uses only flights already landed at t0 (no leakage)."""
    a = f[(f.DEST == code) & f.act_arr_utc.notna()].sort_values("act_arr_utc")
    T = a.act_arr_utc.values.astype("datetime64[ns]")
    D = a.ARR_DELAY.values.astype(float)
    S = (a.ARR_DELAY.values > 45).astype(float)
    cD, cS = np.concatenate([[0], np.cumsum(D)]), np.concatenate([[0], np.cumsum(S)])
    t0 = np.asarray(t0, dtype="datetime64[ns]")
    hi = np.searchsorted(T, t0, side="right")
    lo = np.searchsorted(T, t0 - np.timedelta64(2, "h"), side="right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({"st_arr_2h": n,
                             "st_mean_arr_delay_2h": np.where(n > 0, (cD[hi] - cD[lo]) / n, np.nan),
                             "st_share_gt45_2h": np.where(n > 0, (cS[hi] - cS[lo]) / n, np.nan)})


def split_of(ts):
    return np.select([ts < SPLIT_VAL, ts < SPLIT_TEST], ["train", "val"], "test")


def main(airport_codes, horizons, flight_sample, seed=7):
    aps = load_airports(airport_codes)
    f = pd.read_parquet(PROC / "bts_flights.parquet")
    f = flight_times(f, {k: v["tz"] for k, v in aps.items()})
    metar = pd.read_parquet(PROC / "metar.parquet")
    mh = metar_features(metar, aps)
    mh.to_parquet(PROC / "metar_hourly.parquet", index=False)
    have_taf = (PROC / "taf.parquet").exists()
    tafh = taf_hourly(pd.read_parquet(PROC / "taf.parquet")) if have_taf else None
    ah = airport_hourly(f, aps)
    ah.to_parquet(PROC / "airport_hourly.parquet", index=False)
    print(f"airport_hourly: {len(ah):,} rows")

    wx_cols = [c for c in mh.columns if c not in ("hour_utc", "airport")]
    # ---------- AAR horizon table ----------
    rows = []
    for H in horizons:
        q = ah[["airport", "hour_utc", "sched_arrivals", "actual_arrivals", "throughput_ratio",
                "arrivals_norm", "high_demand_hour", "severe_arr_share"]].rename(columns={"hour_utc": "target_hour_utc"})
        q["horizon_h"] = H
        q["t0_utc"] = q.target_hour_utc - pd.Timedelta(hours=H)
        # weather as of t0 (hour_utc == t0 means last obs in (t0-1h, t0])
        q = q.merge(mh.rename(columns={"hour_utc": "t0_utc", **{c: f"obs_{c}" for c in wx_cols}}),
                    on=["airport", "t0_utc"], how="left")
        # schedule known in advance: demand next hours
        lag = ah[["airport", "hour_utc", "sched_arrivals"]].rename(columns={"hour_utc": "target_hour_utc", "sched_arrivals": "sched_arrivals_next1h"})
        lag["target_hour_utc"] -= pd.Timedelta(hours=1)
        q = q.merge(lag, on=["airport", "target_hour_utc"], how="left")
        # airport state at t0
        parts = []
        for code, g in q.groupby("airport"):
            parts.append(pd.concat([g, state_at_t0(f, code, g.t0_utc.values).set_index(g.index)], axis=1))
        q = pd.concat(parts)
        if tafh is not None:
            q = attach_taf(q, tafh)
        rows.append(q)
    aar = pd.concat(rows, ignore_index=True)
    aar["target_hour_local"] = -1
    for code, idx in aar.groupby("airport").groups.items():
        aar.loc[idx, "target_hour_local"] = aar.loc[idx, "target_hour_utc"].dt.tz_localize("UTC").dt.tz_convert(aps[code]["tz"]).dt.hour
    aar["dow"] = aar.target_hour_utc.dt.dayofweek
    aar["month"] = aar.target_hour_utc.dt.month
    aar["split"] = split_of(aar.target_hour_utc)
    aar.to_parquet(PROC / "aar_horizon.parquet", index=False)
    print(f"aar_horizon: {len(aar):,} rows")

    # ---------- flight-level horizon table ----------
    ff = f[f.DEST.isin(aps) | f.ORIGIN.isin(aps)]
    if flight_sample < 1:
        ff = ff.sample(frac=flight_sample, random_state=seed)
    # inbound previous leg by tail
    legs = f.dropna(subset=["TAIL_NUM"]).sort_values(["TAIL_NUM", "sched_dep_utc"])
    legs["prev_sched_arr_utc"] = legs.groupby("TAIL_NUM").sched_arr_utc.shift()
    legs["prev_act_arr_utc"] = legs.groupby("TAIL_NUM").act_arr_utc.shift()
    legs["prev_arr_delay"] = legs.groupby("TAIL_NUM").ARR_DELAY.shift()
    # only a true rotation: previous leg ends where this one starts, within 18 h
    cont = ((legs.groupby("TAIL_NUM").DEST.shift() == legs.ORIGIN)
            & ((legs.sched_dep_utc - legs.prev_sched_arr_utc) <= pd.Timedelta(hours=18)))
    legs.loc[~cont, ["prev_sched_arr_utc", "prev_act_arr_utc", "prev_arr_delay"]] = np.nan
    ff = ff.join(legs[["prev_sched_arr_utc", "prev_act_arr_utc", "prev_arr_delay"]], how="left")
    out = []
    for H in horizons:
        # a flight between two focus airports appears twice: as a departure and as an arrival
        g = pd.concat([ff[ff.DEST.isin(aps)].assign(focus_airport=lambda x: x.DEST, direction="arrival"),
                       ff[ff.ORIGIN.isin(aps)].assign(focus_airport=lambda x: x.ORIGIN, direction="departure")])
        g["event_utc"] = np.where(g.direction == "arrival", g.sched_arr_utc, g.sched_dep_utc)
        g["event_utc"] = pd.to_datetime(g.event_utc)
        g["horizon_h"] = H
        g["t0_utc"] = g.event_utc - pd.Timedelta(hours=H)
        g["t0_hour"] = g.t0_utc.dt.floor("h")
        g = g.merge(mh.rename(columns={"hour_utc": "t0_hour", "airport": "focus_airport", **{c: f"obs_{c}" for c in wx_cols}}),
                    on=["focus_airport", "t0_hour"], how="left")
        known = g.prev_act_arr_utc <= g.t0_utc
        g["inbound_delay_known"] = known.astype(int)
        g["inbound_prev_arr_delay"] = g.prev_arr_delay.where(known)
        g["target_hour_utc"] = g.event_utc.dt.floor("h")
        g = g.rename(columns={"focus_airport": "airport"})
        g = g.merge(ah[["airport", "hour_utc", "sched_arrivals", "sched_departures"]].rename(columns={"hour_utc": "target_hour_utc"}),
                    on=["airport", "target_hour_utc"], how="left")
        if tafh is not None:
            g = attach_taf(g, tafh)
        g["y_severe"] = np.where(g.direction == "arrival", g.severe_arr, g.severe_dep)
        out.append(g)
    fl = pd.concat(out, ignore_index=True)
    # drop post-event columns from the feature table to prevent accidental leakage
    post = ["DEP_TIME", "DEP_DELAY", "TAXI_OUT", "TAXI_IN", "ARR_TIME", "ARR_DELAY", "CANCELLATION_CODE",
            "ACTUAL_ELAPSED_TIME", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY", "act_dep_utc", "act_arr_utc", "prev_act_arr_utc", "prev_arr_delay",
            "severe_arr", "severe_dep", "CANCELLED", "DIVERTED"]
    fl = fl.drop(columns=[c for c in post if c in fl])
    fl["split"] = split_of(fl.event_utc)
    fl.to_parquet(PROC / "flights_horizon.parquet", index=False)
    print(f"flights_horizon: {len(fl):,} rows; severe rate {fl.y_severe.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--airports", nargs="+", default=["ATL", "DFW"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 9, 12])
    ap.add_argument("--flight-sample", type=float, default=0.25,
                    help="fraction of flights to keep for the flight-level table (memory)")
    a = ap.parse_args()
    main(a.airports, a.horizons, a.flight_sample)
