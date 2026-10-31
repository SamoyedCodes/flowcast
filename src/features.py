"""Feature tables for the two prediction tasks.

AAR table   : one row per (airport, target hour T, lead h).  Features come from the latest METAR hour that
              had been *published* at the issue time T - h, plus the newest TAF issued before T - h.
Delay table : one row per (flight event, lead h).  The flight event is a departure from, or an arrival at,
              a configured hub; the issue time is the scheduled event time minus h.  Airport-wide BTS traffic
              context comes from stage1_pipeline.attach_bts.

Nothing recorded after the issue time is ever used (asserted in code).  Weather whose last real observation
is more than MAX_WX_AGE old at the issue time is treated as unknown, not carried forward.
"""
import numpy as np
import pandas as pd

import stage1_pipeline
from . import config as C, taf as tafmod

NO_CEILING_FT = 12_000                      # fill for "no ceiling reported" (a has_ceiling flag keeps the information)
FCAT = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}
MAX_WX_AGE = pd.Timedelta(hours=2)          # same limit as stage1_pipeline.attach_metar
PURGE = pd.Timedelta(days=1)                # events this soon after the test start are dropped


def split_of(ts):
    """train / val / test by UTC time; shared by both tasks so no leakage across them.  The first PURGE of the
    test period is 'purged' so test labels never sit next to the last pre-test training labels."""
    ts = pd.DatetimeIndex(ts)
    test = pd.Timestamp(C.VAL_END, tz="UTC")
    return np.select([ts >= test + PURGE, ts >= test, ts >= pd.Timestamp(C.TRAIN_END, tz="UTC")],
                     ["test", "purged", "val"], "train")


# --------------------------------------------------------------------------------------------------
# hourly weather -> features that describe "the state of the weather as published at time known_at_utc"
# --------------------------------------------------------------------------------------------------
def hourly_features(hourly):
    out = []
    for ap, g in hourly.groupby("airport"):
        g = g.set_index("hour_utc").sort_index()
        f = pd.DataFrame(index=g.index)
        f["airport"] = ap
        f["known_at_utc"] = g["known_at_utc"]
        f["wx_obs_missing"] = (g["n_obs"] == 0).astype(int)
        has = g["n_obs"] > 0                                    # "not reported" defaults only apply to real reports
        f["last_obs_utc"] = g["known_at_utc"].where(has).ffill()
        fill = ["temp_c", "dewpt_c", "slp_hpa", "wind_kt", "peak_wind_kt", "headwind_kt", "crosswind_kt", "xwind_max_kt",
                "tailwind_max_kt", "vis_mi", "vis_min_mi", "aar_ratio"]
        g = g.copy()
        g[fill] = g[fill].ffill(limit=3)                       # bridge short gaps only
        f["temp_c"] = g["temp_c"]
        f["dewpt_spread_c"] = g["temp_c"] - g["dewpt_c"]
        f["slp_hpa"] = g["slp_hpa"]
        f["wind_kt"] = g["wind_kt"]
        f["gust_kt"] = g["gust_kt"].fillna(0.0).where(has)      # null in a report = no gust
        f["has_gust"] = g["gust_kt"].notna().astype(float).where(has)
        f["peak_wind_kt"] = g["peak_wind_kt"]
        f["headwind_kt"] = g["headwind_kt"]                      # >0 headwind, <0 tailwind
        f["crosswind_abs_kt"] = g["crosswind_kt"].abs()
        f["xwind_max_kt"] = g["xwind_max_kt"]
        f["tailwind_max_kt"] = g["tailwind_max_kt"]
        f["vis_mi"] = g["vis_mi"]
        f["vis_min_mi"] = g["vis_min_mi"]
        f["has_ceiling"] = g["ceiling_ft"].notna().astype(float).where(has)   # null in a report = no cloud ceiling
        f["ceiling_ft"] = g["ceiling_ft"].fillna(NO_CEILING_FT).where(has)
        f["ceiling_min_ft"] = g["ceiling_min_ft"].fillna(NO_CEILING_FT).where(has)
        f["precip_mm"] = g["precip_mm"].fillna(0.0).where(has)
        f["thunder"] = g["thunder"].astype(float).where(has)
        f["winter"] = g["winter"].astype(float).where(has)
        f["wxcode_present"] = g["wxcodes"].notna().astype(float).where(has)
        f["flight_cat"] = g["flight_category"].map(FCAT)
        f["aar_ratio_now"] = g["aar_ratio"]
        # 3-hour trends
        for c in ("wind_kt", "vis_mi", "ceiling_ft", "slp_hpa", "temp_c", "aar_ratio_now"):
            f[f"{c}_d3h"] = f[c] - f[c].shift(3)
        # worst cases over the previous 6 / 12 hours
        f["xwind_max_6h"] = f["xwind_max_kt"].rolling(6, min_periods=1).max()
        f["gust_max_6h"] = f["gust_kt"].rolling(6, min_periods=1).max()
        f["vis_min_6h"] = f["vis_min_mi"].rolling(6, min_periods=1).min()
        f["ceiling_min_6h"] = f["ceiling_min_ft"].rolling(6, min_periods=1).min()
        f["precip_6h_mm"] = f["precip_mm"].rolling(6, min_periods=1).sum()
        f["thunder_6h"] = f["thunder"].rolling(6, min_periods=1).max()
        f["flight_cat_max_6h"] = f["flight_cat"].rolling(6, min_periods=1).max()
        f["aar_ratio_min_6h"] = f["aar_ratio_now"].rolling(6, min_periods=1).min()
        f["aar_ratio_min_12h"] = f["aar_ratio_now"].rolling(12, min_periods=1).min()
        f["aar_ratio_mean_24h"] = f["aar_ratio_now"].rolling(24, min_periods=1).mean()
        out.append(f.reset_index(names="obs_hour_utc"))
    return pd.concat(out, ignore_index=True).sort_values("known_at_utc")


def attach_state(df, hw, issue_col="issue_utc"):
    """Attach, per row, the newest hourly weather state published at or before the issue time.  If the last real
    observation is older than MAX_WX_AGE the row is kept but its weather is unknown (wx_stale = 1)."""
    df = df.sort_values(issue_col)
    m = pd.merge_asof(df, hw, left_on=issue_col, right_on="known_at_utc", by="airport", direction="backward")
    m = m.dropna(subset=["known_at_utc"])        # issue time earlier than the first observation
    assert (m["known_at_utc"] <= m[issue_col]).all(), "leakage: observation published after the issue time"
    m["wx_age_h"] = (m[issue_col] - m["obs_hour_utc"]).dt.total_seconds() / 3600
    stale = ~((m[issue_col] - m["last_obs_utc"]) <= MAX_WX_AGE)
    wx = [c for c in hw.columns if c not in ("airport", "known_at_utc", "obs_hour_utc", "last_obs_utc", "wx_obs_missing")]
    m.loc[stale, wx] = np.nan
    m["wx_stale"] = stale.astype(int)
    return m.drop(columns=["known_at_utc", "last_obs_utc"])


def calendar_features(local_ts, prefix=""):
    t = pd.DatetimeIndex(local_ts)
    h = t.hour + t.minute / 60
    return pd.DataFrame({f"{prefix}hour_local": t.hour, f"{prefix}dow": t.dayofweek, f"{prefix}month": t.month,
                         f"{prefix}is_weekend": (t.dayofweek >= 5).astype(int),
                         f"{prefix}hour_sin": np.sin(2 * np.pi * h / 24), f"{prefix}hour_cos": np.cos(2 * np.pi * h / 24),
                         f"{prefix}doy_sin": np.sin(2 * np.pi * t.dayofyear / 366),
                         f"{prefix}doy_cos": np.cos(2 * np.pi * t.dayofyear / 366)}, index=None)


# --------------------------------------------------------------------------------------------------
# Task 1: AAR table
# --------------------------------------------------------------------------------------------------
AAR_LEADS = tuple(range(3, 13))


def build_aar_table(hourly, taf_table, leads=AAR_LEADS):
    hw = hourly_features(hourly)
    tgt = hourly[hourly["n_obs"] > 0][["airport", "hour_utc", "aar", "aar_ratio", "aar_vfr", "flight_category"]].copy()
    tgt = tgt.rename(columns={"aar": "y_aar", "aar_ratio": "y_ratio", "flight_category": "y_flight_cat"})
    parts = []
    for lead in leads:
        d = tgt.copy()
        d["lead_h"] = lead
        d["issue_utc"] = d["hour_utc"] - pd.Timedelta(hours=lead)
        parts.append(attach_state(d, hw))
    t = pd.concat(parts, ignore_index=True)
    tt = taf_table[taf_table["lead_h"].isin(leads)]
    t = t.merge(tt, on=["airport", "hour_utc", "lead_h"], how="left")
    # calendar of the TARGET hour in the airport's local time
    loc = pd.Series(index=t.index, dtype="datetime64[ns]")
    for icao, cfg in C.AIRPORTS.items():
        m = t["airport"] == icao
        loc[m] = t.loc[m, "hour_utc"].dt.tz_convert(cfg["tz"]).dt.tz_localize(None)
    t = pd.concat([t, calendar_features(loc).set_index(t.index)], axis=1)
    t["persist_ratio"] = t["aar_ratio_now"]                 # baseline: last published AAR carried forward
    t["split"] = split_of(t["hour_utc"])
    return t.sort_values(["airport", "hour_utc", "lead_h"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# Task 2: delay table
# --------------------------------------------------------------------------------------------------
def _to_utc(local, tz):
    return local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def flight_events(fl):
    """Turn the flight file into hub events: a departure from a hub or an arrival at a hub."""
    fl = fl.copy()
    fl["icao_o"] = fl["ORIGIN"].map(C.IATA_TO_ICAO)
    fl["icao_d"] = fl["DEST"].map(C.IATA_TO_ICAO)
    ev = []
    for direction, hub_col, time_col, delay_col, other in [("dep", "icao_o", "sched_dep", "DEP_DELAY", "DEST"),
                                                           ("arr", "icao_d", "sched_arr", "ARR_DELAY", "ORIGIN")]:
        d = fl[fl[hub_col].notna()].copy()
        d["airport"] = d[hub_col]
        d["direction"] = direction
        d["delay_min"] = d[delay_col]
        d["other_airport"] = d[other]
        d["event_utc"] = pd.concat([_to_utc(g[time_col], C.AIRPORTS[a]["tz"]) for a, g in d.groupby("airport")])
        d["sched_local"] = d[time_col]
        ev.append(d)
    ev = pd.concat(ev, ignore_index=True)
    print(f"dropped {int(ev['event_utc'].isna().sum())} events with an ambiguous/nonexistent local (DST) time")
    ev = ev.dropna(subset=["event_utc"])
    ev["cancelled"] = ev["CANCELLED"].fillna(0).astype(int)
    ev["diverted"] = ev["DIVERTED"].fillna(0).astype(int)
    return ev


def recent_disruption(events, rows, windows=(6, 24)):
    """For each row (airport, issue_utc) summarise hub flights whose ACTUAL departure/arrival time was already
    in the past at the issue time: how many, mean delay, share of severe delays."""
    ok = events[(events["cancelled"] == 0) & (events["diverted"] == 0) & events["delay_min"].notna()].copy()
    ok["actual_utc"] = ok["event_utc"] + pd.to_timedelta(ok["delay_min"], unit="m")
    ok["clip_delay"] = ok["delay_min"].clip(-30, 300)
    ok["severe"] = (ok["delay_min"] > C.SEVERE_DELAY_MIN).astype(float)
    res = {}
    for direction in ("dep", "arr"):
        for w in windows:
            for stat in ("n", "mean", "severe"):
                res[f"rec_{direction}_{stat}_{w}h"] = np.full(len(rows), np.nan)
    ap_arr = rows["airport"].values
    issue_ns = rows["issue_utc"].values.astype("datetime64[ns]").astype("int64")
    for (ap, direction), g in ok.groupby(["airport", "direction"]):
        g = g.sort_values("actual_utc")
        t = g["actual_utc"].values.astype("datetime64[ns]").astype("int64")
        cs = np.concatenate([[0.0], np.cumsum(g["clip_delay"].values)])
        cv = np.concatenate([[0.0], np.cumsum(g["severe"].values)])
        sel = np.where(ap_arr == ap)[0]
        hi = np.searchsorted(t, issue_ns[sel], side="right")
        for w in windows:
            lo = np.searchsorted(t, issue_ns[sel] - w * 3600 * 10 ** 9, side="right")
            n = hi - lo
            res[f"rec_{direction}_n_{w}h"][sel] = n
            with np.errstate(invalid="ignore", divide="ignore"):
                res[f"rec_{direction}_mean_{w}h"][sel] = np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)
                res[f"rec_{direction}_severe_{w}h"][sel] = np.where(n > 0, (cv[hi] - cv[lo]) / n, np.nan)
    return pd.DataFrame(res, index=rows.index)


def load_traffic():
    """BTS flights touching the hubs, from the validated cache (stage1_pipeline.download_sources must have run)."""
    records = stage1_pipeline.download_sources(C.EXTERNAL, pd.Timestamp(C.SOURCE_START), pd.Timestamp(C.SOURCE_END),
                                               offline=True)
    return stage1_pipeline.load_bts(records, stage1_pipeline.airport_zones(C.EXTERNAL / "airports.dat"), C.EXTERNAL)


def attach_traffic(rows, bts):
    """Airport-wide BTS context: scheduled movements in the event hour, operations completed in the 3 h before the
    issue time and their severe-departure-delay rate (BTS = reporting carriers, a demand proxy, not AAR)."""
    ev = pd.DataFrame({"airport": rows["airport"].map(lambda icao: C.AIRPORTS[icao]["iata"]),
                       "event_utc": rows["event_utc"], "prediction_utc": rows["issue_utc"]}, index=rows.index)
    out, _ = stage1_pipeline.attach_bts(ev, bts)
    feats = out.drop(columns=[*ev.columns, "bts_observed_through_utc"])
    feats["bts_missing"] = feats["bts_missing"].astype(int)
    return feats


def build_delay_table(hourly, taf_table, flights, leads=AAR_LEADS):
    ev_all = flight_events(flights)
    ev = ev_all
    n_events = len(ev)
    # label: a cancelled flight has no observed delay and a diverted flight has no arrival delay -> dropped
    ev = ev[(ev["cancelled"] == 0) & ev["delay_min"].notna()].copy()
    ev["severe_delay"] = (ev["delay_min"] > C.SEVERE_DELAY_MIN).astype(int)
    ev["event_hour_utc"] = ev["event_utc"].dt.floor("h")
    ev["sched_load_sample"] = ev.groupby(["airport", "direction", "event_hour_utc"])["flight_id"].transform("size")
    ev["to_other_hub"] = ev["other_airport"].isin(C.IATA_TO_ICAO).astype(int)
    cal = calendar_features(ev["sched_local"]).set_index(ev.index)
    ev = pd.concat([ev, cal], axis=1)
    base = ev[["flight_id", "airport", "direction", "other_airport", "OP_UNIQUE_CARRIER", "event_utc", "event_hour_utc",
               "DISTANCE", "CRS_ELAPSED_TIME", "sched_load_sample", "to_other_hub", "delay_min", "severe_delay",
               *cal.columns]].rename(columns={"OP_UNIQUE_CARRIER": "carrier", "DISTANCE": "distance_mi",
                                              "CRS_ELAPSED_TIME": "sched_elapsed_min"})
    hw = hourly_features(hourly)
    bts = load_traffic()
    parts = []
    for lead in leads:
        b = base.copy()
        b["lead_h"] = lead
        b["issue_utc"] = b["event_utc"] - pd.Timedelta(hours=lead)
        b = attach_state(b, hw)
        # recent hub performance, summarising only flights already completed at the issue time
        rec = recent_disruption(events=ev_all, rows=b)
        parts.append(pd.concat([b, rec, attach_traffic(b, bts)], axis=1))
    t = pd.concat(parts, ignore_index=True)
    tt = taf_table.rename(columns={"hour_utc": "event_hour_utc"})
    t = t.merge(tt, on=["airport", "event_hour_utc", "lead_h"], how="left")
    t["split"] = split_of(t["event_utc"])
    t = drop_purged(t)
    t = t.sort_values(["event_utc", "flight_id", "direction", "lead_h"]).reset_index(drop=True)
    print(f"events {n_events:,} -> labelled {t['flight_id'].nunique():,} flights, {len(t):,} rows")
    return t


def drop_purged(t):
    """Drop rows in the purge window and every row of a flight whose departure and arrival land in different splits.
    train_delay trains on split != 'test', so purged rows must not stay in the table."""
    spans = t.groupby("flight_id")["split"].transform("nunique") > 1
    purged = spans | (t["split"] == "purged")
    print(f"purged {int(purged.sum()):,} rows at the test boundary")
    # ponytail: the 2024 out-of-fold quarters in train_delay have no purge gaps; add them if OOF scores look optimistic
    return t[~purged]


def load_flights():
    return pd.read_csv(C.FLIGHTS_CSV, parse_dates=["sched_dep", "sched_arr"])


if __name__ == "__main__":
    h = pd.read_csv(C.PROC / "hourly_weather.csv", parse_dates=["hour_utc", "known_at_utc"])
    taf_all = tafmod.build_taf_table(h, AAR_LEADS)
    taf_all.to_parquet(C.PROC / "taf_features.parquet", index=False)
    aar_t = build_aar_table(h, taf_all)
    aar_t.to_parquet(C.PROC / "aar_table.parquet", index=False)
    print("AAR table", aar_t.shape, aar_t["split"].value_counts().to_dict())
    delay_t = build_delay_table(h, taf_all, load_flights())
    delay_t.to_parquet(C.PROC / "delay_table.parquet", index=False)
    print("delay table", delay_t.shape, delay_t["split"].value_counts().to_dict())
