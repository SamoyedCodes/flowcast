"""
Clean the provided Flight_Data_With_Weather_Final.csv and add derived, documented columns.

Usage:
    python clean_provided_file.py <input_csv> <output_dir>

What it does (every step is logged to audit_log.csv):
  1. Builds scheduled departure/arrival timestamps in local time and UTC
     (airport time zones from the `airportsdata` package, arrival = departure + CRS_ELAPSED_TIME).
  2. Re-times the provided weather. Verified against NOAA ISD: the weather on each row is the
     observation for hour H in UTC, but it was joined to hour H in LOCAL time, and destination
     weather uses the DEPARTURE date (FL_DATE). We do not change the values; we record their
     true observation time and how many hours before the scheduled event they were observed.
  3. Converts units (Meteostat wspd/wpgt are km/h -> knots), marks invalid values as missing
     (coco 0 is not a valid code; snow is 0 in every row so it is dropped; wpgt 0 means
     "no gust reported", kept as a flag).
  4. Decodes Meteostat condition codes into hazard flags.
  5. Builds targets for severe delay (>45 min) with cancellations/diversions handled explicitly.
  6. Adds ATL/DFW scope flags and the challenge's assumed runway headings.
  7. Drops columns that are empty in every row.
It never invents values. Crosswind/headwind are NOT computed: the file has no wind direction.
"""
import sys
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

RUNWAY_HEADING = {"ATL": 80, "DFW": 180}  # challenge assumption (PDF p.6)
FOCUS = list(RUNWAY_HEADING)
KMH_PER_KT = 1.852
# airportsdata (IATA index) lacks PBI; Palm Beach International is in Florida, US Eastern time.
TZ_OVERRIDE = {"PBI": "America/New_York"}

COCO = {1: "Clear", 2: "Fair", 3: "Cloudy", 4: "Overcast", 5: "Fog", 6: "Freezing Fog",
        7: "Light Rain", 8: "Rain", 9: "Heavy Rain", 10: "Freezing Rain",
        11: "Heavy Freezing Rain", 12: "Sleet", 13: "Heavy Sleet", 14: "Light Snowfall",
        15: "Snowfall", 16: "Heavy Snowfall", 17: "Rain Shower", 18: "Heavy Rain Shower",
        19: "Sleet Shower", 20: "Heavy Sleet Shower", 21: "Snow Shower",
        22: "Heavy Snow Shower", 23: "Lightning", 24: "Hail", 25: "Thunderstorm",
        26: "Heavy Thunderstorm", 27: "Storm"}
FLAGS = {
    "thunder": {23, 24, 25, 26, 27},
    "freezing": {6, 10, 11, 12, 13, 19, 20},
    "snow": {14, 15, 16, 21, 22},
    "fog": {5, 6},
    "heavy_precip": {9, 11, 13, 16, 18, 20, 22, 26},
    "any_precip": set(range(7, 23)) | {24, 25, 26, 27},
}

log = []


def note(step, detail, rows=None):
    log.append({"step": step, "detail": detail, "rows_affected": rows})
    print(f"[{step}] {detail}" + (f" ({rows})" if rows is not None else ""))


def hhmm_to_td(x):
    x = x.astype("Int64")
    return pd.to_timedelta((x // 100) * 60 + (x % 100), unit="m")


def main(inp, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp, low_memory=False)
    n0 = len(df)
    note("load", f"{n0} rows x {df.shape[1]} cols from {Path(inp).name}")

    # ---------- 0. drop all-empty columns ----------
    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    note("drop_empty_cols", "dropped columns empty in every row: " + ", ".join(empty), len(empty))
    for side in ("ORIGIN", "DEST"):
        if (df[f"snow_{side}"].fillna(0) == 0).all():
            df = df.drop(columns=[f"snow_{side}"])
            note("drop_snow", f"snow_{side} is 0 in every row (incl. DEN/ORD winters) -> treated as not populated, dropped")

    # ---------- 1. timestamps ----------
    ap = airportsdata.load("IATA")
    tzmap = {k: v["tz"] for k, v in ap.items()}
    tzmap.update(TZ_OVERRIDE)
    airports = set(df.ORIGIN) | set(df.DEST)
    missing_tz = sorted(a for a in airports if a not in tzmap)
    note("tz_lookup", f"{len(airports)} airports, missing tz: {missing_tz or 'none'}")

    date = pd.to_datetime(df.FL_DATE)
    dep_local_naive = date + hhmm_to_td(df.CRS_DEP_TIME)
    dep_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for a, idx in df.groupby("ORIGIN").groups.items():
        dep_utc.loc[idx] = (dep_local_naive.loc[idx]
                            .dt.tz_localize(tzmap[a], ambiguous="NaT", nonexistent="shift_forward")
                            .dt.tz_convert("UTC"))
    arr_utc = dep_utc + pd.to_timedelta(df.CRS_ELAPSED_TIME, unit="m")
    arr_local = pd.Series(pd.NaT, index=df.index, dtype="object")
    for a, idx in df.groupby("DEST").groups.items():
        arr_local.loc[idx] = arr_utc.loc[idx].dt.tz_convert(tzmap[a]).dt.tz_localize(None)
    arr_local = pd.to_datetime(arr_local)
    df["sched_dep_local"] = dep_local_naive
    df["sched_dep_utc"] = dep_utc.dt.tz_localize(None)
    df["sched_arr_utc"] = arr_utc.dt.tz_localize(None)
    df["sched_arr_local"] = arr_local
    # consistency: rebuilt arrival clock time vs CRS_ARR_TIME
    rebuilt = arr_local.dt.hour * 100 + arr_local.dt.minute
    ok = rebuilt == df.CRS_ARR_TIME
    note("check_arrival_time", f"sched_arr_local reproduces CRS_ARR_TIME in {ok.mean():.4%} of rows "
         "(mismatches are BTS rounding/DST edge cases, kept and flagged)", int((~ok).sum()))
    df["flag_sched_time_mismatch"] = (~ok).astype(int)
    df["sched_arr_next_day"] = (arr_local.dt.normalize() > date).astype(int)

    # ---------- 2. re-time weather ----------
    # verified: value = METAR for hour H UTC on FL_DATE, where H = local scheduled hour label.
    dep_h = (df.CRS_DEP_TIME // 100) % 24
    arr_h = (df.CRS_ARR_TIME // 100) % 24
    df["wx_ORIGIN_obs_utc"] = date + pd.to_timedelta(dep_h, unit="h")
    df["wx_DEST_obs_utc"] = date + pd.to_timedelta(arr_h, unit="h")
    df["wx_ORIGIN_hours_before_sched_dep"] = ((df.sched_dep_utc - df.wx_ORIGIN_obs_utc)
                                              .dt.total_seconds() / 3600).round(2)
    df["wx_DEST_hours_before_sched_arr"] = ((df.sched_arr_utc - df.wx_DEST_obs_utc)
                                            .dt.total_seconds() / 3600).round(2)
    note("retime_weather", "added true UTC observation time of provided weather and lead vs scheduled time; "
         f"median lead origin {df.wx_ORIGIN_hours_before_sched_dep.median()}h, dest "
         f"{df.wx_DEST_hours_before_sched_arr.median()}h")
    stale = df.wx_DEST_hours_before_sched_arr > 12
    df["flag_dest_wx_stale_gt12h"] = stale.astype(int)
    note("flag_stale_dest_wx", "dest weather observed >12h before scheduled arrival (overnight arrivals joined on departure date)",
         int(stale.sum()))

    # ---------- 3. units & invalid values ----------
    for side in ("ORIGIN", "DEST"):
        df[f"wspd_kt_{side}"] = (df[f"wspd_{side}"] / KMH_PER_KT).round(1)
        g = df[f"wpgt_{side}"]
        df[f"gust_reported_{side}"] = (g > 0).astype(int)
        df[f"wpgt_kt_{side}"] = (g.where(g > 0) / KMH_PER_KT).round(1)
        df[f"gust_factor_kt_{side}"] = (df[f"wpgt_kt_{side}"] - df[f"wspd_kt_{side}"]).round(1)
        bad = ~df[f"coco_{side}"].isin(COCO.keys())
        note("coco_invalid", f"coco_{side} not in 1..27 set to missing", int(bad.sum()))
        df.loc[bad, f"coco_{side}"] = np.nan
        df[f"coco_label_{side}"] = df[f"coco_{side}"].map(COCO)
        for name, codes in FLAGS.items():
            df[f"wx_{name}_{side}"] = df[f"coco_{side}"].isin(codes).astype(int)
        # gusts below mean wind are physically inconsistent
        inc = df[f"gust_factor_kt_{side}"] < 0
        note("gust_lt_wind", f"{side}: gust < mean wind (kept, flagged)", int(inc.sum()))
        df[f"flag_gust_lt_wind_{side}"] = inc.astype(int)
    note("units", "Meteostat units: temp degC, prcp mm/h, wspd & wpgt km/h (converted to kt), pres hPa sea-level")

    # ---------- 4. targets ----------
    canc = df.CANCELLED == 1
    div = df.DIVERTED == 1
    df["y_arr_delay_gt45"] = np.where(canc | div, np.nan, (df.ARR_DELAY > 45).astype(float))
    df["y_dep_delay_gt45"] = np.where(canc, np.nan, (df.DEP_DELAY > 45).astype(float))
    df["y_arr_severe_incl_cxl_div"] = ((df.ARR_DELAY > 45) | canc | div).astype(int)
    df["y_dep_severe_incl_cxl"] = ((df.DEP_DELAY > 45) | canc).astype(int)
    note("targets", "y_*_gt45 exclude cancelled/diverted (NaN); y_*_severe_incl_* count them as severe")

    # ---------- 5. scope ----------
    df["focus_arrival"] = df.DEST.isin(FOCUS).astype(int)
    df["focus_departure"] = df.ORIGIN.isin(FOCUS).astype(int)
    df["in_scope_ATL_DFW"] = ((df.focus_arrival == 1) | (df.focus_departure == 1)).astype(int)
    df["focus_airport"] = np.select(
        [df.DEST.isin(FOCUS) & ~df.ORIGIN.isin(FOCUS), df.ORIGIN.isin(FOCUS) & ~df.DEST.isin(FOCUS),
         df.ORIGIN.isin(FOCUS) & df.DEST.isin(FOCUS)],
        [df.DEST, df.ORIGIN, df.ORIGIN + "-" + df.DEST], default="")
    df["runway_heading_ORIGIN"] = df.ORIGIN.map(RUNWAY_HEADING)
    df["runway_heading_DEST"] = df.DEST.map(RUNWAY_HEADING)
    note("scope", f"rows touching ATL/DFW: {int(df.in_scope_ATL_DFW.sum())}; both ends ATL/DFW: "
         f"{int((df.focus_arrival & df.focus_departure).sum())}")

    # ---------- 6. calendar ----------
    df["sched_dep_hour_local"] = dep_local_naive.dt.hour
    df["sched_arr_hour_local"] = arr_local.dt.hour
    df["is_weekend"] = (df.DAY_OF_WEEK >= 6).astype(int)

    assert len(df) == n0, "row count changed"
    df.to_csv(outdir / "Flight_Data_With_Weather_Cleaned.csv", index=False)
    pd.DataFrame(log).to_csv(outdir / "audit_log.csv", index=False)
    note("write", f"wrote {len(df)} rows x {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
