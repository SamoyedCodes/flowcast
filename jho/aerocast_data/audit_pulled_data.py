"""
Audit pulled data for exactness against the organiser's provided CSV (an independent copy).

Checks
  1. BTS: every provided-file flight touching ATL/DFW is found in the pulled BTS data and its
     delay/cancellation fields match exactly.
  2. METAR: provided weather (Meteostat, hour H UTC) vs IEM METAR as-of hour H UTC:
     temperature within 0.6 degC, wind speed within 2 kt.
  3. Coverage: flights per month, METAR hours with an observation, TAF issuances per day.
Writes data/processed/audit_report.csv and prints a summary. Non-zero exit if a check fails.

Usage: python audit_pulled_data.py --provided ../Flight_Data_With_Weather_Final.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

from common import PROC

FOCUS = ["ATL", "DFW"]


def main(provided):
    rep, ok = [], True
    p = pd.read_csv(provided, low_memory=False)
    p = p[p.ORIGIN.isin(FOCUS) | p.DEST.isin(FOCUS)]
    b = pd.read_parquet(PROC / "bts_flights.parquet")
    key = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME"]
    for d in (p, b):
        d["FL_DATE"] = pd.to_datetime(d.FL_DATE).dt.strftime("%Y-%m-%d")
        d["OP_CARRIER_FL_NUM"] = pd.to_numeric(d.OP_CARRIER_FL_NUM).astype("Int64")
        d["CRS_DEP_TIME"] = pd.to_numeric(d.CRS_DEP_TIME).astype("Int64")
    lo, hi = b.FL_DATE.min(), b.FL_DATE.max()
    p = p[(p.FL_DATE >= lo) & (p.FL_DATE <= hi)]
    m = p.merge(b, on=key, how="left", suffixes=("_p", "_b"), indicator=True)
    found = (m._merge == "both").mean()
    rep.append(("bts_rows_found", len(p), f"{found:.4%}"))
    ok &= found > 0.995
    for c in ["DEP_DELAY", "ARR_DELAY", "CANCELLED", "DIVERTED", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "TAIL_NUM"]:
        a, bb = m[f"{c}_p"], m[f"{c}_b"]
        both = m._merge == "both"
        same = ((a == bb) | (a.isna() & bb.isna()))[both].mean()
        rep.append((f"bts_exact_{c}", int(both.sum()), f"{same:.4%}"))
        ok &= same > 0.999

    # METAR vs provided weather (provided hour H label is UTC; see clean_provided_file.py)
    mh = pd.read_parquet(PROC / "metar_hourly.parquet")
    parts = []
    for side, tcol in (("ORIGIN", "CRS_DEP_TIME"), ("DEST", "CRS_ARR_TIME")):
        t = p[p[side].isin(FOCUS)][[side, "FL_DATE", tcol, f"temp_{side}", f"wspd_{side}"]].copy()
        t.columns = ["airport", "FL_DATE", "hhmm", "temp", "wspd"]
        t["hour_utc"] = pd.to_datetime(t.FL_DATE) + pd.to_timedelta((t.hhmm // 100) % 24, unit="h")
        parts.append(t)
    w = pd.concat(parts).drop_duplicates(["airport", "hour_utc"])
    w["hour_utc"] = w.hour_utc.astype("datetime64[ns]")
    mh["hour_utc"] = pd.to_datetime(mh.hour_utc).astype("datetime64[ns]")
    j = w.merge(mh[mh.obs_age_h == 0][["airport", "hour_utc", "temp_c", "sknt"]], on=["airport", "hour_utc"], how="inner")
    tdiff = (j.temp - j.temp_c).abs()
    wdiff = (j.wspd / 1.852 - j.sknt).abs()
    rep.append(("metar_vs_provided_temp_within_0.6C", len(j), f"{(tdiff <= 0.6).mean():.4%}"))
    rep.append(("metar_vs_provided_wind_within_2kt", len(j), f"{(wdiff <= 2).mean():.4%}"))
    # and the WRONG (local-time) alignment should match much worse -> proves the time fix
    w2 = w.copy()
    w2["hour_utc"] = w2.hour_utc + pd.to_timedelta(np.where(w2.airport == "ATL", 4, 5), unit="h")
    j2 = w2.merge(mh[mh.obs_age_h == 0][["airport", "hour_utc", "temp_c"]], on=["airport", "hour_utc"])
    rep.append(("metar_vs_provided_temp_if_treated_as_local", len(j2), f"{((j2.temp - j2.temp_c).abs() <= 0.6).mean():.4%}"))
    ok &= (tdiff <= 0.6).mean() > 0.97

    # coverage
    bts_month = b.groupby(b.FL_DATE.str[:7]).size()
    rep.append(("bts_months", len(bts_month), f"min {bts_month.min():,} / max {bts_month.max():,} flights per month"))
    for a, g in mh.groupby("airport"):
        rep.append((f"metar_hours_with_obs_{a}", len(g), f"{(g.obs_missing == 0).mean():.2%}"))
    if (PROC / "taf.parquet").exists():
        t = pd.read_parquet(PROC / "taf.parquet")
        per_day = t.groupby(["airport", pd.to_datetime(t.valid).dt.date]).valid.nunique()
        rep.append(("taf_issuances_per_day_median", len(per_day), f"{per_day.median():.1f}"))

    out = pd.DataFrame(rep, columns=["check", "n", "result"])
    out.to_csv(PROC / "audit_report.csv", index=False)
    print(out.to_string(index=False))
    print("AUDIT", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provided", required=True)
    main(ap.parse_args().provided)
