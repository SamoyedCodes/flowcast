"""
Download NOAA NCEI Storm Events 'details' files and keep events near each airport.

Directory: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
Files: StormEvents_details-ftp_v1.0_d{YEAR}_c{CREATED}.csv.gz  (newest c-date is picked automatically)
Times: BEGIN/END are local STANDARD time of CZ_TIMEZONE (e.g. 'EST-5', 'CST-6'); converted to UTC here.

IMPORTANT (leakage): Storm Events are compiled AFTER the event. Use them only to label or
stratify evaluation (e.g. 'hours with a severe convective event within 50 km'), never as a model
input at prediction time.

Usage: python pull_storm_events.py --years 2023 2024 2025 --airports ATL DFW --radius-km 50
"""
import argparse
import re

import numpy as np
import pandas as pd

from common import PROC, RAW, get, load_airports

BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def latest_name(listing_html, year):
    names = re.findall(rf"StormEvents_details-ftp_v1\.0_d{year}_c(\d{{8}})\.csv\.gz", listing_html)
    if not names:
        raise SystemExit(f"No Storm Events details file for {year}")
    return f"StormEvents_details-ftp_v1.0_d{year}_c{max(names)}.csv.gz"


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def to_utc(df):
    """BEGIN_YEARMONTH (YYYYMM) + BEGIN_DAY + BEGIN_TIME (hhmm) in CZ_TIMEZONE standard time -> UTC."""
    out = {}
    for p in ("BEGIN", "END"):
        ym = df[f"{p}_YEARMONTH"].astype(int)
        t = df[f"{p}_TIME"].astype(int)
        local = pd.to_datetime(dict(year=ym // 100, month=ym % 100, day=df[f"{p}_DAY"].astype(int),
                                    hour=t // 100, minute=t % 100))
        off = df["CZ_TIMEZONE"].astype(str).str.extract(r"(-?\d+)\s*$")[0].astype(float)
        out[f"{p.lower()}_utc"] = local - pd.to_timedelta(off, unit="h")  # EST-5 -> local + 5h
    return pd.DataFrame(out, index=df.index), df["CZ_TIMEZONE"].astype(str).str.extract(r"(-?\d+)\s*$")[0].isna()


def main(years, airports, radius_km):
    listing = get(BASE).text
    aps = load_airports(airports)
    keep = []
    for y in years:
        name = latest_name(listing, y)
        dest = RAW / "storm_events" / name
        if not dest.exists():
            get(BASE + name, dest=dest, stream=True)
        se = pd.read_csv(dest, compression="gzip", low_memory=False)
        times, bad_tz = to_utc(se)
        se = pd.concat([se, times], axis=1)
        se["flag_tz_unparsed"] = bad_tz.astype(int)
        for code, a in aps.items():
            d = haversine_km(a["lat"], a["lon"], se.BEGIN_LAT, se.BEGIN_LON)
            near = d <= radius_km
            # zone-based events (no lat/lon, e.g. Heavy Snow, Winter Storm) -> keep if same state
            zone = se.BEGIN_LAT.isna() & (se.STATE == a.get("storm_state"))
            sel = se[near | zone].copy()
            sel["airport"] = code
            sel["dist_km"] = d[near | zone]
            sel["match"] = np.where(near[near | zone], "within_radius", "same_state_zone_event")
            keep.append(sel)
        print(f"{y}: {name} ({len(se):,} events)")
    out = pd.concat(keep, ignore_index=True)
    cols = ["airport", "match", "dist_km", "EVENT_ID", "EPISODE_ID", "EVENT_TYPE", "STATE", "CZ_NAME",
            "CZ_TYPE", "CZ_TIMEZONE", "begin_utc", "end_utc", "MAGNITUDE", "MAGNITUDE_TYPE",
            "BEGIN_LAT", "BEGIN_LON", "flag_tz_unparsed"]
    PROC.mkdir(parents=True, exist_ok=True)
    out[cols].to_parquet(PROC / "storm_events.parquet", index=False)
    print(out.groupby(["airport", "match"]).size())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--airports", nargs="+", default=["ATL", "DFW"])
    ap.add_argument("--radius-km", type=float, default=50)
    a = ap.parse_args()
    main(a.years, a.airports, a.radius_km)
