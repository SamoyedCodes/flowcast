"""
Download BTS 'Reporting Carrier On-Time Performance (1987-present)' monthly files and keep
flights departing from or arriving at the configured airports.

Source: https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YYYY}_{M}.zip
(month is NOT zero-padded). This is the same table the provided CSV was drawn from
(identical field set; the provided file uses the TranStats upper-case field names).

Usage: python pull_bts.py --start 2023-01 --end 2025-07 --airports ATL DFW
"""
import argparse
import io
import zipfile

import pandas as pd

from common import PROC, RAW, get, load_airports, months

URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"

# PREZIP camel-case -> provided-file upper-case names (only the fields we use)
KEEP = {
    "FlightDate": "FL_DATE", "Reporting_Airline": "OP_UNIQUE_CARRIER", "Tail_Number": "TAIL_NUM",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM", "Origin": "ORIGIN", "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME", "DepTime": "DEP_TIME", "DepDelay": "DEP_DELAY",
    "TaxiOut": "TAXI_OUT", "TaxiIn": "TAXI_IN",
    "CRSArrTime": "CRS_ARR_TIME", "ArrTime": "ARR_TIME", "ArrDelay": "ARR_DELAY",
    "Cancelled": "CANCELLED", "CancellationCode": "CANCELLATION_CODE", "Diverted": "DIVERTED",
    "CRSElapsedTime": "CRS_ELAPSED_TIME", "ActualElapsedTime": "ACTUAL_ELAPSED_TIME",
    "Distance": "DISTANCE", "CarrierDelay": "CARRIER_DELAY", "WeatherDelay": "WEATHER_DELAY",
    "NASDelay": "NAS_DELAY", "SecurityDelay": "SECURITY_DELAY", "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",
}


def read_zip(path_or_bytes):
    zf = zipfile.ZipFile(path_or_bytes if not isinstance(path_or_bytes, bytes) else io.BytesIO(path_or_bytes))
    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise ValueError(f"expected one CSV in zip, found {csvs}")
    with zf.open(csvs[0]) as f:
        df = pd.read_csv(f, usecols=lambda c: c in KEEP, low_memory=False, dtype={"CancellationCode": str})
    missing = set(KEEP) - set(df.columns)
    if missing:
        raise ValueError(f"BTS schema changed, missing columns: {sorted(missing)}")
    return df.rename(columns=KEEP)


def main(start, end, airports):
    aps = set(load_airports(airports))
    out = []
    for y, m in months(start, end):
        z = RAW / "bts" / f"ontime_{y}_{m:02d}.zip"
        if not z.exists():
            get(URL.format(y=y, m=m), dest=z, stream=True, pause=0.5, timeout=600)
        df = read_zip(z)
        n_all = len(df)
        df = df[df.ORIGIN.isin(aps) | df.DEST.isin(aps)]
        print(f"{y}-{m:02d}: {n_all:,} US flights -> {len(df):,} touching {sorted(aps)}")
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    PROC.mkdir(parents=True, exist_ok=True)
    res.to_parquet(PROC / "bts_flights.parquet", index=False)
    print(f"wrote {len(res):,} flights -> data/processed/bts_flights.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2025-07")
    ap.add_argument("--airports", nargs="+", default=["ATL", "DFW"])
    a = ap.parse_args()
    main(a.start, a.end, a.airports)
