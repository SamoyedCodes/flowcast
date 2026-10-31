"""
Download archived METARs (routine + specials) from the Iowa Environmental Mesonet (IEM).

Endpoint: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
Documented params used: station, data, sts, ets (ISO, Z), tz=Etc/UTC, format=onlycomma,
missing=empty, trace=0.0001, report_type=3 (routine) & 4 (specials).
IEM throttles to 1 request/second/IP and caps 1,000 station-years per request.

Usage: python pull_metar.py --start 2023-01 --end 2025-07 --airports ATL DFW
"""
import argparse
import pandas as pd

from common import PROC, RAW, ROOT, get, load_airports, record, to_utc_naive

# teammate notebook (weather_download.ipynb) caches the same IEM endpoint with data=all, missing=M
NOTEBOOK_CACHE = ROOT.parents[1] / "data" / "weather" / "raw"   # repo-root data/ (pipeline lives in jho/)

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby", "alti", "mslp", "p01i",
          "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes",
          "peak_wind_gust", "peak_wind_drct", "metar"]


def fetch_year(station, year, start, end):
    sts = max(pd.Timestamp(f"{year}-01-01"), start)
    ets = min(pd.Timestamp(f"{year + 1}-01-01"), end)
    dest = RAW / "metar" / f"{station}_{sts:%Y%m%d}_{ets:%Y%m%d}.csv"
    nb = NOTEBOOK_CACHE / f"asos_{station}_{sts:%Y-%m-%d}_{ets:%Y-%m-%d}.csv"
    if not dest.exists() and nb.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(nb.read_bytes())
        record(f"cache:{nb.name} (IEM asos.py, data=all, report_type=3,4)", dest)
        print(f"{station} {year}: reused notebook cache {nb.name}")
    if not dest.exists():
        params = [("station", station), *[("data", f) for f in FIELDS],
                  ("sts", f"{sts:%Y-%m-%dT%H:%M}Z"), ("ets", f"{ets:%Y-%m-%dT%H:%M}Z"),
                  ("tz", "Etc/UTC"), ("format", "onlycomma"), ("missing", "empty"),
                  ("trace", "0.0001"), ("report_type", "3"), ("report_type", "4")]
        get(URL, params=params, dest=dest)
    df = pd.read_csv(dest, low_memory=False, na_values=["M"], comment="#")
    if "valid" not in df.columns or df.empty:
        raise SystemExit(f"IEM returned no data for {station} {year}: {dest.read_text()[:300]}")
    return df


def main(start, end, airports):
    start = pd.Timestamp(start + "-01")
    end = (pd.Timestamp(end + "-01") + pd.offsets.MonthBegin(1))
    frames = []
    for code in load_airports(airports):
        for y in range(start.year, end.year + 1):
            if pd.Timestamp(f"{y}-01-01") >= end:
                break
            d = fetch_year(code, y, start, end)
            d["station"] = code
            frames.append(d)
            print(f"{code} {y}: {len(d):,} METARs")
    m = pd.concat(frames, ignore_index=True)
    m["valid"] = to_utc_naive(m["valid"])  # UTC (tz=Etc/UTC requested)
    m = m[[c for c in ["station", "valid"] + FIELDS if c in m.columns]]
    m = m.drop_duplicates(["station", "valid", "metar"]).sort_values(["station", "valid"])
    PROC.mkdir(parents=True, exist_ok=True)
    m.to_parquet(PROC / "metar.parquet", index=False)
    print(f"wrote {len(m):,} METARs -> data/processed/metar.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2025-07")
    ap.add_argument("--airports", nargs="+", default=["ATL", "DFW"])
    a = ap.parse_args()
    main(a.start, a.end, a.airports)
