"""
Download archived Terminal Aerodrome Forecasts (TAFs) from IEM, already decoded per forecast group.

Endpoint: https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py
Params: station, sts, ets (filter on ISSUANCE time), fmt=csv, tz=UTC (default).
Output columns: station, valid (issuance), fx_valid, raw, is_tempo, fx_valid_end, sknt, drct,
gust, visibility, presentwx, skyc, skyl, ws_level, ws_drct, ws_sknt, product_id, ftype, is_amendment.

TAFs are the legitimate source of *forecast* weather for 3-12 h horizons: a TAF issued at or
before the prediction time t0 is information a duty manager really had at t0.

Usage: python pull_taf.py --start 2023-01 --end 2025-07 --airports ATL DFW
"""
import argparse

import pandas as pd

from common import PROC, RAW, get, load_airports, months, to_utc_naive

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"


def main(start, end, airports):
    frames = []
    for code in load_airports(airports):
        for y, m in months(start, end):
            sts = pd.Timestamp(year=y, month=m, day=1)
            ets = sts + pd.offsets.MonthBegin(1)
            dest = RAW / "taf" / f"{code}_{y}_{m:02d}.csv"
            if not dest.exists():
                get(URL, params={"station": code, "sts": f"{sts:%Y-%m-%dT%H:%M}Z",
                                 "ets": f"{ets:%Y-%m-%dT%H:%M}Z", "fmt": "csv"}, dest=dest)
            d = pd.read_csv(dest)
            if d.empty:
                print(f"[warn] no TAFs for {code} {y}-{m:02d}")
                continue
            d["airport"] = code
            frames.append(d)
        print(f"{code}: done")
    t = pd.concat(frames, ignore_index=True)
    for c in ("valid", "fx_valid", "fx_valid_end"):
        t[c] = to_utc_naive(t[c])
    PROC.mkdir(parents=True, exist_ok=True)
    t.to_parquet(PROC / "taf.parquet", index=False)
    print(f"wrote {len(t):,} TAF forecast groups ({t.groupby(['airport','valid']).ngroups:,} issuances)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2025-07")
    ap.add_argument("--airports", nargs="+", default=["ATL", "DFW"])
    a = ap.parse_args()
    main(a.start, a.end, a.airports)
