"""Download METAR observations from the Iowa State Mesonet archive (no manual steps, cached on disk).

  * ASOS/METAR observations  -> data/weather/raw/asos_<IATA>_<start>_<end>.csv

TAF forecasts with their validity metadata, BTS traffic and airport time zones are fetched by
stage1_pipeline.download_sources into data/external/ (validated, hash receipts, offline replay).

Re-running only fetches chunks that are missing.   python -m src.download
"""
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from . import config as C

UA = {"User-Agent": "acacia-hackathon-challenge5 (student project)"}
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def chunks(start, end, freq):
    """Split [start, end) into pieces: 'Y' = calendar years, 'Q' = calendar quarters."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    step = {"Y": pd.DateOffset(years=1), "Q": pd.DateOffset(months=3)}[freq]
    edges = [s]
    cur = pd.Timestamp(year=s.year, month=1 if freq == "Y" else ((s.month - 1) // 3) * 3 + 1, day=1)
    while cur < e:
        cur = cur + step
        if s < cur < e:
            edges.append(cur)
    edges.append(e)
    return [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(edges[:-1], edges[1:])]


def _get(url, params, expect_header, retries=4):
    full = url + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(full, headers=UA)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if not text.lstrip().startswith(expect_header + ("#",)):
                raise ValueError("unexpected response: " + text[:200])
            return text
        except (urllib.error.URLError, TimeoutError, ValueError, http.client.HTTPException, ConnectionError) as err:
            print(f"  attempt {attempt} failed: {err}")
            if attempt == retries:
                raise
            time.sleep(10 * attempt)


def download_asos(icaos=None):
    C.RAW_WX.mkdir(parents=True, exist_ok=True)
    out = []
    for icao in icaos or C.AIRPORTS:
        iata = C.AIRPORTS[icao]["iata"]
        for sts, ets in chunks(C.START, C.END, "Y"):
            path = C.RAW_WX / f"asos_{iata}_{sts}_{ets}.csv"
            if not path.exists():
                print(f"downloading METAR {iata} {sts} -> {ets}")
                params = [("station", iata), ("data", "all"), ("sts", f"{sts}T00:00:00Z"), ("ets", f"{ets}T00:00:00Z"),
                          ("tz", "UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
                          ("missing", "M"), ("trace", "0.0001"), ("report_type", "3"), ("report_type", "4")]
                path.write_text(_get(ASOS_URL, params, ("station,valid",)), encoding="utf-8")
                time.sleep(2)
            out.append(path)
    return out


if __name__ == "__main__":
    download_asos()
    print("done")
