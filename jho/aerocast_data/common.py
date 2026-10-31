"""Shared helpers: airport registry, polite HTTP with retries, provenance manifest."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AEROCAST_DATA", ROOT / "data")).resolve()  # override for smoke tests
RAW = DATA / "raw"
PROC = DATA / "processed"
MANIFEST = DATA / "manifest.jsonl"
UA = {"User-Agent": "AeroCast-student-hackathon/1.0 (NUS Autonomous SG 2026)"}


def load_airports(codes=None):
    cfg = json.loads((ROOT / "airports.json").read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if codes:
        out = {}
        for c in codes:
            c = c.upper()
            c = c[1:] if len(c) == 4 and c.startswith("K") else c  # accept ICAO KXXX
            if c in cfg:
                out[c] = cfg[c]
            else:
                import airportsdata
                ap = airportsdata.load("IATA").get(c)
                if ap is None:
                    raise SystemExit(f"Unknown airport {c}; add it to airports.json")
                out[c] = {"icao": ap["icao"], "tz": ap["tz"], "lat": ap["lat"], "lon": ap["lon"],
                          "runway_heading_deg": None, "storm_state": None}
                print(f"[warn] {c} not in airports.json: no runway heading, wind components will be NaN")
        return out
    return cfg


def get(url, params=None, dest=None, retries=5, pause=1.1, timeout=180, stream=False):
    """GET with retry/backoff. IEM asks for <=1 request/second per IP."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout, stream=stream)
            if r.status_code in (429, 503):
                raise requests.HTTPError(f"{r.status_code} throttled")
            r.raise_for_status()
            time.sleep(pause)
            if dest:
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
                record(url if params is None else r.url, dest)
                return dest
            return r
        except Exception as e:  # noqa: BLE001
            wait = 2 ** i * 2
            print(f"[retry {i + 1}/{retries}] {url} -> {e}; sleeping {wait}s")
            time.sleep(wait)
    raise SystemExit(f"FAILED after {retries} tries: {url}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def record(url, path):
    """Append provenance (source URL, retrieval time, size, hash) for traceability."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    row = {"url": url, "file": str(Path(path).resolve().relative_to(DATA)), "bytes": Path(path).stat().st_size,
           "sha256": sha256(path), "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(row) + "\n")


def months(start, end):
    """Inclusive list of (year, month) from 'YYYY-MM' to 'YYYY-MM'."""
    y, m = map(int, start.split("-"))
    y2, m2 = map(int, end.split("-"))
    out = []
    while (y, m) <= (y2, m2):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def to_utc_naive(s):
    """Parse timestamps as UTC and drop the tz, whether the source wrote '...Z' or a naive string."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)
