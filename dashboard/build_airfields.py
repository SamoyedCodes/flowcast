"""Airfield geometry for the dashboard stage: runways, taxiways, aprons and terminals of every airport in
src/config.AIRPORTS, from OpenStreetMap (© OpenStreetMap contributors, ODbL) via the Overpass API.
Writes dashboard/web/airfields.js. Coordinates are metres from the centre of the runways, x east, y north.
Run once (and again after adding an airport):  python dashboard/build_airfields.py
"""
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "web" / "airfields.js"
OVERPASS = "https://overpass-api.de/api/interpreter"
QUERY = """[out:json][timeout:120];
wr[aeroway=aerodrome][icao={icao}]->.ap;.ap map_to_area->.a;
(way(area.a)[aeroway~"^(runway|taxiway)$"];wr(area.a)[aeroway~"^(terminal|apron)$"];wr(area.a)[building=terminal];);
out geom;"""


def overpass(icao):
    req = urllib.request.Request(OVERPASS, urllib.parse.urlencode({"data": QUERY.format(icao=icao)}).encode(),
                                 headers={"User-Agent": "LARP-dashboard/1.0 (airfield geometry)"})
    with urllib.request.urlopen(req, timeout=150) as r:
        return json.load(r)["elements"]


def simplify(pts, tol):
    """Ramer-Douglas-Peucker on [(x, y), ...]."""
    if len(pts) < 3:
        return pts
    (ax, ay), (bx, by) = pts[0], pts[-1]
    L = math.hypot(bx - ax, by - ay)
    d = [abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) / L if L > 1e-6 else math.hypot(x - ax, y - ay) for x, y in pts[1:-1]]
    i = max(range(len(d)), key=d.__getitem__)
    if d[i] <= tol:
        return [pts[0], pts[-1]]
    return simplify(pts[:i + 2], tol)[:-1] + simplify(pts[i + 1:], tol)


def stitch(lines):
    """Join open ways that share end nodes into rings (multipolygon outers come split)."""
    rings, open_ = [g for g in lines if g[0] == g[-1]], [g for g in lines if g[0] != g[-1]]
    while open_:
        r = open_.pop()
        while r[0] != r[-1]:
            nxt = next((g for g in open_ if r[-1] in (g[0], g[-1])), None)
            if nxt is None:
                break
            open_.remove(nxt)
            r = r + (nxt if nxt[0] == r[-1] else nxt[::-1])[1:]
        rings.append(r)
    return rings


def width_m(tag):
    """OSM width tag ("45.72", "150 ft", "61 m") in metres; None if unreadable."""
    m = re.match(r"\s*([\d.]+)\s*(ft|')?", tag)
    return m and float(m.group(1)) * (0.3048 if m.group(2) else 1)


def field(elements):
    ll = lambda g: [(p["lon"], p["lat"]) for p in g]  # noqa: E731
    runways, widths = {}, {}
    for e in elements:
        t = e.get("tags", {})
        if t.get("aeroway") == "runway" and t.get("ref"):
            runways.setdefault(t["ref"], []).extend(ll(e["geometry"]))
            if t.get("width"):
                widths.setdefault(t["ref"], width_m(t["width"]))
    lons = [p[0] for v in runways.values() for p in v]
    lats = [p[1] for v in runways.values() for p in v]
    lon0, lat0 = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540   # equirectangular, fine over a few km
    xy = lambda pts: [((lo - lon0) * kx, (la - lat0) * ky) for lo, la in pts]  # noqa: E731
    flat = lambda pts: [round(v) for p in pts for v in p]  # noqa: E731

    rw = []
    for ref, pts in runways.items():
        pts = xy(pts)
        a = max(pts, key=lambda p: math.dist(p, pts[0]))
        b = max(pts, key=lambda p: math.dist(p, a))
        hdg = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360   # compass heading a -> b
        n = int(re.match(r"\d+", ref).group()) * 10 if re.match(r"\d+", ref) else hdg
        if abs((hdg - n + 180) % 360 - 180) > 90:
            a, b = b, a   # first end is the threshold of the first designator
        rw.append([ref, *flat([a, b]), round(widths.get(ref) or 45)])
    r = max(math.hypot(*p) for v in runways.values() for p in xy(v))

    taxi, aprons, terminals = [], [], []
    for e in elements:
        t = e.get("tags", {})
        if t.get("aeroway") == "taxiway":   # kept only where it stays inside the capacity ring
            run = []
            for p in simplify(xy(ll(e["geometry"])), 5) + [(1e9, 0)]:
                if math.hypot(*p) < 1.04 * r:
                    run.append(p)
                    continue
                if len(run) > 1:
                    taxi.append(flat(run))
                run = []
        elif t.get("aeroway") in ("apron", "terminal") or t.get("building") == "terminal":
            parts = [e["geometry"]] if e["type"] == "way" else [m["geometry"] for m in e["members"] if m["role"] == "outer"]
            for ring in stitch([[(p["lon"], p["lat"]) for p in g] for g in parts]):
                pts = xy(ring[:-1] if ring[0] == ring[-1] else ring)
                cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
                if len(pts) >= 3 and math.hypot(cx, cy) < 1.04 * r:
                    (aprons if t.get("aeroway") == "apron" else terminals).append(flat(simplify(pts, 2)))
    return {"r": round(r), "runways": rw, "taxiways": taxi, "aprons": aprons, "terminals": terminals}


def main():
    fields = {}
    for icao, cfg in C.AIRPORTS.items():
        fields[cfg["iata"]] = f = field(overpass(icao))
        print(f"{cfg['iata']}: {len(f['runways'])} runways, {len(f['taxiways'])} taxiways, "
              f"{len(f['aprons'])} aprons, {len(f['terminals'])} terminals")
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}" for k, v in fields.items())
    OUT.write_text("// Generated by dashboard/build_airfields.py from OpenStreetMap (© OpenStreetMap contributors, ODbL).\n"
                   "// Metres from the centre of the runways, x east, y north. r: distance to the farthest runway end.\n"
                   "// runways: [ref, x1, y1, x2, y2, width], (x1, y1) the threshold of the first designator.\n"
                   f"export default {{\n{body},\n}};\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
