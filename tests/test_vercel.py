"""Exercise the deployed handler against the committed replay snapshot."""
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlencode

import pytest

from api.index import handler


@pytest.fixture(scope="module")
def api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get(path, **query):
        conn = HTTPConnection(*server.server_address, timeout=30)
        try:
            conn.request("GET", path + "?" + urlencode(query))
            response = conn.getresponse()
            data = response.read()
            assert len(data) < 4_500_000
            assert response.getheader("Content-Type") == "application/json"
            body = json.loads(data)
            json.dumps(body, allow_nan=False)
            return response.status, body
        finally:
            conn.close()

    yield get
    server.shutdown()
    server.server_close()
    thread.join()


def test_meta(api):
    status, body = api("/api/meta")
    assert status == 200
    assert {a["icao"] for a in body["airports"]} == {"KATL", "KDFW"}
    assert body["default"]["t0"] == "2025-03-15T06:00Z"
    assert body["events"]


@pytest.mark.parametrize("airport", ["KATL", "KDFW"])
@pytest.mark.parametrize("sens", ["balanced", "cautious"])
@pytest.mark.parametrize("thr", [0.15, 0.5])
def test_outlook_controls(api, airport, sens, thr):
    # This hour has sampled flights and explanations at both airports.
    status, body = api("/api/outlook", airport=airport, t0="2025-02-18T17", sens=sens, thr=thr)
    assert status == 200
    assert (body["icao"], body["t0"], body["sens"], body["thr"]) == (airport, "2025-02-18T17:00Z", sens, thr)
    assert [h["lead"] for h in body["hours"]] == list(range(3, 13))
    assert body["n_high"] == sum(f["p"] >= thr for f in body["flights"])
    assert body["flights"] and any(f["drivers"] for f in body["flights"])
    assert all(h["hour"] < body["t0"] for h in body["hist"])


@pytest.mark.parametrize("airport", ["KATL", "KDFW"])
def test_history_and_evaluation(api, airport):
    status, body = api("/api/history", airport=airport, t0="2025-03-15T06:00Z")
    assert status == 200 and body["icao"] == airport
    assert body["daily"] and body["delay"]
    assert len(body["heat"]) == 7 and all(len(row) == 24 for row in body["heat"])
    for lead in range(3, 13):
        status, body = api("/api/eval", airport=airport, lead=lead)
        assert status == 200 and body["lead"] == lead
        assert body["aar"] and body["delay"]
        if lead in (3, 12):
            assert body["ladder"] and body["reliability"]


@pytest.mark.parametrize("path,query,expected", [
    ("/api/meta", {"airport": "UNKNOWN"}, 400),
    ("/api/outlook", {"t0": "invalid"}, 400),
    ("/api/outlook", {"t0": "2025-03-15T06", "thr": "invalid"}, 400),
    ("/api/eval", {"lead": "invalid"}, 400),
    ("/api/eval", {"lead": 99}, 400),
    ("/api/unknown", {}, 404),
])
def test_api_errors(api, path, query, expected):
    status, body = api(path, **query)
    assert status == expected and body["error"]


def test_missing_forecast(api, monkeypatch):
    from dashboard import server

    monkeypatch.setattr(server, "outlook", lambda *args: None)
    status, body = api("/api/outlook", t0="2025-03-15T06")
    assert status == 404 and "No forecast issued" in body["error"]
