"""Exercise the deployed handler against the committed replay snapshot."""
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
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


@pytest.mark.parametrize("airport", ["KATL", "KDFW"])
@pytest.mark.parametrize("lead", [3, 6, 9, 12])
def test_storm_reports(api, airport, lead):
    status, body = api("/api/storm", airport=airport, lead=lead)
    assert status == 200 and body["icao"] == airport and body["lead"] == lead
    assert body["hours"] > 0 and body["arr"]["model"] > 0
    assert [r["lead"] for r in body["by_lead"]] == [3, 6, 9, 12]
    assert {r["dir"] for r in body["delay"]} == {"arrival", "departure"}
    assert body["fullbts"]


def test_explain_without_key(api, monkeypatch):
    from dashboard import server

    monkeypatch.setattr(server, "api_key", lambda: None)
    monkeypatch.setattr(server, "meta_cached", server.meta)
    status, body = api("/api/meta")
    assert status == 200 and body["explain"] is False
    status, body = api("/api/explain", t0="2025-03-15T06")
    assert status == 400 and "no ANTHROPIC_API_KEY" in body["error"]


@pytest.mark.parametrize("upstream_fails", [False, True])
def test_explain_with_mocked_claude(api, monkeypatch, upstream_fails):
    from dashboard import server

    def post(*args, **kwargs):
        if upstream_fails:
            raise server.requests.Timeout("private upstream details")
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"content": [{"type": "text", "text": "Test outlook summary."}]})

    monkeypatch.setattr(server, "api_key", lambda: "test-key")
    monkeypatch.setattr(server, "meta_cached", server.meta)
    monkeypatch.setattr(server.requests, "post", post)
    status, body = api("/api/meta")
    assert status == 200 and body["explain"] is True
    status, body = api("/api/explain", airport="KDFW", t0="2025-02-18T17")
    if upstream_fails:
        assert status == 502 and "Claude summary unavailable" in body["error"]
        assert "private upstream details" not in body["error"]
    else:
        assert status == 200 and body["text"] == "Test outlook summary."


@pytest.mark.parametrize("path,query,expected", [
    ("/api/meta", {"airport": "UNKNOWN"}, 400),
    ("/api/outlook", {"t0": "invalid"}, 400),
    ("/api/outlook", {"t0": "2025-03-15T06", "thr": "invalid"}, 400),
    ("/api/eval", {"lead": "invalid"}, 400),
    ("/api/eval", {"lead": 99}, 400),
    ("/api/storm", {"lead": 4}, 400),
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
