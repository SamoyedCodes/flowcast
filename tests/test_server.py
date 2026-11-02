"""The web dashboard API (dashboard/server.py) returns strict JSON that only looks ahead of the issue time."""
import json
import sys
from pathlib import Path

import pytest

from src import config as C

pytestmark = pytest.mark.skipif(not (C.PROC / "delay_table.parquet").exists(), reason="run run_pipeline.py first")


def test_outlook_is_strict_json_and_only_looks_ahead():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
    import server as S

    o = S.outlook("KDFW", S.parse_t0("2025-02-18T17"), "balanced", 0.25)
    json.dumps(o, allow_nan=False)
    assert [h["lead"] for h in o["hours"]] == list(range(3, 13))
    assert o["now"]["hour"] < o["t0"] and all(h["hour"] < o["t0"] for h in o["hist"])
    assert all(2.5 <= f["h"] < 12.5 for f in o["flights"])
    assert [f["p"] for f in o["flights"]] == sorted((f["p"] for f in o["flights"]), reverse=True)
    assert S.parse_t0("2030-01-01T00") == S.TEST1 and S.parse_t0("2020-01-01T00") == S.TEST0
    for bad in ("", None, "garbage"):
        with pytest.raises(ValueError):
            S.parse_t0(bad)
    assert S.parse_thr(None) == 0.25 and S.parse_thr("0.4") == 0.4
    for bad in ("nan", "5", "0", "abc"):
        with pytest.raises(ValueError):
            S.parse_thr(bad)
    for body in (S.history("KATL", S.TEST1), S.evaluation("KATL", 3), S.meta()):
        json.dumps(body, allow_nan=False)
