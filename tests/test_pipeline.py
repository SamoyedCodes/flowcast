"""Invariants that must hold for the results to be trustworthy.  Run:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from src import aar, config as C, features, taf, weather

PROC = C.PROC
needs_data = pytest.mark.skipif(not (PROC / "delay_table.parquet").exists(), reason="run run_pipeline.py first")
ATL = C.AIRPORTS["KATL"]


# ---------------------------------------------------------------- physics / rules
def test_wind_components_follow_runway_heading():
    head, cross = weather.wind_components(np.array([10.0]), np.array([80.0]), 80)      # wind straight down the runway
    assert head[0] == pytest.approx(10) and cross[0] == pytest.approx(0, abs=1e-9)
    head, cross = weather.wind_components(np.array([10.0]), np.array([170.0]), 80)     # 90 degrees off the runway
    assert head[0] == pytest.approx(0, abs=1e-9) and abs(cross[0]) == pytest.approx(10)
    head, _ = weather.wind_components(np.array([10.0]), np.array([260.0]), 80)         # blowing from behind
    assert head[0] == pytest.approx(-10)
    head, cross = weather.wind_components(np.array([0.0]), np.array([np.nan]), 80)      # calm, no direction
    assert head[0] == 0 and cross[0] == 0


def test_flight_categories():
    cat = aar.flight_category(np.array([10, 4, 2, 0.5, 10, np.nan]), np.array([np.nan, 5000, 2000, 5000, 400, np.nan]))
    assert list(cat) == ["VFR", "MVFR", "IFR", "LIFR", "LIFR", "VFR"]


def test_aar_rule():
    one = lambda **k: aar.aar_from_conditions(*[np.array([v]) for v in (k.get("vis", 10), k.get("ceil", np.nan), k.get("xw", 0),
                                                                          k.get("tw", 0), k.get("pk", 5), k.get("ts", False),
                                                                          k.get("winter", False))], ATL["aar"])[0][0]
    assert one() == 132                                           # clear day = FAA called VFR rate
    assert one(vis=2) == 110                                      # IFR rate
    assert one(ts=True) == pytest.approx(132 * 0.7)
    assert one(xw=25) == pytest.approx(132 * (1 - 0.025 * 10))
    assert one(tw=15) == pytest.approx(132 * (1 - 0.04 * 10))
    assert one(vis=0.5, ts=True, xw=60, tw=40, winter=True) == pytest.approx(98 * 0.45)   # floor
    assert one(xw=10, tw=3) == 132                                # below thresholds: no reduction


# ---------------------------------------------------------------- data integrity
@needs_data
def test_splits_are_chronological_and_disjoint():
    for name, tcol in (("aar_table", "hour_utc"), ("delay_table", "event_utc")):
        t = pd.read_parquet(PROC / f"{name}.parquet", columns=[tcol, "split"])
        mx = t.groupby("split")[tcol].max()
        mn = t.groupby("split")[tcol].min()
        assert mx["train"] < mn["val"] and mx["val"] < mn["test"], name


@needs_data
def test_weather_features_published_before_issue_time():
    t = pd.read_parquet(PROC / "delay_table.parquet", columns=["issue_utc", "obs_hour_utc", "lead_h", "event_utc"])
    assert ((t["event_utc"] - t["issue_utc"]) == pd.to_timedelta(t["lead_h"], unit="h")).all()
    # the observation hour started at least 1 h before the forecast was issued (published at most 59 min after hour start)
    assert ((t["issue_utc"] - t["obs_hour_utc"]) >= pd.Timedelta(minutes=1)).all()
    a = pd.read_parquet(PROC / "aar_table.parquet", columns=["issue_utc", "obs_hour_utc", "hour_utc", "lead_h"])
    assert (a["obs_hour_utc"] < a["issue_utc"]).all() and (a["obs_hour_utc"] < a["hour_utc"]).all()


@needs_data
def test_taf_issued_before_forecast_time():
    a = pd.read_parquet(PROC / "aar_table.parquet", columns=["taf_age_h", "lead_h"]).dropna()
    # age = target hour - TAF issue time; a TAF used at lead h must have been issued at least h hours before the hour
    assert (a["taf_age_h"] >= a["lead_h"] - 1e-6).all()
    assert (a["taf_age_h"] <= taf.MAX_AGE_H).all()


@needs_data
def test_no_leaky_columns_in_model_features():
    feats = set(json.load(open(C.MODELS / "delay_features.json")))
    banned = {"delay_min", "severe_delay", "TAXI_OUT", "TAXI_IN", "ARR_DELAY", "DEP_DELAY", "CANCELLED", "event_utc", "issue_utc", "flight_id"}
    assert not (feats & banned)
    assert not any(c.startswith(("y_", "arr_wx")) for c in feats)
    af = set(json.load(open(C.MODELS / "aar_features.json")))
    assert not any(c.startswith("y_") or c in {"y_ratio", "y_aar"} for c in af)


@needs_data
def test_recent_disruption_excludes_the_flight_itself():
    t = pd.read_parquet(PROC / "delay_table.parquet", columns=["lead_h", "rec_dep_n_6h", "rec_dep_n_24h"])
    assert (t["rec_dep_n_6h"] <= t["rec_dep_n_24h"]).all()


@needs_data
def test_aar_target_range_and_missing():
    a = pd.read_parquet(PROC / "aar_table.parquet", columns=["y_ratio", "airport"])
    assert a["y_ratio"].between(0.3, 1.0001).all()
    assert a["airport"].nunique() == len(C.AIRPORTS)


def test_heading_fallback_uses_prevailing_wind():
    obs = pd.DataFrame({"wind_kt": [10, 12, 15, 9, 3], "wind_dir_deg": [183, 190, 187, 186, 90]})
    assert weather.estimate_heading(obs) == 185.0
