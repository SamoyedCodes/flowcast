"""
Offline tests: synthetic inputs in each source's documented format -> run parsers and the
dataset builder -> assert physics, time handling and leakage rules.
Run: python -m pytest -q tests   (or: python tests/test_pipeline.py)
"""
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_datasets as B  # noqa: E402
import pull_bts  # noqa: E402
import pull_storm_events as SE  # noqa: E402


def test_wind_components():
    # wind from 080 at 10 kt on heading 080: pure headwind
    xw, hw = B.wind_components(np.array([10.0]), np.array([80.0]), 80)
    assert abs(xw[0]) < 1e-9 and abs(hw[0] - 10) < 1e-9
    # wind from 170 on heading 080: pure crosswind
    xw, hw = B.wind_components(np.array([10.0]), np.array([170.0]), 80)
    assert abs(xw[0] - 10) < 1e-9 and abs(hw[0]) < 1e-9
    # wind from 260 on heading 080: pure tailwind (negative headwind)
    xw, hw = B.wind_components(np.array([10.0]), np.array([260.0]), 80)
    assert abs(hw[0] + 10) < 1e-9
    # wind from 360 at 20 kt on DFW 180: tailwind 20
    _, hw = B.wind_components(np.array([20.0]), np.array([360.0]), 180)
    assert abs(hw[0] + 20) < 1e-9
    # wrap-around: from 010 vs heading 350 -> 20 deg off
    xw, hw = B.wind_components(np.array([10.0]), np.array([10.0]), 350)
    assert abs(xw[0] - 10 * np.sin(np.deg2rad(20))) < 1e-9
    # calm -> zeros even with missing direction
    xw, hw = B.wind_components(np.array([0.0]), np.array([np.nan]), 80)
    assert xw[0] == 0 and hw[0] == 0
    # variable (missing dir) with speed -> NaN, never a made-up number
    xw, _ = B.wind_components(np.array([5.0]), np.array([np.nan]), 80)
    assert np.isnan(xw[0])


def test_flight_category():
    c = B.flight_category(np.array([400.0, 900, 2500, 5000, np.nan]), np.array([10.0, 10, 10, 10, 0.5]))
    assert list(c) == ["LIFR", "IFR", "MVFR", "VFR", "LIFR"]


def test_ceiling():
    assert B.ceiling_ft(["FEW", "BKN", "OVC", None], [1000, 2500, 800, None]) == 800
    assert np.isnan(B.ceiling_ft(["FEW", "SCT", None, None], [1000, 2500, None, None]))


def test_storm_tz():
    df = pd.DataFrame({"BEGIN_YEARMONTH": [202407], "BEGIN_DAY": [15], "BEGIN_TIME": [1530],
                       "END_YEARMONTH": [202407], "END_DAY": [15], "END_TIME": [1645],
                       "CZ_TIMEZONE": ["EST-5"]})
    t, bad = SE.to_utc(df)
    assert t.begin_utc.iloc[0] == pd.Timestamp("2024-07-15 20:30") and not bad.iloc[0]


def _mock_bts_zip(path, flights):
    cols = list(pull_bts.KEEP) + ["Year", "Quarter", "DivAirportLandings"]  # extra cols ignored
    df = pd.DataFrame(flights)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_7.csv", buf.getvalue())
        z.writestr("readme.html", "<html/>")


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    proc = tmp / "processed"
    proc.mkdir()
    B.PROC = proc
    rng = np.random.default_rng(0)
    # ---- flights: ATL arrivals every 10 min for 3 days, a tail rotation, one cancellation
    fl = []
    for d in pd.date_range("2024-07-14", "2024-07-16"):
        for k in range(6 * 18):
            hh, mm = 6 + k // 6, (k % 6) * 10
            arr = hh * 100 + mm
            dep_local = pd.Timestamp(d) + pd.Timedelta(hours=hh, minutes=mm) - pd.Timedelta(minutes=120)
            delay = 60 if (d.day == 15 and 18 <= hh <= 20) else int(rng.integers(-5, 20))
            fl.append({"FlightDate": d.strftime("%Y-%m-%d"), "Reporting_Airline": "DL",
                       "Tail_Number": f"N{k:03d}DL", "Flight_Number_Reporting_Airline": k,
                       "Origin": "ORD", "Dest": "ATL",
                       "CRSDepTime": dep_local.hour * 100 + dep_local.minute, "DepDelay": delay,
                       "CRSArrTime": arr, "ArrDelay": delay, "Cancelled": 0, "Diverted": 0,
                       "CRSElapsedTime": 60, "Distance": 606})
    fl[5]["Cancelled"], fl[5]["ArrDelay"], fl[5]["DepDelay"] = 1, np.nan, np.nan
    # rotation: N001DL continues ATL -> DFW after landing
    fl.append({"FlightDate": "2024-07-15", "Reporting_Airline": "DL", "Tail_Number": "N001DL",
               "Flight_Number_Reporting_Airline": 999, "Origin": "ATL", "Dest": "DFW",
               "CRSDepTime": 800, "DepDelay": 10, "CRSArrTime": 920, "ArrDelay": 5,
               "Cancelled": 0, "Diverted": 0, "CRSElapsedTime": 140, "Distance": 731})
    z = tmp / "m.zip"
    _mock_bts_zip(z, fl)
    bts = pull_bts.read_zip(z)
    assert {"FL_DATE", "ORIGIN", "DEST", "ARR_DELAY", "CRS_ELAPSED_TIME"} <= set(bts.columns)
    bts.to_parquet(proc / "bts_flights.parquet", index=False)

    # ---- METAR, IEM onlycomma, missing=empty, UTC; storm passes 2024-07-15 22Z
    rows = []
    for t in pd.date_range("2024-07-13 00:52", "2024-07-17 23:52", freq="h"):
        storm = pd.Timestamp("2024-07-15 21:00") <= t <= pd.Timestamp("2024-07-16 00:00")
        for st, drct in (("ATL", 260), ("DFW", 170)):
            rows.append({"station": st, "valid": t.strftime("%Y-%m-%d %H:%M"), "tmpf": 88, "dwpf": 70,
                         "relh": 55, "drct": drct, "sknt": 25 if storm else 8, "p01i": 0.5 if storm else 0,
                         "alti": 29.80 if storm else 30.00, "mslp": "", "vsby": 2 if storm else 10,
                         "gust": 40 if storm else "", "skyc1": "BKN" if storm else "FEW", "skyc2": "OVC" if storm else "",
                         "skyc3": "", "skyc4": "", "skyl1": 900 if storm else 5000, "skyl2": 2000 if storm else "",
                         "skyl3": "", "skyl4": "", "wxcodes": "+TSRA" if storm else "",
                         "peak_wind_gust": "", "peak_wind_drct": "", "metar": "X"})
    pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False))).to_parquet(proc / "metar.parquet", index=False)

    # ---- TAF (IEM taf.py csv): issued 12Z forecasting TSRA from 21Z; second TAF issued 18Z
    taf = pd.DataFrame([
        {"station": "KATL", "valid": "2024-07-15 12:00", "fx_valid": "2024-07-15 12:00", "fx_valid_end": "", "is_tempo": False, "ftype": "OB",
         "sknt": 8, "drct": 260, "gust": None, "visibility": 6, "presentwx": "", "skyc": "['FEW']", "skyl": "[5000]", "raw": "x", "airport": "ATL"},
        {"station": "KATL", "valid": "2024-07-15 12:00", "fx_valid": "2024-07-15 21:00", "fx_valid_end": "", "is_tempo": False, "ftype": "FM",
         "sknt": 20, "drct": 270, "gust": 35, "visibility": 3, "presentwx": "['TSRA']", "skyc": "['BKN', 'OVC']", "skyl": "[1500, 3000]", "raw": "x", "airport": "ATL"},
        {"station": "KATL", "valid": "2024-07-15 12:00", "fx_valid": "2024-07-15 22:00", "fx_valid_end": "2024-07-16 01:00", "is_tempo": True, "ftype": "TEMPO",
         "sknt": None, "drct": None, "gust": None, "visibility": 1, "presentwx": "['+TSRA']", "skyc": "['OVC']", "skyl": "[800]", "raw": "x", "airport": "ATL"},
        {"station": "KATL", "valid": "2024-07-15 18:00", "fx_valid": "2024-07-15 18:00", "fx_valid_end": "", "is_tempo": False, "ftype": "OB",
         "sknt": 22, "drct": 270, "gust": 38, "visibility": 2, "presentwx": "['TSRA']", "skyc": "['OVC']", "skyl": "[900]", "raw": "x", "airport": "ATL"},
    ])
    for c in ("valid", "fx_valid", "fx_valid_end"):
        taf[c] = pd.to_datetime(taf[c], errors="coerce")
    taf.to_parquet(proc / "taf.parquet", index=False)

    B.main(["ATL", "DFW"], [3, 6], flight_sample=1.0)

    mh = pd.read_parquet(proc / "metar_hourly.parquet")
    a = mh[(mh.airport == "ATL") & (mh.hour_utc == "2024-07-15 22:00")].iloc[0]
    assert a.flight_cat == "IFR" and a.wx_ts == 1
    assert abs(a.tailwind_on_assumed_hdg_kt - 25) < 1e-6  # wind 260 on hdg 080 = tailwind
    assert abs(a.best_end_headwind_kt - 25) < 1e-6        # but runway 26 gets full headwind
    d = mh[(mh.airport == "DFW") & (mh.hour_utc == "2024-07-15 10:00")].iloc[0]
    assert abs(d.xwind_kt - 8 * np.sin(np.deg2rad(10))) < 1e-6

    aar = pd.read_parquet(proc / "aar_horizon.parquet")
    assert (aar.t0_utc == aar.target_hour_utc - pd.to_timedelta(aar.horizon_h, unit="h")).all()
    assert (aar.taf_issued_utc.isna() | (aar.taf_issued_utc <= aar.t0_utc)).all()
    # target 2024-07-15 23Z, horizon 3 -> t0 20Z -> must use the 18Z TAF, not a later one
    r = aar[(aar.airport == "ATL") & (aar.target_hour_utc == "2024-07-15 23:00") & (aar.horizon_h == 3)].iloc[0]
    assert r.taf_issued_utc == pd.Timestamp("2024-07-15 18:00")
    # horizon 6 -> t0 17Z -> only the 12Z TAF exists; its FM21 group (TSRA) + TEMPO +TSRA should show
    r6 = aar[(aar.airport == "ATL") & (aar.target_hour_utc == "2024-07-15 23:00") & (aar.horizon_h == 6)].iloc[0]
    assert r6.taf_issued_utc == pd.Timestamp("2024-07-15 12:00") and r6.taf_ts == 1 and r6.taf_tempo_or_prob_ts == 1
    # obs at t0 must be pre-storm for horizon 6 (t0 17Z) and in-storm for horizon 3? t0 20Z is pre-storm too
    assert r6.obs_wx_ts == 0
    # state at t0 counts only flights landed by t0
    assert (aar.st_arr_2h >= 0).all()

    fl = pd.read_parquet(proc / "flights_horizon.parquet")
    for bad in ("ARR_DELAY", "DEP_DELAY", "WEATHER_DELAY", "act_arr_utc", "CANCELLED"):
        assert bad not in fl.columns, f"post-event column {bad} leaked into features"
    rot = fl[(fl.OP_CARRIER_FL_NUM == 999)]
    # ATL dep 08:00 EDT = 12:00Z; inbound N001DL landed ~11:20Z -> known at h=3? t0=09:00Z -> not yet landed
    # ATL dep 08:00 EDT = 12:00Z; inbound N001DL lands ~10:10Z.
    # departure row, h=3 -> t0 09:00Z -> inbound not landed yet -> unknown
    dep3 = rot[(rot.horizon_h == 3) & (rot.direction == "departure")]
    assert len(dep3) == 1 and dep3.inbound_delay_known.iloc[0] == 0
    # arrival row at DFW (14:20Z), h=3 -> t0 11:20Z -> inbound already landed -> known
    arr3 = rot[(rot.horizon_h == 3) & (rot.direction == "arrival")]
    assert len(arr3) == 1 and arr3.inbound_delay_known.iloc[0] == 1
    assert fl.y_severe.isin([0, 1]).all()
    print("all end-to-end assertions passed")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
