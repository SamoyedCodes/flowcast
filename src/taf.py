"""TAF (Terminal Aerodrome Forecast) parsing.

For a target hour T and a forecast lead h, take the newest TAF issued at or before T - h and read what
it said about hour T.  Because only TAFs issued before the forecast time are used, this is a genuine
"what did forecasters know at the time" feature (no look-ahead).

Per TAF product:
  * base groups  = the initial group ('Observation' rows) + every FM group; the base for hour T is the
    last group starting at or before T
  * change groups = TEMPO and PROB30 periods that overlap hour T (kept separately: they are the worst
    case the forecaster expects, not the prevailing condition)
"""
import ast

import numpy as np
import pandas as pd

from . import aar, config as C
from .weather import wind_components

MAX_AGE_H = 30          # a TAF is valid for 24-30 h; older products are ignored


def _list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except (ValueError, SyntaxError):
        return []


def read_raw(icao):
    files = sorted(C.RAW_TAF.glob(f"taf_{icao}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no TAF files for {icao}; run python -m src.download")
    d = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
    d = d.drop_duplicates(["product_id", "fx_valid", "ftype", "raw"])
    return d


def parse_groups(icao):
    """One row per TAF group with numeric fields and the issue time."""
    d = read_raw(icao)
    heading = C.AIRPORTS[icao]["runway_heading_deg"]
    out = pd.DataFrame({
        "product_id": d["product_id"],
        "issue": pd.to_datetime(d["valid"], utc=True),
        "start": pd.to_datetime(d["fx_valid"], utc=True),
        "end": pd.to_datetime(d["fx_valid_end"], utc=True),
        "kind": np.where(d["ftype"].isin(["Observation", "Forecast"]), "base", "change"),
        "is_prob": d["ftype"].str.startswith("Prob"),
        "wind_kt": pd.to_numeric(d["sknt"], errors="coerce"),
        "wind_dir_deg": pd.to_numeric(d["drct"], errors="coerce"),
        "gust_kt": pd.to_numeric(d["gust"], errors="coerce").fillna(0.0),
        "vis_mi": pd.to_numeric(d["visibility"], errors="coerce"),
    })
    sky_c = d["skyc"].map(_list)
    sky_l = d["skyl"].map(_list)
    out["ceiling_ft"] = [min([h for c, h in zip(cs, hs) if c in ("BKN", "OVC", "VV") and h is not None], default=np.nan)
                         for cs, hs in zip(sky_c, sky_l)]
    wx = d["presentwx"].map(_list).map(lambda l: " ".join(l))
    out["thunder"] = wx.str.contains("TS") | d["raw"].str.contains("TS", na=False)
    out["winter"] = wx.str.contains(r"SN|FZ|PL|GR|GS|SG|IC")
    out["precip"] = wx.str.contains(r"RA|SN|DZ|GR|GS|PL|SG|IC|SH")
    peak = np.fmax(out["wind_kt"], out["gust_kt"])
    hp, cp = wind_components(peak.values, out["wind_dir_deg"].values, heading)
    # VRB winds have no direction: use the expected |sin| of a random direction (2/pi) for crosswind, no tailwind
    vrb = out["wind_dir_deg"].isna()
    out["peak_kt"] = peak
    out["xwind_kt"] = np.where(vrb, peak * 2 / np.pi, np.abs(cp))
    out["tail_kt"] = np.where(vrb, 0.0, np.clip(-hp, 0, None))
    out["fcat"] = aar.flight_category(out["vis_mi"].values, out["ceiling_ft"].values)
    return out.sort_values(["issue", "product_id", "start"]).reset_index(drop=True)


FIELDS = ["vis_mi", "ceiling_ft", "xwind_kt", "tail_kt", "peak_kt", "thunder", "winter", "precip", "gust_kt", "wind_kt"]
BIG = 10 ** 12


def _lookup(groups, hours, leads):
    """Vectorised: for every (lead, hour) find the newest TAF issued <= hour - lead and read its
    base group for that hour, then widen the 'worst case' with overlapping TEMPO / PROB30 groups."""
    prods = groups.drop_duplicates("product_id")[["product_id", "issue"]].sort_values("issue").reset_index(drop=True)
    pidx = {pid: i for i, pid in enumerate(prods["product_id"])}
    issue_ns = prods["issue"].values.astype("datetime64[ns]")
    sec = lambda x: pd.DatetimeIndex(x).values.astype("datetime64[ns]").astype("int64") // 10 ** 9

    base = groups[groups["kind"] == "base"].copy()
    base["pi"] = base["product_id"].map(pidx)
    base["key"] = base["pi"].astype("int64") * BIG + sec(base["start"])
    base = base.sort_values("key").reset_index(drop=True)
    bkey = base["key"].values

    hours = pd.DatetimeIndex(hours)
    n = len(hours)
    t_sec = np.tile(sec(hours), len(leads))
    lead_arr = np.repeat(list(leads), n)
    cut = np.tile(hours.values.astype("datetime64[ns]"), len(leads)) - lead_arr.astype("timedelta64[h]")
    lp = np.searchsorted(issue_ns, cut, side="right") - 1                      # newest product issued before cut
    age_h = (t_sec - np.where(lp >= 0, issue_ns[np.clip(lp, 0, None)].astype("int64") // 10 ** 9, 0)) / 3600
    ok = (lp >= 0) & (age_h <= MAX_AGE_H)
    bi = np.searchsorted(bkey, np.where(ok, lp, 0).astype("int64") * BIG + t_sec, side="right") - 1
    ok &= (bi >= 0) & (base["pi"].values[np.clip(bi, 0, None)] == lp)

    out = {}
    bidx = np.clip(bi, 0, None)
    for f in FIELDS:
        v = base[f].values[bidx]
        out[f] = np.where(ok, v, np.nan if v.dtype.kind in "fiu" else False)
    out["w_vis"], out["w_ceil"] = out["vis_mi"].copy(), out["ceiling_ft"].copy()
    out["w_xw"], out["w_tail"], out["w_peak"] = out["xwind_kt"].copy(), out["tail_kt"].copy(), out["peak_kt"].copy()
    out["w_ts"], out["w_winter"] = out["thunder"].astype(bool).copy(), out["winter"].astype(bool).copy()
    out["has_tempo"] = np.zeros(len(ok), int)
    out["has_prob"] = np.zeros(len(ok), int)
    out["ts_prob"] = np.zeros(len(ok), int)

    ch = groups[groups["kind"] == "change"].copy()
    ch["pi"] = ch["product_id"].map(pidx)
    order = np.argsort(np.where(ok, lp, -1), kind="stable")
    sorted_lp = np.where(ok, lp, -1)[order]
    for r in ch.itertuples(index=False):
        lo, hi = np.searchsorted(sorted_lp, r.pi, "left"), np.searchsorted(sorted_lp, r.pi, "right")
        if lo == hi:
            continue
        idx = order[lo:hi]
        t = t_sec[idx]
        end = (r.end if pd.notna(r.end) else r.start + pd.Timedelta(hours=4)).timestamp()
        m = (r.start.timestamp() < t + 3600) & (end > t)
        idx = idx[m]
        if not len(idx):
            continue
        out["w_vis"][idx] = np.fmin(out["w_vis"][idx], r.vis_mi)
        out["w_ceil"][idx] = np.fmin(out["w_ceil"][idx], r.ceiling_ft)
        out["w_xw"][idx] = np.fmax(out["w_xw"][idx], r.xwind_kt)
        out["w_tail"][idx] = np.fmax(out["w_tail"][idx], r.tail_kt)
        out["w_peak"][idx] = np.fmax(out["w_peak"][idx], r.peak_kt)
        out["w_ts"][idx] |= bool(r.thunder)
        out["w_winter"][idx] |= bool(r.winter)
        out["has_prob" if r.is_prob else "has_tempo"][idx] = 1
        if r.is_prob and r.thunder:
            out["ts_prob"][idx] = 1
    out["taf_age_h"] = np.where(ok, age_h, np.nan)
    return out, lead_arr, np.tile(hours.values, len(leads))


def taf_features(icao, hours, leads, groups=None):
    """TAF-based forecast features for each target hour (UTC, tz-aware) and each lead.

    Returns a long DataFrame keyed by (airport, hour_utc, lead_h).
    """
    groups = parse_groups(icao) if groups is None else groups
    table = C.AIRPORTS[icao]["aar"]
    hours = pd.DatetimeIndex(hours)
    res, lead_arr, hour_arr = _lookup(groups, hours, leads)
    df = pd.DataFrame(res)
    df.insert(0, "lead_h", lead_arr)
    df.insert(0, "hour_utc", pd.DatetimeIndex(hour_arr, tz="UTC"))
    df.insert(0, "airport", icao)
    ok = df["vis_mi"].notna() | df["ceiling_ft"].notna() | df["wind_kt"].notna()
    df["taf_available"] = ok.astype(int)
    base_aar, base_cat = aar.aar_from_conditions(df["vis_mi"].values, df["ceiling_ft"].values, df["xwind_kt"].values,
                                                 df["tail_kt"].values, df["peak_kt"].values, df["thunder"].fillna(False).astype(bool).values,
                                                 df["winter"].fillna(False).astype(bool).values, table)
    worst_aar, worst_cat = aar.aar_from_conditions(df["w_vis"].values, df["w_ceil"].values, df["w_xw"].values, df["w_tail"].values,
                                                   df["w_peak"].values, df["w_ts"].fillna(False).astype(bool).values,
                                                   df["w_winter"].fillna(False).astype(bool).values, table)
    cat_ord = {c: i for i, c in enumerate(aar.FLIGHT_CATS)}
    ren = {"vis_mi": "taf_vis_mi", "ceiling_ft": "taf_ceiling_ft", "xwind_kt": "taf_xwind_kt", "tail_kt": "taf_tail_kt",
           "peak_kt": "taf_peak_kt", "thunder": "taf_thunder", "winter": "taf_winter", "precip": "taf_precip",
           "gust_kt": "taf_gust_kt", "wind_kt": "taf_wind_kt", "w_vis": "taf_worst_vis_mi", "w_ceil": "taf_worst_ceiling_ft",
           "w_xw": "taf_worst_xwind_kt", "w_tail": "taf_worst_tail_kt", "w_peak": "taf_worst_peak_kt", "w_ts": "taf_worst_thunder",
           "w_winter": "taf_worst_winter", "has_tempo": "taf_has_tempo", "has_prob": "taf_has_prob", "ts_prob": "taf_ts_prob30"}
    df = df.rename(columns=ren)
    for c in ["taf_thunder", "taf_winter", "taf_precip", "taf_worst_thunder", "taf_worst_winter"]:
        df[c] = df[c].astype("float")
    df["taf_fcat"] = pd.Series(base_cat).map(cat_ord).where(df["taf_available"] == 1).values
    df["taf_fcat_worst"] = pd.Series(worst_cat).map(cat_ord).where(df["taf_available"] == 1).values
    df["taf_aar"] = np.where(df["taf_available"] == 1, base_aar, np.nan)
    df["taf_aar_worst"] = np.where(df["taf_available"] == 1, worst_aar, np.nan)
    df["taf_aar_ratio"] = df["taf_aar"] / table["VFR"]
    df["taf_aar_worst_ratio"] = df["taf_aar_worst"] / table["VFR"]
    return df.drop(columns=["w_vis_x"], errors="ignore")


def build_taf_table(hourly, leads, icaos=None):
    parts = []
    for icao in icaos or C.AIRPORTS:
        hrs = pd.DatetimeIndex(hourly.loc[hourly["airport"] == icao, "hour_utc"])
        parts.append(taf_features(icao, hrs, leads))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    h = pd.read_csv(C.PROC / "hourly_weather.csv", parse_dates=["hour_utc", "known_at_utc"])
    t = build_taf_table(h, C.LEADS)
    t.to_parquet(C.PROC / "taf_features.parquet", index=False)
    print(t.groupby(["airport", "lead_h"])["taf_available"].mean().round(3))
