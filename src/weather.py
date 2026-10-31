"""METAR cleaning -> hourly airport weather table (one row per airport-hour, UTC).

Every row carries `known_at_utc`, the time the last observation of that hour was published, so the
feature builders can refuse to use anything that was not yet available at forecast time.
"""
import numpy as np
import pandas as pd

from . import aar, config as C

SKY_CEIL = ["BKN", "OVC", "VV"]
WINTER_RE = r"SN|FZ|PL|GR|GS|SG|IC"


def wind_components(speed, direction, heading):
    """Head(+)/tail(-) and crosswind (signed) components of wind relative to a runway heading."""
    ang = np.deg2rad(direction - heading)
    head = np.where(speed == 0, 0.0, speed * np.cos(ang))
    cross = np.where(speed == 0, 0.0, speed * np.sin(ang))
    return head, cross


def estimate_heading(obs):
    """Fallback for airports without a configured runway heading: modal wind direction (10-degree bins)
    of winds >= 8 kt. Runways are aligned with the prevailing wind, so this is a sane default."""
    w = obs[(obs["wind_kt"] >= 8) & obs["wind_dir_deg"].notna()]
    bins = w["wind_dir_deg"] // 10 * 10 + 5
    return float(bins.mode().iloc[0])


def read_raw(icao):
    iata = C.AIRPORTS[icao]["iata"]
    files = sorted(C.RAW_WX.glob(f"asos_{iata}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no METAR files for {iata}; run python -m src.download")
    return pd.concat([pd.read_csv(p, na_values=["M", ""], comment="#", low_memory=False) for p in files],
                     ignore_index=True)


def clean_obs(raw, icao):
    df = raw.copy()
    df["valid_utc"] = pd.to_datetime(df["valid"], utc=True)
    for c in ["tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "mslp", "vsby", "gust", "skyl1", "skyl2", "skyl3", "skyl4"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    for c in ["skyc1", "skyc2", "skyc3", "skyc4", "wxcodes"]:
        df[c] = df[c].astype("string").str.upper() if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    out = pd.DataFrame({
        "airport": icao, "valid_utc": df["valid_utc"],
        "temp_c": (df["tmpf"] - 32) * 5 / 9, "dewpt_c": (df["dwpf"] - 32) * 5 / 9, "relh": df["relh"],
        "slp_hpa": df["mslp"], "wind_dir_deg": df["drct"], "wind_kt": df["sknt"], "gust_kt": df["gust"],
        "vis_mi": df["vsby"], "precip_mm": df["p01i"] * 25.4, "wxcodes": df["wxcodes"]})
    layers = [df[f"skyl{i}"].where(df[f"skyc{i}"].isin(SKY_CEIL)) for i in range(1, 5)]
    out["ceiling_ft"] = pd.concat(layers, axis=1).min(axis=1)
    wx = out["wxcodes"].fillna("")
    out["thunder"] = wx.str.contains("TS")
    out["winter"] = wx.str.contains(WINTER_RE)
    out["precip_type"] = wx.str.contains(r"RA|SN|DZ|GR|GS|PL|SG|IC|UP")
    return out.sort_values("valid_utc").drop_duplicates("valid_utc").reset_index(drop=True)


def hourly_for_airport(icao):
    cfg = C.AIRPORTS[icao]
    obs = clean_obs(read_raw(icao), icao)
    heading = cfg.get("runway_heading_deg")
    if heading is None:
        heading = estimate_heading(obs)
        print(f"WARNING {icao}: no runway heading configured, using prevailing wind {heading:.0f} deg")
    peak = np.fmax(obs["wind_kt"], obs["gust_kt"])
    obs["headwind_kt"], obs["crosswind_kt"] = wind_components(obs["wind_kt"], obs["wind_dir_deg"], heading)
    hp, cp = wind_components(peak, obs["wind_dir_deg"], heading)
    obs["peak_kt"] = peak
    obs["abs_cross_peak"] = np.abs(cp)
    obs["tail_peak"] = np.clip(-hp, 0, None)
    # per-observation weather-adjusted AAR, averaged within the hour below
    obs["aar"], _ = aar.aar_from_conditions(obs["vis_mi"].values, obs["ceiling_ft"].values, obs["abs_cross_peak"].values,
                                            obs["tail_peak"].values, peak.values, obs["thunder"].values,
                                            obs["winter"].values, cfg["aar"])
    obs["hour_utc"] = obs["valid_utc"].dt.floor("h")

    last_cols = ["temp_c", "dewpt_c", "relh", "slp_hpa", "wind_dir_deg", "wind_kt", "gust_kt", "vis_mi",
                 "ceiling_ft", "headwind_kt", "crosswind_kt", "wxcodes"]
    g = obs.groupby("hour_utc")
    last = g.tail(1).set_index("hour_utc")[last_cols + ["valid_utc"]].rename(columns={"valid_utc": "known_at_utc"})
    agg = g.agg(n_obs=("valid_utc", "size"), peak_wind_kt=("peak_kt", "max"), xwind_max_kt=("abs_cross_peak", "max"),
                tailwind_max_kt=("tail_peak", "max"), vis_min_mi=("vis_mi", "min"), ceiling_min_ft=("ceiling_ft", "min"),
                precip_mm=("precip_mm", "max"), thunder=("thunder", "max"), winter=("winter", "max"),
                precip_type=("precip_type", "max"), aar=("aar", "mean"))
    hourly = last.join(agg)
    grid = pd.date_range(pd.Timestamp(C.START, tz="UTC"), pd.Timestamp(C.END, tz="UTC") - pd.Timedelta(hours=1), freq="h")
    hourly = hourly.reindex(grid)
    hourly.index.name = "hour_utc"
    hourly["n_obs"] = hourly["n_obs"].fillna(0).astype(int)
    for c in ["thunder", "winter", "precip_type"]:
        hourly[c] = hourly[c].fillna(False).astype(bool)
    hourly["flight_category"] = aar.flight_category(hourly["vis_mi"].values, hourly["ceiling_ft"].values)
    hourly.loc[hourly["n_obs"] == 0, "flight_category"] = None
    hourly["aar_vfr"] = cfg["aar"]["VFR"]
    hourly["aar_ratio"] = hourly["aar"] / hourly["aar_vfr"]
    hourly.insert(0, "airport", icao)
    hourly["runway_heading_deg"] = heading
    # a missing hour is published (at the latest) 59 minutes after the hour starts
    hourly["known_at_utc"] = hourly["known_at_utc"].fillna(pd.Series(hourly.index + pd.Timedelta(minutes=59), index=hourly.index))
    return hourly.reset_index()


def build_hourly(icaos=None):
    return pd.concat([hourly_for_airport(a) for a in (icaos or C.AIRPORTS)], ignore_index=True)


if __name__ == "__main__":
    C.PROC.mkdir(parents=True, exist_ok=True)
    h = build_hourly()
    h.to_csv(C.PROC / "hourly_weather.csv", index=False)
    print(h.groupby("airport").agg(hours=("hour_utc", "size"), with_obs=("n_obs", lambda s: (s > 0).sum()),
                                   aar_mean=("aar", "mean"), aar_min=("aar", "min")).round(1))
