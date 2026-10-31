"""Weather-adjusted Arrival Acceptance Rate (AAR).

No public dataset publishes the hourly AAR a facility actually called, so the target is built from
published facility practice:

  1. BASE RATE by flight category (VFR/MVFR/IFR/LIFR) = the facility-called AAR in the FAA Airport
     Capacity Profiles (ATL 132/124/110/98, DFW 114/102/90/80*).  *DFW LIFR is scaled, see config.
  2. Multiplicative REDUCTIONS for the hazards that cut runway throughput even when ceiling and
     visibility are fine: runway crosswind, tailwind, thunderstorms, wind gusts and winter/hail precip.

The reduction parameters below are engineering assumptions (they are NOT published per-airport
numbers).  They are kept in one dict so they can be changed; reports/aar_sensitivity.csv shows how the
headline model comparison changes when they are perturbed, and validate_against_flights checks that
low-AAR hours really do coincide with worse flight performance in the organiser data.
"""
import numpy as np

PARAMS = {
    "xwind_start_kt": 15.0, "xwind_per_kt": 0.025,     # -2.5% per kt of crosswind above 15 kt
    "tail_start_kt": 5.0, "tail_per_kt": 0.04,         # -4% per kt of tailwind above 5 kt
    "gust_start_kt": 30.0, "gust_factor": 0.90,        # peak wind >= 30 kt
    "thunder_factor": 0.70,                            # thunderstorm at the field
    "winter_factor": 0.85,                             # snow / freezing / ice pellets / hail
    "floor": 0.45,                                     # never below 45% of the base rate
}

FLIGHT_CATS = ["VFR", "MVFR", "IFR", "LIFR"]


def flight_category(vis_mi, ceiling_ft):
    """US flight category. Missing ceiling = no ceiling; missing visibility = 10 mi."""
    ceil = np.where(np.isnan(ceiling_ft), np.inf, ceiling_ft)
    vis = np.where(np.isnan(vis_mi), 10.0, vis_mi)
    return np.select([(ceil < 500) | (vis < 1), (ceil < 1000) | (vis < 3), (ceil <= 3000) | (vis <= 5)],
                     ["LIFR", "IFR", "MVFR"], default="VFR")


def aar_from_conditions(vis_mi, ceiling_ft, xwind_kt, tailwind_kt, peak_wind_kt, thunder, winter, table, params=PARAMS):
    """Vectorised AAR (arrivals/hour) and flight category. Inputs are arrays; NaN wind = no reduction."""
    p = params
    cat = flight_category(np.asarray(vis_mi, float), np.asarray(ceiling_ft, float))
    base = np.array([table[c] for c in cat], dtype=float)
    xw = np.nan_to_num(np.abs(np.asarray(xwind_kt, float)))
    tw = np.nan_to_num(np.asarray(tailwind_kt, float)).clip(min=0)
    pk = np.nan_to_num(np.asarray(peak_wind_kt, float))
    f = np.ones(len(base))
    f *= 1 - p["xwind_per_kt"] * np.clip(xw - p["xwind_start_kt"], 0, None)
    f *= 1 - p["tail_per_kt"] * np.clip(tw - p["tail_start_kt"], 0, None)
    f *= np.where(pk >= p["gust_start_kt"], p["gust_factor"], 1.0)
    f *= np.where(np.asarray(thunder, bool), p["thunder_factor"], 1.0)
    f *= np.where(np.asarray(winter, bool), p["winter_factor"], 1.0)
    f = np.clip(f, p["floor"], 1.0)
    return base * f, cat
