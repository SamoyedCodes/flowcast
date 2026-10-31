"""Rule-based decision support built on top of the forecasts.

The ML models say WHAT is likely (AAR, delay risk, with uncertainty).  These rules turn that into plain
advice for the traffic management unit.  They are deliberately simple, transparent and conservative: the
duty manager decides, the tool only proposes.
"""
import numpy as np
import pandas as pd

MODERATE, SEVERE = 0.80, 0.60          # AAR as a share of the airport's best-case rate
HAZARD_TEXT = {
    "thunder": "thunderstorms in the forecast",
    "low_vis": "low ceiling / visibility (IFR or worse)",
    "xwind": "strong crosswind on the active runway",
    "tail": "tailwind component on the active runway",
}


def hazards_for_hour(row):
    """Which forecast ingredients explain a low AAR at this target hour (from the newest TAF)."""
    h = []
    if row.get("taf_worst_thunder", 0) == 1 or row.get("taf_thunder", 0) == 1:
        h.append("thunder")
    if row.get("taf_fcat", 0) >= 2:
        h.append("low_vis")
    if row.get("taf_xwind_kt", 0) >= 15:
        h.append("xwind")
    if row.get("taf_tail_kt", 0) >= 5:
        h.append("tail")
    return h


def capacity_windows(fc, basis="pred_ratio"):
    """Group consecutive forecast hours whose predicted AAR share is below the moderate threshold.

    fc columns: hour_utc, pred_ratio, lo_ratio, hi_ratio, aar_vfr (+ optional taf hazard columns).
    basis="pred_ratio" is the balanced setting (2025 test, 3 h ahead: precision 0.71, recall 0.69);
    basis="lo_ratio" is the cautious setting that alerts on the 10% lower bound (precision 0.47, recall 0.90)."""
    fc = fc.sort_values("hour_utc").reset_index(drop=True)
    low = (fc[basis] <= MODERATE).values
    out, i = [], 0
    while i < len(fc):
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(fc) and low[j + 1]:
            j += 1
        seg = fc.iloc[i:j + 1]
        worst = seg.loc[seg[basis].idxmin()]
        haz = []
        for _, r in seg.iterrows():
            haz += hazards_for_hour(r)
        out.append({
            "start": seg["hour_utc"].iloc[0], "end": seg["hour_utc"].iloc[-1] + pd.Timedelta(hours=1),
            "min_aar": float(worst["pred_ratio"] * worst["aar_vfr"]),
            "min_aar_low": float(worst["lo_ratio"] * worst["aar_vfr"]),
            "min_aar_high": float(worst["hi_ratio"] * worst["aar_vfr"]),
            "pct_reduction": float(100 * (1 - worst["pred_ratio"])),
            "severity": "severe" if worst[basis] <= SEVERE else "moderate",
            "confidence": "high" if worst["hi_ratio"] <= MODERATE else "medium" if worst["lo_ratio"] <= MODERATE else "low",
            "hazards": sorted(set(haz), key=haz.index),
        })
        i = j + 1
    return out


def advice(window, delay_high_risk_flights=0):
    """Plain-language recommendation for one forecast capacity window."""
    t0, t1 = window["start"].strftime("%H:%MZ"), window["end"].strftime("%H:%MZ")
    why = ", ".join(HAZARD_TEXT[h] for h in window["hazards"]) or "weather-driven capacity loss"
    if window["pct_reduction"] < 100 * (1 - MODERATE):
        # cautious mode: the point forecast is milder than the alert threshold, the lower bound is not
        lines = [f"Cautious alert: the central forecast is only {window['pct_reduction']:.0f}% below best case ({window['min_aar']:.0f}/h) "
                 f"but the plausible range reaches {window['min_aar_low']:.0f}/h between {t0} and {t1}. Driver: {why}. "
                 f"Roughly one in two such alerts has verified in 2025 testing; treat as a heads-up, not a decision trigger."]
    else:
        lines = [f"Expect arrival rate near {window['min_aar']:.0f}/h "
                 f"(-{window['pct_reduction']:.0f}% vs best case; plausible range {window['min_aar_low']:.0f}-{window['min_aar_high']:.0f}) "
                 f"between {t0} and {t1}. Driver: {why}. Confidence: {window['confidence']}."]
    if window["severity"] == "severe":
        lines += ["Open a TMU coordination call now and evaluate a ground delay program / arrival metering for the window.",
                  "Ask airlines to review proactive cancellations and swaps for flights in the window to limit airborne holding.",
                  "Brief tower and approach on likely runway configuration change; check fuel-critical and diversion planning."]
    else:
        lines += ["Alert the TMU and the airline operations desks; pre-position arrival metering for the window.",
                  "Review departure demand against the reduced rate and consider early gate holds for departures with tight turns."]
    if "tail" in window["hazards"] or "xwind" in window["hazards"]:
        lines.append("Check runway configuration options for the forecast wind before the window starts.")
    if delay_high_risk_flights:
        lines.append(f"{delay_high_risk_flights} sampled flights in the next 12 h carry an elevated severe-delay risk (see table).")
    lines.append("This is decision support: confirm against the latest METAR/TAF before acting.")
    return lines


def risk_level(p):
    p = np.asarray(p, float)
    return np.where(p >= 0.30, "high", np.where(p >= 0.20, "elevated", "normal"))
