"""How much do the AAR conclusions depend on the assumed hazard reductions in src/aar.py?

For each scenario the AAR target, the persistence baseline, the TAF-implied baseline and the LightGBM model
are all rebuilt from scratch with perturbed parameters, then scored on the 2025 test period.
Nothing in data/processed is overwritten.  Output: reports/aar_sensitivity.csv

validate_against_traffic checks the AAR rule itself against real airport-wide BTS arrivals.
Output: reports/aar_vs_traffic.csv
"""
import copy
import json

import pandas as pd
from scipy.stats import spearmanr

from . import aar, config as C, features as F, taf as tafmod, train_aar as T, weather

SCENARIOS = {
    "base": {},
    "thunderstorm x0.50": {"thunder_factor": 0.50},
    "thunderstorm x0.85": {"thunder_factor": 0.85},
    "crosswind 1.25%/kt": {"xwind_per_kt": 0.0125},
    "crosswind 5%/kt": {"xwind_per_kt": 0.05},
    "tailwind 2%/kt": {"tail_per_kt": 0.02},
    "tailwind 8%/kt": {"tail_per_kt": 0.08},
    "category-only (no hazards)": {"xwind_per_kt": 0.0, "tail_per_kt": 0.0, "thunder_factor": 1.0,
                                   "winter_factor": 1.0, "gust_factor": 1.0},
}
LEADS = (3, 12)


AAR_BANDS = [0, 0.7, 0.85, 0.95, 1.0001]


def validate_against_traffic(hourly, bts=None):
    """Does a lower rule-based AAR coincide with fewer real arrivals?  Only busy hours (scheduled BTS arrivals at or
    above the airport's 75th percentile) can show it, because a quiet hour never reaches capacity."""
    bts = F.load_traffic() if bts is None else bts
    rows = []
    for icao, cfg in C.AIRPORTS.items():
        arr = bts[bts["Dest"] == cfg["iata"]]
        h = hourly[(hourly["airport"] == icao) & (hourly["n_obs"] > 0)].set_index("hour_utc")
        h = h.assign(scheduled=arr["sched_arr_utc"].dt.floor("h").value_counts().reindex(h.index, fill_value=0),
                     actual=arr["wheels_on_utc"].dt.floor("h").value_counts().reindex(h.index, fill_value=0))
        busy = h[h["scheduled"] >= h["scheduled"].quantile(0.75)].copy()
        busy["completion"] = busy["actual"] / busy["scheduled"]
        rho_actual = spearmanr(busy["aar"], busy["actual"]).statistic
        rho_completion = spearmanr(busy["aar"], busy["completion"]).statistic
        busy["aar_band"] = pd.cut(busy["aar_ratio"], AAR_BANDS, right=False)
        for band, g in busy.groupby("aar_band", observed=True):
            rows.append({"airport": icao, "aar_ratio_band": str(band), "busy_hours": len(g),
                         "mean_rule_aar": g["aar"].mean(), "mean_scheduled_arrivals": g["scheduled"].mean(),
                         "mean_actual_arrivals": g["actual"].mean(), "actual_to_scheduled": g["completion"].mean(),
                         "spearman_aar_vs_actual": rho_actual, "spearman_aar_vs_completion": rho_completion})
    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "aar_vs_traffic.csv", index=False)
    print(res.round(3).to_string(index=False))
    return res


def main():
    original = copy.deepcopy(aar.PARAMS)
    best = json.load(open(C.MODELS / "aar_params.json"))
    low_w = json.load(open(C.MODELS / "aar_weight.json"))["low_weight"]
    rows = []
    try:
        for name, override in SCENARIOS.items():
            aar.PARAMS.clear()
            aar.PARAMS.update({**original, **override})
            h = weather.build_hourly()
            taf_t = tafmod.build_taf_table(h, LEADS)
            t = F.build_aar_table(h, taf_t, leads=LEADS)
            feats = T.feature_cols(t)
            for lead in LEADS:
                d = t[t.lead_h == lead]
                tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
                m = T.fit_lgb(tr[feats], tr["y_ratio"], va[feats], va["y_ratio"], best, weight=T.low_weights(tr["y_ratio"].values, low_w))
                cand = {"persistence": te["persist_ratio"].fillna(1.0).values,
                        "taf_implied": te["taf_aar_ratio"].fillna(te["persist_ratio"]).fillna(1.0).values,
                        "lightgbm": m.predict(te[feats], num_iteration=m.best_iteration)}
                for model, p in cand.items():
                    s = T.score(te, p, model, lead, "all")
                    rows.append({"scenario": name, **s})
            print("done:", name)
    finally:
        aar.PARAMS.clear()
        aar.PARAMS.update(original)
    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "aar_sensitivity.csv", index=False)
    piv = res.pivot_table(index="scenario", columns=["lead_h", "model"], values="rmse_aar").round(2)
    print(piv)
    win = res.pivot_table(index=["scenario", "lead_h"], columns="model", values="rmse_aar")
    print("\nLightGBM beats both baselines in every scenario/lead:",
          bool(((win["lightgbm"] < win["persistence"]) & (win["lightgbm"] < win["taf_implied"])).all()))


if __name__ == "__main__":
    main()
