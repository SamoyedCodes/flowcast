"""Forecast the hourly Arrival Acceptance Rate 3-12 hours ahead and score it by RMSE.

Target  : y_ratio = weather-adjusted AAR / the airport's VFR (best-case) AAR, so one model serves any airport;
          scores are reported back in arrivals/hour (RMSE_aar = RMSE_ratio * aar_vfr).
Models  : LightGBM (point + 10/90% quantiles) per lead hour.
Baselines (all use only information available at the issue time):
          persistence        last published AAR carried forward
          climatology        mean AAR for that airport / month / local hour in the training years
          TAF-implied        the AAR rule applied to what the newest TAF forecast for that hour
Split   : train 2023-01..2024-06, val 2024-07..2024-12 (early stopping + hyper-parameters), test 2025-01..07.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C

KEYS = {"airport", "hour_utc", "issue_utc", "obs_hour_utc", "lead_h", "y_aar", "y_ratio", "aar_vfr", "y_flight_cat",
        "split", "persist_ratio"}
GRID = [dict(num_leaves=7, min_child_samples=100, learning_rate=0.05, feature_fraction=0.9),
        dict(num_leaves=15, min_child_samples=50, learning_rate=0.05, feature_fraction=0.8),
        dict(num_leaves=31, min_child_samples=100, learning_rate=0.05, feature_fraction=0.8),
        dict(num_leaves=63, min_child_samples=200, learning_rate=0.03, feature_fraction=0.7)]
EVENT_RATIO = 0.80          # a "capacity-reduction event": AAR at least 20% below the airport's best-case rate
TUNE_LEADS = (3, 12)
WEIGHTS = (1.0, 2.0, 4.0)
MAX_RMSE_INCREASE = 0.05


def feature_cols(t):
    return [c for c in t.columns if c not in KEYS and pd.api.types.is_numeric_dtype(t[c])]


def fit_lgb(X, y, Xv, yv, params, objective="regression", alpha=None, rounds=2000, weight=None):
    p = dict(objective=objective, verbosity=-1, seed=0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, **params)
    if alpha is not None:
        p["alpha"] = alpha
    m = lgb.train(p, lgb.Dataset(X, y, weight=weight), rounds, valid_sets=[lgb.Dataset(Xv, yv)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def low_weights(y, w):
    """Sample weights: hours with a real capacity reduction count w times (1 for all other hours)."""
    return np.where(np.asarray(y) <= EVENT_RATIO, w, 1.0)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def climatology(train, test):
    key = ["airport", "month", "hour_local"]
    clim = train.drop_duplicates(["airport", "hour_utc"]).groupby(key)["y_ratio"].mean().rename("clim_ratio")
    return test.join(clim, on=key)["clim_ratio"].values


def score(df, pred_ratio, name, lead, scope):
    y, aarv = df["y_ratio"].values, df["aar_vfr"].values
    p = np.clip(pred_ratio, 0.3, 1.0)
    ev = y <= EVENT_RATIO
    flag = p <= EVENT_RATIO
    tp = int((flag & ev).sum())
    red = ev  # reduction hours only
    return {"lead_h": lead, "model": name, "scope": scope, "n": len(df),
            "rmse_aar": rmse(y * aarv, p * aarv), "mae_aar": float(np.mean(np.abs(y * aarv - p * aarv))),
            "rmse_aar_reduction_hrs": rmse(y[red] * aarv[red], p[red] * aarv[red]) if red.any() else np.nan,
            "event_rate": float(ev.mean()), "event_precision": tp / max(int(flag.sum()), 1),
            "event_recall": tp / max(int(ev.sum()), 1),
            "event_pr_auc": average_precision_score(ev, 1 - p) if ev.any() else np.nan,
            "event_roc_auc": roc_auc_score(ev, 1 - p) if 0 < ev.sum() < len(ev) else np.nan}


def main():
    t = pd.read_parquet(C.PROC / "aar_table.parquet")
    feats = feature_cols(t)
    C.MODELS.mkdir(exist_ok=True, parents=True)
    C.REPORTS.mkdir(exist_ok=True, parents=True)
    json.dump(feats, open(C.MODELS / "aar_features.json", "w"))
    print(f"{len(feats)} features")

    # ---- choose hyper-parameters on validation RMSE at a near and a far lead -----------------
    scores = []
    for cfg in GRID:
        s = 0
        for lead in TUNE_LEADS:
            d = t[t.lead_h == lead]
            tr, va = d[d.split == "train"], d[d.split == "val"]
            m = fit_lgb(tr[feats], tr["y_ratio"], va[feats], va["y_ratio"], cfg)
            s += rmse(va["y_ratio"], m.predict(va[feats], num_iteration=m.best_iteration))
        scores.append(s / len(TUNE_LEADS))
    best = GRID[int(np.argmin(scores))]
    print("val RMSE (ratio) by config:", np.round(scores, 4), "-> chosen", best)
    json.dump(best, open(C.MODELS / "aar_params.json", "w"))

    # ---- a least-squares model shrinks the rare, severe hours towards the mean. Up-weight capacity-reduction
    # hours, choosing the weight on VALIDATION data only by a rule fixed in advance: smallest RMSE on reduction
    # hours among weights whose overall RMSE is at most 5% worse than unweighted.
    cand = {}
    for w in WEIGHTS:
        ov, rd = [], []
        for lead in TUNE_LEADS:
            d = t[t.lead_h == lead]
            tr, va = d[d.split == "train"], d[d.split == "val"]
            m = fit_lgb(tr[feats], tr["y_ratio"], va[feats], va["y_ratio"], best, weight=low_weights(tr["y_ratio"].values, w))
            pr = np.clip(m.predict(va[feats], num_iteration=m.best_iteration), 0.3, 1.0)
            y, a = va["y_ratio"].values, va["aar_vfr"].values
            ev = y <= EVENT_RATIO
            ov.append(rmse(y * a, pr * a))
            rd.append(rmse(y[ev] * a[ev], pr[ev] * a[ev]))
        cand[w] = (float(np.mean(ov)), float(np.mean(rd)))
    ok = {w: v for w, v in cand.items() if v[0] <= cand[1.0][0] * (1 + MAX_RMSE_INCREASE)}
    low_w = min(ok, key=lambda w: ok[w][1])
    print("val (overall RMSE, reduction-hours RMSE) by low-AAR weight:", {w: tuple(round(x, 2) for x in v) for w, v in cand.items()}, "-> chosen", low_w)
    json.dump({"low_weight": low_w, "val_by_weight": {str(w): v for w, v in cand.items()}}, open(C.MODELS / "aar_weight.json", "w"))

    rows, preds = [], []
    for lead in sorted(t.lead_h.unique()):
        d = t[t.lead_h == lead]
        tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
        m = fit_lgb(tr[feats], tr["y_ratio"], va[feats], va["y_ratio"], best, weight=low_weights(tr["y_ratio"].values, low_w))
        q10 = fit_lgb(tr[feats], tr["y_ratio"], va[feats], va["y_ratio"], best, "quantile", 0.10)
        q90 = fit_lgb(tr[feats], tr["y_ratio"], va[feats], va["y_ratio"], best, "quantile", 0.90)
        for name, mod in (("mean", m), ("q10", q10), ("q90", q90)):
            mod.save_model(str(C.MODELS / f"aar_lead{lead}_{name}.txt"), num_iteration=mod.best_iteration)
        cand = {
            "persistence": te["persist_ratio"].fillna(1.0).values,
            "climatology": climatology(tr, te),
            "taf_implied": te["taf_aar_ratio"].fillna(te["persist_ratio"]).fillna(1.0).values,
            "taf_worst_case": te["taf_aar_worst_ratio"].fillna(te["persist_ratio"]).fillna(1.0).values,
            "lightgbm": m.predict(te[feats], num_iteration=m.best_iteration),
        }
        for name, p in cand.items():
            rows.append(score(te, p, name, lead, "all"))
            for ap in sorted(te.airport.unique()):
                mk = (te.airport == ap).values
                rows.append(score(te[mk], p[mk], name, lead, ap))
        # uncertainty band calibration
        lo, hi = q10.predict(te[feats], num_iteration=q10.best_iteration), q90.predict(te[feats], num_iteration=q90.best_iteration)
        cov = float(np.mean((te["y_ratio"].values >= lo - 1e-9) & (te["y_ratio"].values <= hi + 1e-9)))
        rows.append({"lead_h": lead, "model": "lightgbm_80pct_band", "scope": "all", "n": len(te), "band_coverage": cov})
        out_idx = d[d.split.isin(["val", "test"])]
        pv = m.predict(out_idx[feats], num_iteration=m.best_iteration)
        preds.append(out_idx[["airport", "hour_utc", "lead_h", "issue_utc", "split", "y_ratio", "aar_vfr", "persist_ratio", "taf_aar_ratio",
                        "y_flight_cat", "taf_thunder", "taf_worst_thunder", "taf_fcat", "taf_xwind_kt", "taf_tail_kt"]].assign(
            pred_ratio=np.clip(pv, 0.3, 1.0),
            lo_ratio=np.clip(np.minimum(q10.predict(out_idx[feats], num_iteration=q10.best_iteration), pv), 0.3, 1.0),
            hi_ratio=np.clip(np.maximum(q90.predict(out_idx[feats], num_iteration=q90.best_iteration), pv), 0.3, 1.0)))
        got = {r["model"]: r["rmse_aar"] for r in rows if r["lead_h"] == lead and r["scope"] == "all" and "rmse_aar" in r}
        print(f"lead {lead:2d}h  persistence {got['persistence']:.2f}  taf {got['taf_implied']:.2f}  lightgbm {got['lightgbm']:.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "aar_metrics.csv", index=False)
    pd.concat(preds).to_parquet(C.PROC / "aar_predictions.parquet", index=False)

    # ---- leave-one-airport-out: does it generalise to an airport it never saw? ------------------
    loo = []
    for lead in TUNE_LEADS:
        d = t[t.lead_h == lead]
        for held in sorted(d.airport.unique()):
            tr = d[(d.airport != held) & (d.split == "train")]
            va = d[(d.airport != held) & (d.split == "val")]
            te = d[(d.airport == held) & (d.split == "test")]
            m = fit_lgb(tr[feats], tr["y_ratio"], va[feats], va["y_ratio"], best, weight=low_weights(tr["y_ratio"].values, low_w))
            loo.append(score(te, m.predict(te[feats], num_iteration=m.best_iteration), "lightgbm_unseen_airport", lead, held))
    pd.DataFrame(loo).to_csv(C.REPORTS / "aar_unseen_airport.csv", index=False)

    # ---- feature importance (gain) of the 3 h and 12 h models --------------------------------
    imp = {}
    for lead in TUNE_LEADS:
        mod = lgb.Booster(model_file=str(C.MODELS / f"aar_lead{lead}_mean.txt"))
        imp[f"lead{lead}"] = dict(zip(feats, mod.feature_importance("gain")))
    pd.DataFrame(imp).sort_values("lead3", ascending=False).to_csv(C.REPORTS / "aar_feature_importance.csv")

    pd.set_option("display.width", 200)
    a = res[(res.scope == "all") & res.rmse_aar.notna()]
    print("\nTEST RMSE (arrivals/hour)\n", a.pivot(index="model", columns="lead_h", values="rmse_aar").round(2))
    print("\nTEST RMSE on capacity-reduction hours only\n", a.pivot(index="model", columns="lead_h", values="rmse_aar_reduction_hrs").round(2))
    print("\nTEST event PR-AUC (reduction >=20%)\n", a.pivot(index="model", columns="lead_h", values="event_pr_auc").round(3))
    print("\n80% band coverage:", res[res.model == "lightgbm_80pct_band"].set_index("lead_h")["band_coverage"].round(3).to_dict())
    print("\nunseen airport\n", pd.DataFrame(loo)[["lead_h", "scope", "rmse_aar", "event_pr_auc"]].round(3))


if __name__ == "__main__":
    main()
