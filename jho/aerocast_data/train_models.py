"""
Train and evaluate the AAR-proxy regressor and the severe-delay (>45 min) classifier.

Inputs (from build_datasets.py): data/processed/aar_horizon.parquet, flights_horizon.parquet
Outputs: data/models/*.txt (LightGBM), data/reports/*.csv, data/processed/*_predictions.parquet

Every model is compared with baselines on the same test rows:
  AAR:      persistence (arrivals in the hour ending at t0), climatology (airport x local hour x
            weekday x month, train only), scheduled arrivals
  Severe:   climatology rate (airport x direction x local hour, train only)
Intervals: split-conformal per airport x horizon, fitted on validation residuals (90%).
Threshold: the severe-risk alert threshold maximises F1 on validation, then is frozen for test.

Usage: python train_models.py [--min-test-rows 50]
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)

from common import DATA, PROC

MODELS, REPORTS = DATA / "models", DATA / "reports"
COVERAGE = 0.90
# targets, scoring masks and timestamps: never inputs
AAR_NOT_FEATURES = {"target_hour_utc", "t0_utc", "taf_issued_utc", "actual_arrivals", "throughput_ratio",
                    "arrivals_norm", "severe_arr_share", "high_demand_hour", "split", "storm_hour"}
FL_NOT_FEATURES = {"FL_DATE", "TAIL_NUM", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME",
                   "sched_dep_utc", "sched_arr_utc", "prev_sched_arr_utc", "event_utc", "t0_utc", "t0_hour",
                   "target_hour_utc", "taf_issued_utc", "y_severe", "split", "storm_hour"}
CATEGORICAL = ["airport", "direction", "OP_UNIQUE_CARRIER", "obs_flight_cat", "taf_flight_cat"]


def prep(df, drop):
    X = df[[c for c in df.columns if c not in drop]].copy()
    for c in X.columns:
        if c in CATEGORICAL or X[c].dtype == object or str(X[c].dtype) in ("string", "str"):
            X[c] = X[c].astype("category")
        elif X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X


def align_categories(X, ref):
    for c in X.columns:
        if isinstance(ref[c].dtype, pd.CategoricalDtype):
            X[c] = pd.Categorical(X[c].astype(object), categories=ref[c].cat.categories)
    return X


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def add_storm_flag(df, time_col):
    """storm_hour = 1 if a NOAA Storm Event within 50 km of the airport overlaps that UTC hour.
    Evaluation label only: Storm Events are compiled after the event, so this is added after
    training and is never a model input. State-wide zone events are too coarse and are left out."""
    df = df.copy()
    p = PROC / "storm_events.parquet"
    if not p.exists():
        df["storm_hour"] = 0
        return df
    s = pd.read_parquet(p)
    s = s[(s.match == "within_radius") & (s.flag_tz_unparsed == 0)].dropna(subset=["begin_utc", "end_utc"])
    hrs = pd.DataFrame([(ap, h) for ap, b, e in zip(s.airport, s.begin_utc, s.end_utc)
                        for h in pd.date_range(b.floor("h"), e.floor("h"), freq="h")], columns=["airport", "hour"])
    hrs = hrs.drop_duplicates().assign(storm_hour=1)
    hrs["hour"] = hrs.hour.astype("datetime64[ns]")
    key = df[time_col].dt.floor("h").astype("datetime64[ns]").rename("hour")
    df["storm_hour"] = (pd.concat([df.airport, key], axis=1).merge(hrs, on=["airport", "hour"], how="left")
                        .storm_hour.fillna(0).astype(int).values)
    return df


# ---------------------------------------------------------------- AAR proxy
def train_aar(a):
    a = a.copy()
    ah = pd.read_parquet(PROC / "airport_hourly.parquet")
    # persistence: arrivals in the hour that ended at t0 (complete at t0, so no leakage)
    pers = ah[["airport", "hour_utc", "actual_arrivals"]].rename(columns={"actual_arrivals": "base_persistence"})
    pers["t0_utc"] = pers.hour_utc + pd.Timedelta(hours=1)
    a = a.merge(pers.drop(columns="hour_utc"), on=["airport", "t0_utc"], how="left")
    keys = ["airport", "target_hour_local", "dow", "month"]
    clim = a[a.split == "train"].groupby(keys).actual_arrivals.mean().rename("base_climatology")
    a = a.join(clim, on=keys)
    a["base_climatology"] = a.base_climatology.fillna(a[a.split == "train"].groupby("airport").actual_arrivals.transform("mean"))
    a["base_schedule"] = a.sched_arrivals

    drop = AAR_NOT_FEATURES | {"base_persistence", "base_climatology", "base_schedule"}
    tr, va, te = (a[a.split == s] for s in ("train", "val", "test"))
    Xtr = prep(tr, drop)
    Xva, Xte = align_categories(prep(va, drop), Xtr), align_categories(prep(te, drop), Xtr)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    m = lgb.train(params, lgb.Dataset(Xtr, tr.actual_arrivals), 2000,
                  valid_sets=[lgb.Dataset(Xva, va.actual_arrivals)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m.save_model(MODELS / "aar_lgbm.txt")
    a.loc[va.index, "pred"] = m.predict(Xva, num_iteration=m.best_iteration)
    a.loc[te.index, "pred"] = m.predict(Xte, num_iteration=m.best_iteration)

    # split-conformal half-width per airport x horizon from validation absolute residuals
    v = a.loc[va.index]
    q = (v.assign(r=(v.actual_arrivals - v.pred).abs()).groupby(["airport", "horizon_h"]).r
         .quantile(COVERAGE).rename("conformal_halfwidth"))
    a = a.join(q, on=["airport", "horizon_h"])
    a["pred_lo"] = (a.pred - a.conformal_halfwidth).clip(lower=0)
    a["pred_hi"] = a.pred + a.conformal_halfwidth
    a = add_storm_flag(a, "target_hour_utc")

    rows = []
    t = a[a.split == "test"]
    p95 = a[a.split == "train"].groupby("airport").actual_arrivals.quantile(0.95)
    for (ap, H), g in t.groupby(["airport", "horizon_h"]):
        for subset, gg in (("all_hours", g), ("high_demand", g[g.high_demand_hour == 1]),
                           ("storm_hours", g[g.storm_hour == 1])):
            if len(gg) == 0:
                continue
            row = {"airport": ap, "horizon_h": H, "subset": subset, "n": len(gg),
                   "interval_coverage": float(((gg.actual_arrivals >= gg.pred_lo) & (gg.actual_arrivals <= gg.pred_hi)).mean())}
            for name, col in (("model", "pred"), ("persistence", "base_persistence"),
                              ("climatology", "base_climatology"), ("schedule", "base_schedule")):
                ok = gg[col].notna()
                row[f"rmse_{name}"] = rmse(gg.actual_arrivals[ok], gg[col][ok])
                row[f"mae_{name}"] = float(mean_absolute_error(gg.actual_arrivals[ok], gg[col][ok]))
                row[f"nrmse_{name}"] = row[f"rmse_{name}"] / p95[ap]   # normalised by the airport's own p95
            row["model_beats_climatology"] = row["rmse_model"] < row["rmse_climatology"]
            rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(REPORTS / "aar_metrics.csv", index=False)
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    imp.to_csv(REPORTS / "aar_feature_importance.csv", header=["gain"])
    a[["airport", "target_hour_utc", "t0_utc", "horizon_h", "split", "sched_arrivals", "actual_arrivals",
       "high_demand_hour", "storm_hour", "pred", "pred_lo", "pred_hi", "base_persistence", "base_climatology"]
      + [c for c in a.columns if c.startswith(("obs_", "taf_"))]].to_parquet(PROC / "aar_predictions.parquet", index=False)
    return rep, m.best_iteration


def leave_one_airport_out(a):
    """Generalisation check: train on all other airports, predict the held-out one.
    Target is arrivals_norm (arrivals / airport's own p95) so airports of different size share a scale."""
    if a.airport.nunique() < 2:
        return pd.DataFrame()
    drop = AAR_NOT_FEATURES | {"airport", "sched_arrivals", "sched_arrivals_next1h", "st_arr_2h"}
    a = a.copy()
    rows = []
    for held in sorted(a.airport.unique()):
        p95 = a[(a.airport == held) & (a.split == "train")].actual_arrivals.quantile(0.95)
        for d in (a,):
            # schedule features rescaled per airport so counts do not reveal airport identity
            for c in ("sched_arrivals", "sched_arrivals_next1h", "st_arr_2h"):
                if c in d:
                    d[c + "_norm"] = d[c] / d.groupby("airport").actual_arrivals.transform(lambda x: x.quantile(0.95))
        tr = a[(a.airport != held) & (a.split == "train")]
        te = a[(a.airport == held) & (a.split == "test")]
        Xtr = prep(tr, drop)
        Xte = align_categories(prep(te, drop), Xtr)
        m = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=7),
                      lgb.Dataset(Xtr, tr.arrivals_norm), 400)
        pred = m.predict(Xte)
        own = a[(a.airport == held) & (a.split == "train")]
        clim = own.groupby(["target_hour_local", "dow"]).arrivals_norm.mean()
        base = te.join(clim.rename("c"), on=["target_hour_local", "dow"]).c.fillna(own.arrivals_norm.mean())
        for subset, mask in (("all_hours", np.ones(len(te), bool)), ("high_demand", (te.high_demand_hour == 1).values)):
            rows.append({"held_out_airport": held, "subset": subset, "n": int(mask.sum()),
                         "nrmse_unseen_airport_model": rmse(te.arrivals_norm[mask], pred[mask]),
                         "nrmse_own_climatology": rmse(te.arrivals_norm[mask], base[mask]),
                         "rmse_arrivals_equiv": rmse(te.arrivals_norm[mask], pred[mask]) * p95})
    rep = pd.DataFrame(rows)
    rep.to_csv(REPORTS / "aar_leave_one_airport_out.csv", index=False)
    return rep


# ---------------------------------------------------------------- severe delay classifier
def train_severe(f):
    f = f.copy()
    f["event_hour_local"] = -1
    import airportsdata  # noqa: F401  (tz already in airports.json for focus airports)
    from common import load_airports
    tz = {k: v["tz"] for k, v in load_airports(sorted(f.airport.unique())).items()}
    for ap, idx in f.groupby("airport").groups.items():
        f.loc[idx, "event_hour_local"] = f.loc[idx, "event_utc"].dt.tz_localize("UTC").dt.tz_convert(tz[ap]).dt.hour
    f["dow"] = f.event_utc.dt.dayofweek
    f["month"] = f.event_utc.dt.month
    keys = ["airport", "direction", "event_hour_local"]
    clim = f[f.split == "train"].groupby(keys).y_severe.mean().rename("base_climatology")
    f = f.join(clim, on=keys)
    f["base_climatology"] = f.base_climatology.fillna(f[f.split == "train"].y_severe.mean())

    drop = FL_NOT_FEATURES | {"base_climatology"}
    tr, va, te = (f[f.split == s] for s in ("train", "val", "test"))
    Xtr = prep(tr, drop)
    Xva, Xte = align_categories(prep(va, drop), Xtr), align_categories(prep(te, drop), Xtr)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    m = lgb.train(params, lgb.Dataset(Xtr, tr.y_severe), 2000, valid_sets=[lgb.Dataset(Xva, va.y_severe)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m.save_model(MODELS / "severe_lgbm.txt")
    f.loc[va.index, "p_severe"] = m.predict(Xva, num_iteration=m.best_iteration)
    f.loc[te.index, "p_severe"] = m.predict(Xte, num_iteration=m.best_iteration)

    v = f.loc[va.index]
    grid = np.linspace(0.02, 0.9, 89)
    thr = float(grid[np.argmax([f1_score(v.y_severe, v.p_severe >= g, zero_division=0) for g in grid])])

    f = add_storm_flag(f, "event_utc")
    rows = []
    t = f[f.split == "test"]
    for (ap, d, H), g0 in t.groupby(["airport", "direction", "horizon_h"]):
      for subset, g in (("all_hours", g0), ("storm_hours", g0[g0.storm_hour == 1])):
        if g.y_severe.nunique() < 2:
            continue
        yhat = g.p_severe >= thr
        tp, fp = int((yhat & (g.y_severe == 1)).sum()), int((yhat & (g.y_severe == 0)).sum())
        fn, tn = int((~yhat & (g.y_severe == 1)).sum()), int((~yhat & (g.y_severe == 0)).sum())
        rows.append({"airport": ap, "direction": d, "horizon_h": H, "subset": subset, "n": len(g),
                     "base_rate": g.y_severe.mean(),
                     "pr_auc_model": average_precision_score(g.y_severe, g.p_severe),
                     "pr_auc_climatology": average_precision_score(g.y_severe, g.base_climatology),
                     "roc_auc_model": roc_auc_score(g.y_severe, g.p_severe),
                     "brier_model": brier_score_loss(g.y_severe, g.p_severe),
                     "brier_climatology": brier_score_loss(g.y_severe, g.base_climatology),
                     "threshold": thr, "precision": precision_score(g.y_severe, yhat, zero_division=0),
                     "recall": recall_score(g.y_severe, yhat, zero_division=0),
                     "f1": f1_score(g.y_severe, yhat, zero_division=0), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    rep = pd.DataFrame(rows)
    rep.to_csv(REPORTS / "severe_metrics.csv", index=False)
    # reliability table for the calibration plot (test, all horizons)
    t = f.loc[te.index]
    bins = pd.cut(t.p_severe, np.linspace(0, 1, 11), include_lowest=True)
    t.groupby(bins, observed=True).agg(mean_pred=("p_severe", "mean"), obs_rate=("y_severe", "mean"),
                                       n=("y_severe", "size")).to_csv(REPORTS / "severe_calibration.csv")
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    imp.to_csv(REPORTS / "severe_feature_importance.csv", header=["gain"])
    f[["airport", "direction", "event_utc", "t0_utc", "horizon_h", "split", "OP_UNIQUE_CARRIER",
       "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "y_severe", "p_severe", "base_climatology"]
      ].to_parquet(PROC / "severe_predictions.parquet", index=False)
    return rep, thr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    aar_rep, aar_iter = train_aar(pd.read_parquet(PROC / "aar_horizon.parquet"))
    loao = leave_one_airport_out(pd.read_parquet(PROC / "aar_horizon.parquet"))
    sev_rep, thr = train_severe(pd.read_parquet(PROC / "flights_horizon.parquet"))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\nAAR proxy, test 2025 (RMSE in arrivals/hour):")
    print(aar_rep[["airport", "horizon_h", "subset", "n", "rmse_model", "rmse_persistence", "rmse_climatology",
                   "rmse_schedule", "interval_coverage", "model_beats_climatology"]].round(3).to_string(index=False))
    if len(loao):
        print("\nLeave-one-airport-out (model never saw the held-out airport; normalised RMSE):")
        print(loao.round(3).to_string(index=False))
    print(f"\nSevere delay (>45 min / cxl / div), test 2025, threshold {thr:.2f} chosen on validation:")
    print(sev_rep[["airport", "direction", "horizon_h", "subset", "n", "base_rate", "pr_auc_model", "pr_auc_climatology",
                   "brier_model", "brier_climatology", "precision", "recall", "f1"]].round(3).to_string(index=False))
    (REPORTS / "run_info.json").write_text(json.dumps({"aar_best_iter": aar_iter, "severe_threshold": thr,
                                                       "coverage_target": COVERAGE}, indent=2))
