"""Severe-delay (> 45 min) classifier, one model per forecast lead (3..12 h before the scheduled event).

Rows are hub events: a departure from, or an arrival at, a configured hub (ATL, DFW).

Protocol (everything time-ordered, nothing shuffled):
  * the model FAMILY and its hyper-parameters are chosen by out-of-fold PR-AUC over the four quarters of 2024
    (expanding window), at a near and a far lead.  Candidates: L2-regularised logistic regression and LightGBM.
    The data are small (about 1,500 severe events) and the signal weak and mostly additive, so the regularised
    linear model wins; the selection table is saved so this is visible, not assumed.
  * calibration: Platt scaling fitted on those out-of-fold predictions
  * final model: refit on all data before 2025-01-01
  * test: 2025-01..2025-07, touched once
Ablation ladder (does each information source earn its place?):
  A schedule only | B + recent hub performance | C + METAR weather | D + TAF forecast (full model)
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

CATS = ["airport", "direction", "carrier", "other_airport"]
SCHED = ["hour_local", "dow", "month", "is_weekend", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
         "distance_mi", "sched_elapsed_min", "sched_load_sample", "to_other_hub"]
NOT_FEATURES = {"flight_id", "event_utc", "event_hour_utc", "delay_min", "severe_delay", "lead_h", "issue_utc",
                "obs_hour_utc", "split", "hour_utc"}
TUNE_LEADS = (3, 12)
FOLD_STARTS = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01"]

# candidate models: name -> (family, params)
CANDIDATES = {f"logit C={c}": ("logit", {"C": c}) for c in (0.001, 0.003, 0.01, 0.03)}
for _gi, _p in enumerate([dict(num_leaves=7, min_child_samples=80, learning_rate=0.03, feature_fraction=0.7, lambda_l2=10.0),
                          dict(num_leaves=15, min_child_samples=100, learning_rate=0.03, feature_fraction=0.7, lambda_l2=10.0)]):
    for _r in (200, 350):
        CANDIDATES[f"lightgbm cfg{_gi} rounds={_r}"] = ("lightgbm", {**_p, "rounds": _r})


def groups(t):
    cols = set(t.columns) - NOT_FEATURES
    rec = sorted(c for c in cols if c.startswith("rec_"))
    taf = sorted(c for c in cols if c.startswith("taf_"))
    obs = sorted(cols - set(rec) - set(taf) - set(SCHED) - set(CATS))
    return {"A_schedule": CATS + SCHED, "B_plus_recent_hub": CATS + SCHED + rec,
            "C_plus_metar": CATS + SCHED + rec + obs, "D_plus_taf": CATS + SCHED + rec + obs + taf}


def prep(t):
    """Group rare carriers / routes into OTHER (counted on the pre-test years only)."""
    t = t.copy()
    for c in ("carrier", "other_airport"):
        vc = t.loc[t["split"] != "test", c].value_counts()
        keep = set(vc[vc >= 40].index)
        t[c] = t[c].where(t[c].isin(keep), "OTHER")
    for c in CATS:
        t[c] = pd.Categorical(t[c], categories=sorted(t.loc[t["split"] != "test", c].unique()))
    return t


# ------------------------------------------------------------------------------------------ model families
def _split_cols(feats):
    cats = [c for c in feats if c in CATS]
    return cats, [c for c in feats if c not in cats]


def _frame(d, feats):
    cats, nums = _split_cols(feats)
    return d[cats + nums].astype({c: str for c in cats})


def fit_model(d, feats, family, params):
    """Returns a bundle dict that predict_bundle / explain_bundle understand."""
    y = d["severe_delay"].values
    if family == "logit":
        cats, nums = _split_cols(feats)
        pipe = Pipeline([("p", ColumnTransformer([
            ("c", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=40), cats),
            ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), nums)])),
            ("m", LogisticRegression(C=params["C"], max_iter=3000))])
        pipe.fit(_frame(d, feats), y)
        Xt = pipe.named_steps["p"].transform(_frame(d, feats))
        Xt = Xt.toarray() if sparse.issparse(Xt) else np.asarray(Xt)
        return {"family": "logit", "model": pipe, "feats": feats, "mu": Xt.mean(0),
                "names": list(pipe.named_steps["p"].get_feature_names_out())}
    p = {k: v for k, v in params.items() if k != "rounds"}
    booster = lgb.train(dict(objective="binary", verbosity=-1, seed=0, bagging_fraction=0.8, bagging_freq=1, **p),
                        lgb.Dataset(d[feats], y), params["rounds"])
    return {"family": "lightgbm", "model": booster, "feats": feats}


def predict_bundle(b, d):
    if b["family"] == "logit":
        return b["model"].predict_proba(_frame(d, b["feats"]))[:, 1]
    return b["model"].predict(d[b["feats"]])


def explain_bundle(b, d):
    """Per-feature contribution to the log-odds, relative to the average training flight. Rows = flights,
    columns = original features.  Exact for logistic regression (coefficient x centred value); TreeSHAP for LightGBM."""
    feats = b["feats"]
    if b["family"] == "lightgbm":
        pc = b["model"].predict(d[feats], pred_contrib=True)[:, :-1]
        return pd.DataFrame(pc, columns=feats, index=d.index)
    pipe = b["model"]
    Xt = pipe.named_steps["p"].transform(_frame(d, feats))
    Xt = Xt.toarray() if sparse.issparse(Xt) else np.asarray(Xt)
    contrib = (Xt - b["mu"]) * pipe.named_steps["m"].coef_[0]
    cats, _ = _split_cols(feats)
    src = []
    for n in b["names"]:
        if n.startswith("c__"):
            src.append(next(c for c in cats if n.startswith(f"c__{c}_")))
        else:
            src.append(n[3:])
    out = pd.DataFrame(contrib, columns=b["names"], index=d.index).T.groupby(src).sum().T
    return out.reindex(columns=[f for f in feats if f in out.columns])


def oof(d, feats, family, params):
    """Expanding-window out-of-fold predictions for the four quarters of 2024."""
    preds, idx = [], []
    for a, b in zip(FOLD_STARTS[:-1], FOLD_STARTS[1:]):
        tr = d[d["event_utc"] < pd.Timestamp(a, tz="UTC")]
        te = d[(d["event_utc"] >= pd.Timestamp(a, tz="UTC")) & (d["event_utc"] < pd.Timestamp(b, tz="UTC"))]
        preds.append(predict_bundle(fit_model(tr, feats, family, params), te))
        idx.append(te.index)
    return np.concatenate(preds), np.concatenate(idx)


def platt(p_oof, y):
    z = np.log(np.clip(p_oof, 1e-6, 1 - 1e-6) / (1 - np.clip(p_oof, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p, a, b):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(a * z + b)))


def metrics(y, p):
    top = p >= np.quantile(p, 0.9)
    out = {"n": len(y), "prevalence": float(y.mean()), "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
           "brier": brier_score_loss(y, np.clip(p, 0, 1)), "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
           "precision_top10pct": float(y[top].mean()), "recall_top10pct": float(y[top].sum() / y.sum())}
    for thr in (0.2, 0.3):
        f = p >= thr
        out[f"precision_at_{thr}"] = float(y[f].mean()) if f.any() else np.nan
        out[f"recall_at_{thr}"] = float(y[f].sum() / y.sum())
        out[f"flag_rate_at_{thr}"] = float(f.mean())
    return out


def main():
    t = prep(pd.read_parquet(C.PROC / "delay_table.parquet"))
    G = groups(t)
    C.MODELS.mkdir(exist_ok=True, parents=True)
    C.REPORTS.mkdir(exist_ok=True, parents=True)
    json.dump(G["D_plus_taf"], open(C.MODELS / "delay_features.json", "w"))
    print({k: len(v) for k, v in G.items()})
    pre = t[t["split"] != "test"]

    # ---- 1. choose the model family + hyper-parameters by out-of-fold PR-AUC (2024), full feature set -----------
    sel = []
    for name, (fam, params) in CANDIDATES.items():
        scores = {}
        for lead in TUNE_LEADS:
            d = pre[pre.lead_h == lead]
            p, idx = oof(d, G["D_plus_taf"], fam, params)
            scores[lead] = average_precision_score(d.loc[idx, "severe_delay"].values, p)
        sel.append({"candidate": name, "family": fam, **{f"oof_pr_auc_lead{l}": v for l, v in scores.items()},
                    "oof_pr_auc_mean": float(np.mean(list(scores.values())))})
    sel = pd.DataFrame(sel).sort_values("oof_pr_auc_mean", ascending=False)
    sel.to_csv(C.REPORTS / "delay_model_selection.csv", index=False)
    print(sel.round(4).to_string(index=False))
    best_name = sel.iloc[0]["candidate"]
    fam, params = CANDIDATES[best_name]
    # best LightGBM candidate, kept as a comparison row
    lg_name = sel[sel.family == "lightgbm"].iloc[0]["candidate"]
    lg_fam, lg_params = CANDIDATES[lg_name]
    print("chosen:", best_name, "| best LightGBM for comparison:", lg_name)
    json.dump({"candidate": best_name, "family": fam, "params": params}, open(C.MODELS / "delay_params.json", "w"))

    rows, preds, calib = [], [], {}
    for lead in sorted(t.lead_h.unique()):
        d_all = t[t.lead_h == lead]
        d_pre, d_te = d_all[d_all.split != "test"], d_all[d_all.split == "test"]
        yte = d_te["severe_delay"].values
        for gname, feats in G.items():
            if gname != "D_plus_taf" and lead not in TUNE_LEADS:
                continue                                                  # ablations only at the two reference leads
            p_oof, idx = oof(d_pre, feats, fam, params)
            y_oof = d_pre.loc[idx, "severe_delay"].values
            a, b = platt(p_oof, y_oof)
            bundle = fit_model(d_pre, feats, fam, params)
            raw = predict_bundle(bundle, d_te)
            cal = apply_platt(raw, a, b)
            rows.append({"lead_h": lead, "model": gname, "variant": "raw", "scope": "all", **metrics(yte, raw)})
            rows.append({"lead_h": lead, "model": gname, "variant": "calibrated", "scope": "all", **metrics(yte, cal)})
            if gname == "D_plus_taf":
                joblib.dump({**bundle, "platt": {"a": a, "b": b}}, C.MODELS / f"delay_lead{lead}.joblib")
                calib[lead] = {"a": a, "b": b}
                for scope_col in ("direction", "airport"):
                    for v in sorted(d_te[scope_col].astype(str).unique()):
                        mk = (d_te[scope_col].astype(str) == v).values
                        rows.append({"lead_h": lead, "model": gname, "variant": "calibrated", "scope": f"{scope_col}={v}", **metrics(yte[mk], cal[mk])})
                preds.append(d_te[["flight_id", "airport", "direction", "other_airport", "carrier", "event_utc", "lead_h", "delay_min",
                                   "severe_delay"]].assign(p_raw=raw, p=cal))
                if lead in TUNE_LEADS:
                    imp = explain_bundle(bundle, d_te).abs().mean().sort_values(ascending=False)
                    imp.to_csv(C.REPORTS / f"delay_driver_importance_lead{lead}.csv", header=["mean_abs_contribution_logodds"])
        # comparison model (best LightGBM, full features) and trivial baseline
        lg = fit_model(d_pre, G["D_plus_taf"], lg_fam, lg_params)
        rows.append({"lead_h": lead, "model": "lightgbm_comparison", "variant": "raw", "scope": "all", **metrics(yte, predict_bundle(lg, d_te))})
        rows.append({"lead_h": lead, "model": "constant_base_rate", "variant": "raw", "scope": "all",
                     **metrics(yte, np.full(len(yte), d_pre["severe_delay"].mean()))})
        full = [r for r in rows if r["lead_h"] == lead and r["model"] == "D_plus_taf" and r["variant"] == "calibrated" and r["scope"] == "all"][0]
        cmp_ = [r for r in rows if r["lead_h"] == lead and r["model"] == "lightgbm_comparison"][0]
        print(f"lead {lead:2d}h  chosen PR-AUC {full['pr_auc']:.3f}  ROC {full['roc_auc']:.3f}  brier {full['brier']:.4f}  | LightGBM PR-AUC {cmp_['pr_auc']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "delay_metrics.csv", index=False)
    json.dump({str(k): v for k, v in calib.items()}, open(C.MODELS / "delay_calibration.json", "w"))
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(C.PROC / "delay_test_predictions.parquet", index=False)

    # ---- 2. uncertainty on the headline comparison: day-block bootstrap of the PR-AUC gain over schedule-only -------
    boot = []
    rng = np.random.default_rng(0)
    for lead in TUNE_LEADS:
        d = t[(t.lead_h == lead) & (t.split == "test")].reset_index(drop=True)
        dp = t[(t.lead_h == lead) & (t.split != "test")]
        pr = {}
        for gname in ("A_schedule", "D_plus_taf"):
            feats = G[gname]
            p_oof, idx = oof(dp, feats, fam, params)
            a, b = platt(p_oof, dp.loc[idx, "severe_delay"].values)
            pr[gname] = apply_platt(predict_bundle(fit_model(dp, feats, fam, params), d), a, b)
        day = d["event_utc"].dt.floor("D").astype(str).values
        days = np.unique(day)
        by_day = {x: np.where(day == x)[0] for x in days}
        gains = []
        for _ in range(300):
            sel_idx = np.concatenate([by_day[x] for x in rng.choice(days, len(days))])
            y = d["severe_delay"].values[sel_idx]
            gains.append(average_precision_score(y, pr["D_plus_taf"][sel_idx]) - average_precision_score(y, pr["A_schedule"][sel_idx]))
        boot.append({"lead_h": lead, "pr_auc_gain_mean": float(np.mean(gains)), "ci_low": float(np.percentile(gains, 2.5)),
                     "ci_high": float(np.percentile(gains, 97.5))})
    pd.DataFrame(boot).to_csv(C.REPORTS / "delay_bootstrap.csv", index=False)

    # ---- 3. reliability table, daily-level agreement, failure cases (lead 3 h and 12 h) ------------------------------
    for lead in TUNE_LEADS:
        q = P[P.lead_h == lead].copy()
        q["bin"] = pd.qcut(q["p"], 10, duplicates="drop")
        q.groupby("bin", observed=True).agg(n=("p", "size"), mean_pred=("p", "mean"), observed=("severe_delay", "mean")).round(4).to_csv(
            C.REPORTS / f"delay_reliability_lead{lead}.csv")
        q["day"] = q["event_utc"].dt.floor("D")
        dd = q.groupby(["airport", "day"]).agg(pred=("p", "mean"), obs=("severe_delay", "mean"), n=("p", "size")).reset_index()
        dd = dd[dd.n >= 5]
        rho = spearmanr(dd["pred"], dd["obs"])
        pd.DataFrame([{"lead_h": lead, "days": len(dd), "spearman_daily_pred_vs_observed": rho.statistic, "p_value": rho.pvalue}]).to_csv(
            C.REPORTS / f"delay_daily_agreement_lead{lead}.csv", index=False)
        thr = q["p"].quantile(0.5)
        miss = q[(q.severe_delay == 1) & (q.p <= thr)].sort_values("delay_min", ascending=False).head(15)
        fa = q[(q.severe_delay == 0)].sort_values("p", ascending=False).head(15)
        pd.concat([miss.assign(kind="missed_severe_delay"), fa.assign(kind="false_alarm")]).to_csv(
            C.REPORTS / f"delay_failure_cases_lead{lead}.csv", index=False)

    pd.set_option("display.width", 220)
    r = res[(res.scope == "all") & ((res.variant == "calibrated") | res.model.isin(["constant_base_rate", "lightgbm_comparison"]))]
    print("\nTEST PR-AUC (prevalence in brackets)\n", r.pivot_table(index="model", columns="lead_h", values="pr_auc").round(3))
    print("\nTEST ROC-AUC\n", r.pivot_table(index="model", columns="lead_h", values="roc_auc").round(3))
    print("\nbootstrap PR-AUC gain (D - A)\n", pd.DataFrame(boot).round(4))


if __name__ == "__main__":
    main()
