"""ML — train + score from the warehouse, write Gold outputs.

  1. Decline risk  — P(next-quarter dispatch >50% below trailing 4-q avg)
                     logistic (calibrated) + OPTIONAL LightGBM challenger
  2. Forecast      — champion of {ridge, seasonal-naive, blend, croston-TSB} by CV WAPE,
                     with P10/P50/P90 prediction intervals
  3. Opportunity   — size x share-headroom x growth (latest complete FY)
  4. Concentration — HHI + top-5 share per FY

Validation is ROLLING-ORIGIN (train on the past, test the next quarter, repeated).
Leakage fix: seasonal indices and weather climatology fallbacks are computed on the
TRAINING slice only — never over the validation/scoring quarters.
"""
import json
import os

import numpy as np
import pandas as pd

from utils import (GOLD, Q_NUM, QTY_COLS, LogisticModel, RidgeModel, auc_score,
                   banner, current_complete_quarter, wh_connect)
from mlx import (precision_recall_f1, PlattCalibrator, croston_tsb, residual_interval,
                 rolling_origin_folds, fit_lgbm_classifier)

WX_COLS = ["rain_mm", "monsoon_idx", "rain_lag1", "rain_lag2",
           "monsoon_cum", "extreme_wet", "dry_spell"]

FEATURES = [
    "mill_dispatched_MT", "disp_lag1", "disp_lag2", "disp_lag3", "disp_lag4",
    "disp_roll4_mean", "disp_roll4_std", "disp_qoq", "disp_slope4",
    "purch_roll4_mean", "offtake_roll4_mean", "share", "share_roll4",
    "q_since_active", "qnum", "region_seasonal_idx", "next_seasonal_idx",
    "rain_mm", "monsoon_idx", "rain_next_q",
    "rain_lag1", "rain_lag2", "monsoon_cum", "extreme_wet", "dry_spell",
]


# ---------------------------------------------------------------- grid & base features
def dense_grid(panel):
    panel = panel.groupby(["supplier", "qidx", "fiscal_year", "fiscal_quarter"], as_index=False)[QTY_COLS].sum()
    qmax = int(panel["qidx"].max())
    frames = [pd.DataFrame({"supplier": sup, "qidx": np.arange(int(g["qidx"].min()), qmax + 1)})
              for sup, g in panel.groupby("supplier")]
    grid = pd.concat(frames, ignore_index=True).merge(panel, on=["supplier", "qidx"], how="left")
    for c in QTY_COLS:
        grid[c] = grid[c].fillna(0.0)
    grid["fiscal_year"] = ((grid["qidx"] - 1) // 4).astype(int)
    grid["qnum"] = grid["qidx"] - grid["fiscal_year"] * 4
    grid["fiscal_quarter"] = "FQ" + grid["qnum"].astype(str)
    return grid.sort_values(["supplier", "qidx"]).reset_index(drop=True)


def build_base_features(grid, region_map, weather):
    """Per-row features that use only each supplier's own past — no cross-row leakage.
    Seasonal indices and weather climatology fallbacks are added later, on train only."""
    g = grid.copy()
    g["region"] = g["supplier"].map(region_map).fillna("UNKNOWN")
    grp = g.groupby("supplier")
    d = "mill_dispatched_MT"
    for lag in (1, 2, 3, 4):
        g[f"disp_lag{lag}"] = grp[d].shift(lag)
    g["disp_roll4_mean"] = grp[d].transform(lambda s: s.rolling(4, min_periods=2).mean())
    g["disp_roll4_std"] = grp[d].transform(lambda s: s.rolling(4, min_periods=2).std())
    g["purch_roll4_mean"] = grp["vaighai_purchased_MT"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    g["offtake_roll4_mean"] = grp["vaighai_offtake_est_MT"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    g["disp_qoq"] = np.where(g["disp_lag1"] > 0, (g[d] - g["disp_lag1"]) / g["disp_lag1"], 0.0)
    g["disp_slope4"] = (g[d] - g["disp_lag3"]) / 3.0
    active = g[d] > 0
    g["_c"] = active.groupby(g["supplier"]).cumsum()
    g["q_since_active"] = g.groupby(["supplier", "_c"]).cumcount()
    g = g.drop(columns="_c")
    g["share"] = np.where(g[d] > 0, (g["vaighai_offtake_est_MT"] / g[d] * 100).clip(0, 100), 0.0)
    g["share_roll4"] = g.groupby("supplier")["share"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    g["next_fq"] = "FQ" + (g["qnum"] % 4 + 1).astype(str)
    # weather ACTUALS for the exact region-quarter (NaN where region unmapped -> filled on train)
    keep = ["region", "fiscal_year", "fiscal_quarter"] + [c for c in WX_COLS if c in weather.columns]
    g = g.merge(weather[keep], on=["region", "fiscal_year", "fiscal_quarter"], how="left")
    return g


# ---------------------------------------------------------------- fit-dependent features (train-only)
def _seasonal_indices(train_df):
    d = "mill_dispatched_MT"
    reg_q = train_df.groupby(["region", "fiscal_quarter"])[d].mean()
    reg_m = train_df.groupby("region")[d].mean()
    seas = (reg_q / reg_m).rename("region_seasonal_idx").reset_index()
    return seas


def add_fit_features(target, source):
    """Add seasonal indices + fill weather climatology, all derived from `source`
    (the training slice) and applied to `target`. Prevents validation/scoring leakage."""
    g = target.copy()
    seas = _seasonal_indices(source)
    g = g.drop(columns=[c for c in ("region_seasonal_idx", "next_seasonal_idx") if c in g], errors="ignore")
    g = g.merge(seas, on=["region", "fiscal_quarter"], how="left")
    g["region_seasonal_idx"] = g["region_seasonal_idx"].fillna(1.0)
    g = g.merge(seas.rename(columns={"fiscal_quarter": "next_fq",
                                     "region_seasonal_idx": "next_seasonal_idx"}),
                on=["region", "next_fq"], how="left")
    g["next_seasonal_idx"] = g["next_seasonal_idx"].fillna(1.0)
    # weather climatology fallback from TRAIN: region-quarter mean, then overall mean
    wx_present = [c for c in WX_COLS if c in g.columns]
    clim = source.groupby(["region", "fiscal_quarter"])[wx_present].mean().reset_index()
    overall = {c: float(source[c].mean()) if source[c].notna().any() else 0.0 for c in wx_present}
    g = g.merge(clim, on=["region", "fiscal_quarter"], how="left", suffixes=("", "_clim"))
    for c in wx_present:
        g[c] = g[c].fillna(g.get(c + "_clim")).fillna(overall[c])
    g = g.drop(columns=[c + "_clim" for c in wx_present if c + "_clim" in g.columns], errors="ignore")
    return g


def add_rain_next_q(g, weather, weather_next, latest_q):
    """rain of the TARGET quarter (t+1): actuals for training rows, forecast estimate for
    scoring rows. Climatology fallback uses only weather up to latest_q (no future leak)."""
    w = weather.copy()
    w["qidx"] = w["fiscal_year"] * 4 + w["fiscal_quarter"].map(Q_NUM)
    nxt = w[["region", "qidx", "rain_mm"]].copy()
    nxt["qidx"] -= 1
    nxt = nxt.rename(columns={"rain_mm": "rain_next_q"})
    g = g.merge(nxt, on=["region", "qidx"], how="left")
    est = dict(zip(weather_next["region"], weather_next["rain_mm_est"]))
    scoring = g["qidx"] == latest_q
    g.loc[scoring, "rain_next_q"] = g.loc[scoring, "region"].map(est)
    w_train = w[w["qidx"] <= latest_q]
    clim_next = (w_train.groupby(["region", "fiscal_quarter"])["rain_mm"].mean()
                 .rename("rain_next_clim").reset_index()
                 .rename(columns={"fiscal_quarter": "next_fq"}))
    g = g.merge(clim_next, on=["region", "next_fq"], how="left")
    g["rain_next_q"] = g["rain_next_q"].fillna(g["rain_next_clim"]).fillna(0.0)
    return g.drop(columns="rain_next_clim", errors="ignore")


# ---------------------------------------------------------------- croston helper
def _croston_map(grid):
    m = {}
    for sup, gg in grid.sort_values("qidx").groupby("supplier"):
        m[sup] = (gg["qidx"].values, gg["mill_dispatched_MT"].values)
    return m


def _croston_for(rows, cmap):
    out = []
    for sup, q in zip(rows["supplier"].values, rows["qidx"].values):
        arr = cmap.get(sup)
        if arr is None:
            out.append(0.0); continue
        qi, dv = arr
        out.append(croston_tsb(dv[qi < q]))
    return np.array(out, dtype=float)


def _wape(pred, actual):
    return float(np.abs(np.asarray(pred) - np.asarray(actual)).sum() / max(np.asarray(actual).sum(), 1) * 100)


# ---------------------------------------------------------------- main
def run():
    banner("ML", "Rolling-origin CV · calibration · Croston+quantiles · optional LightGBM")
    con, backend = wh_connect()
    panel = pd.read_sql("SELECT * FROM supply_panel", con)
    weather = pd.read_sql("SELECT * FROM weather_quarterly", con)
    weather_next = pd.read_sql("SELECT * FROM weather_next_quarter", con)
    if backend == "sqlite":
        con.close()
    panel["qidx"] = panel["fiscal_year"] * 4 + panel["fiscal_quarter"].map(Q_NUM)
    region_map = (panel[panel["region"] != "UNKNOWN"].groupby("supplier")["region"]
                  .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "UNKNOWN"))

    cutoff = current_complete_quarter()
    grid = dense_grid(panel[(panel["fiscal_year"] >= 2021) & (panel["qidx"] <= cutoff)])
    cmap = _croston_map(grid)
    base = build_base_features(grid, region_map, weather)
    latest_q = min(int(base.loc[base["mill_dispatched_MT"] > 0, "qidx"].max()), cutoff)
    base = add_rain_next_q(base, weather, weather_next, latest_q)
    latest_fy, latest_qn = (latest_q - 1) // 4, latest_q - ((latest_q - 1) // 4) * 4
    latest_complete_fy = latest_fy - 1 if latest_qn < 4 else latest_fy

    grp = base.groupby("supplier")
    base["disp_next"] = grp["mill_dispatched_MT"].shift(-1)
    base["declined_next"] = (base["disp_next"] < 0.5 * base["disp_roll4_mean"]).astype(int)
    eligible = (base["disp_roll4_mean"] > 0) & base["disp_lag1"].notna()
    pool = base[eligible & base["disp_next"].notna() & (base["qidx"] < latest_q)].copy()

    # ---- rolling-origin folds (train on the past, test the next quarter)
    quarters = sorted(pool["qidx"].unique())
    test_qs = rolling_origin_folds(quarters, k=6, min_train_quarters=4)
    folds = []
    for q in test_qs:
        tr = pool[pool["qidx"] < q]
        va = pool[pool["qidx"] == q]
        if len(tr) < 30 or len(va) < 5 or va["declined_next"].nunique() < 2:
            continue
        folds.append((q, add_fit_features(tr, tr), add_fit_features(va, tr)))
    if not folds:  # too few quarters — single-split fallback
        cut = pool["qidx"].max() - 4
        tr, va = pool[pool["qidx"] <= cut], pool[pool["qidx"] > cut]
        folds = [(int(va["qidx"].min()), add_fit_features(tr, tr), add_fit_features(va, tr))]

    # ---- decline classifier: tune (l2, lr) on pooled CV AUC
    best = None
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1):
        for lr in (0.15, 0.3):
            ys, ps = [], []
            for _, trF, vaF in folds:
                clf = LogisticModel().fit(trF[FEATURES].values, trF["declined_next"].values, lr=lr, l2=l2)
                ps += list(clf.predict_proba(vaF[FEATURES].values)); ys += list(vaF["declined_next"].values)
            a = auc_score(ys, ps)
            if best is None or a > best[1]:
                best = ((l2, lr), a)
    (bl2, blr), _ = best

    # pooled out-of-fold predictions for the best config (+ optional LightGBM challenger)
    pool_y, pool_lr, pool_gb = [], [], []
    gb_ok = True
    for _, trF, vaF in folds:
        clf = LogisticModel().fit(trF[FEATURES].values, trF["declined_next"].values, lr=blr, l2=bl2)
        pool_lr += list(clf.predict_proba(vaF[FEATURES].values))
        pool_y += list(vaF["declined_next"].values)
        if gb_ok:
            gb = fit_lgbm_classifier(trF[FEATURES].values, trF["declined_next"].values)
            if gb is None:
                gb_ok = False
            else:
                pool_gb += list(gb.predict_proba(vaF[FEATURES].values))
    auc_lr = auc_score(pool_y, pool_lr)
    auc_gb = auc_score(pool_y, pool_gb) if (gb_ok and len(pool_gb) == len(pool_y)) else None
    use_gb = bool(auc_gb is not None and auc_gb > auc_lr)
    scorer_name = "lightgbm" if use_gb else "logistic"
    pooled_scores = pool_gb if use_gb else pool_lr
    auc = auc_gb if use_gb else auc_lr
    calib = PlattCalibrator().fit(pooled_scores, pool_y)
    prf = precision_recall_f1(pool_y, calib.transform(pooled_scores), thr=0.5)
    print(f"  decline: logistic AUC {auc_lr:.3f}"
          + (f" | lightgbm AUC {auc_gb:.3f}" if auc_gb is not None else " | lightgbm n/a")
          + f" -> scoring with {scorer_name}; P/R/F1 {prf['precision']}/{prf['recall']}/{prf['f1']}")

    # ---- forecast champion across the same folds (ridge / seasonal-naive / blend / croston)
    wape_acc = {k: [] for k in ("ridge", "seasonal_naive", "blend", "croston")}
    champ_rel_err = {k: [] for k in wape_acc}
    for q, trF, vaF in folds:
        reg = RidgeModel().fit(trF[FEATURES].values, np.log1p(trF["disp_next"].values), l2=1.0)
        ml = np.clip(np.expm1(reg.predict(vaF[FEATURES].values)), 0, None)
        naive = (vaF["disp_roll4_mean"] * vaF["next_seasonal_idx"]).clip(lower=0).values
        cros = _croston_for(vaF, cmap)
        y = vaF["disp_next"].values
        preds = {"ridge": ml, "seasonal_naive": naive, "blend": 0.5 * ml + 0.5 * naive, "croston": cros}
        for k, p in preds.items():
            wape_acc[k].append(_wape(p, y))
            champ_rel_err[k] += list((y - p) / np.clip(p, 1e-6, None))
    wapes = {k: float(np.mean(v)) for k, v in wape_acc.items() if v}
    champion = min(wapes, key=wapes.get)
    print("  forecast champion: " + champion + " (" + ", ".join(f"{k} {v:.1f}%" for k, v in wapes.items()) + ")")

    # ---- refit on the full training universe, score the latest quarter
    finalTr = add_fit_features(pool, pool)
    clf_final = LogisticModel().fit(finalTr[FEATURES].values, finalTr["declined_next"].values, lr=blr, l2=bl2)
    reg_final = RidgeModel().fit(finalTr[FEATURES].values, np.log1p(finalTr["disp_next"].values), l2=1.0)
    scorer = fit_lgbm_classifier(finalTr[FEATURES].values, finalTr["declined_next"].values) if use_gb else clf_final

    watch = base[eligible & (base["qidx"] == latest_q)].copy()
    watchF = add_fit_features(watch, pool)
    watch["decline_risk"] = calib.transform(scorer.predict_proba(watchF[FEATURES].values))
    watch["risk_band"] = pd.cut(watch["decline_risk"], [0, 0.4, 0.7, 1.0],
                                labels=["Low", "Moderate", "Critical"]).astype(str)
    ml = np.clip(np.expm1(reg_final.predict(watchF[FEATURES].values)), 0, None)
    naive = (watchF["disp_roll4_mean"] * watchF["next_seasonal_idx"]).clip(lower=0).values
    cros = _croston_for(watchF, cmap)
    point = {"ridge": ml, "seasonal_naive": naive, "blend": 0.5 * ml + 0.5 * naive, "croston": cros}[champion]
    watch["forecast_next_q_MT"] = np.round(point, 1)
    q_int = residual_interval(point, champ_rel_err[champion], (0.1, 0.5, 0.9))
    watch["forecast_p10_MT"], watch["forecast_p50_MT"], watch["forecast_p90_MT"] = q_int[0.1], q_int[0.5], q_int[0.9]
    watch["expected_next_q_MT"] = (watch["decline_risk"] * 0.25 * watch["disp_roll4_mean"]
                                   + (1 - watch["decline_risk"]) * watch["forecast_next_q_MT"]).round(1)

    watch_out = watch[["supplier", "region", "fiscal_year", "fiscal_quarter",
                       "mill_dispatched_MT", "disp_roll4_mean", "share", "decline_risk", "risk_band",
                       "forecast_next_q_MT", "forecast_p10_MT", "forecast_p90_MT",
                       "expected_next_q_MT"]].rename(columns={
        "mill_dispatched_MT": "latest_q_dispatch_MT", "disp_roll4_mean": "trailing_4q_avg_MT",
        "share": "our_share_pct"}).sort_values("decline_risk", ascending=False).round(3)
    watch_out.to_csv(os.path.join(GOLD, "watchlist_decline_risk.csv"), index=False)

    # ---- opportunities (unchanged logic)
    cur = panel[panel["fiscal_year"] == latest_complete_fy].groupby(
        ["supplier", "region"], as_index=False).agg(
        dispatched_MT=("mill_dispatched_MT", "sum"), offtake_MT=("vaighai_offtake_est_MT", "sum"),
        purchased_MT=("vaighai_purchased_MT", "sum"))
    prev = panel[panel["fiscal_year"] == latest_complete_fy - 1].groupby("supplier")["mill_dispatched_MT"].sum()
    cur["prev_dispatched_MT"] = cur["supplier"].map(prev).fillna(0)
    cur = cur[cur["dispatched_MT"] > 0].copy()
    cur["share_pct"] = (cur["offtake_MT"] / cur["dispatched_MT"] * 100).clip(0, 100)
    cur["growth_pct"] = np.where(cur["prev_dispatched_MT"] > 0,
                                 (cur["dispatched_MT"] / cur["prev_dispatched_MT"] - 1) * 100, np.nan)
    size = cur["dispatched_MT"].rank(pct=True)
    headroom = 1 - cur["share_pct"] / 100
    growth = np.clip(1 + cur["growth_pct"].fillna(0) / 100, 0.5, 2.0) / 2.0
    cur["opportunity_score"] = (100 * size * headroom * growth).round(1)
    cur["untapped_MT"] = (cur["dispatched_MT"] - cur["offtake_MT"]).clip(lower=0).round(0)
    cur.sort_values("opportunity_score", ascending=False).round(2).to_csv(
        os.path.join(GOLD, "sourcing_opportunities.csv"), index=False)

    # ---- concentration (unchanged logic)
    rows = []
    for fy, gg in panel[(panel["fiscal_year"] >= 2021)
                        & (panel["fiscal_year"] <= latest_complete_fy)].groupby("fiscal_year"):
        tot = gg.groupby("supplier")["vaighai_purchased_MT"].sum()
        tot = tot[tot > 0]
        w = tot / tot.sum()
        rows.append({"fiscal_year": int(fy), "active_suppliers": len(tot),
                     "purchased_MT": round(float(tot.sum()), 1),
                     "hhi": round(float((w ** 2).sum()), 4),
                     "top5_share_pct": round(float(w.nlargest(5).sum() * 100), 1)})
    pd.DataFrame(rows).to_csv(os.path.join(GOLD, "concentration_risk.csv"), index=False)

    metrics = {"latest_quarter": f"FY{latest_fy} FQ{latest_qn}",
               "next_quarter": f"FY{latest_fy + (1 if latest_qn == 4 else 0)} FQ{1 if latest_qn == 4 else latest_qn + 1}",
               "latest_complete_fy": int(latest_complete_fy),
               "cv_folds": len(folds),
               "n_suppliers_scored": int(len(watch_out)),
               "n_critical": int((watch_out["risk_band"] == "Critical").sum()),
               "n_moderate": int((watch_out["risk_band"] == "Moderate").sum()),
               "decline_scoring_model": scorer_name,
               "decline_auc": round(float(auc), 3),
               "decline_auc_logistic": round(float(auc_lr), 3),
               "decline_auc_lightgbm": (round(float(auc_gb), 3) if auc_gb is not None else None),
               "decline_precision": prf["precision"], "decline_recall": prf["recall"],
               "decline_f1": prf["f1"], "decline_brier": prf["brier"],
               "forecast_champion": champion,
               "forecast_wape_pct": round(float(wapes[champion]), 1),
               "baseline_wape_pct": round(float(wapes.get("seasonal_naive", wapes[champion])), 1)}

    # ---- MLOps: register the (serializable) logistic model; note the scoring choice
    import model_registry
    note = f"scored with {scorer_name}; logistic AUC {auc_lr:.3f}" + (
        f", lightgbm AUC {auc_gb:.3f}" if auc_gb is not None else "")
    version = model_registry.register_run(
        clf_final, FEATURES, {"l2": bl2, "lr": blr, "forecast_champion": champion,
                              "scoring_model": scorer_name},
        {"val_auc": round(float(auc), 4), "val_wape_pct": round(float(wapes[champion]), 1),
         "precision": prf["precision"], "recall": prf["recall"],
         "train_rows": len(finalTr), "cv_folds": len(folds),
         "data_through": f"FY{latest_fy} FQ{latest_qn}"},
        promoted=True, note=note)
    metrics["model_version"] = version
    print(f"  registry: run logged as v{version} ({len(model_registry.history())} runs tracked)")

    with open(os.path.join(GOLD, "model_metrics.json"), "w") as f:
        json.dump(metrics, f)
    return metrics


if __name__ == "__main__":
    print(run())
