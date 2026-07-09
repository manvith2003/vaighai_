"""STAGE 5 — ML MODELS (Gold). Azure equivalent: Azure ML batch endpoint + registry.

  1. Decline risk  — P(next-quarter dispatch falls >50% below trailing 4-q avg)
  2. Forecast      — champion of {ridge, seasonal-naive, blend} picked by backtest WAPE
  3. Opportunity   — size x share-headroom x growth ranking (latest complete FY)
  4. Concentration — HHI + top-5 share per FY

Reads the Silver panel + weather index, writes Gold CSVs and returns metrics.
"""
import os

import numpy as np
import pandas as pd

from common import (GOLD, SILVER, Q_NUM, QTY_COLS, LogisticModel, RidgeModel,
                    auc_score, banner, current_complete_quarter)

FEATURES = [
    "mill_dispatched_MT", "disp_lag1", "disp_lag2", "disp_lag3", "disp_lag4",
    "disp_roll4_mean", "disp_roll4_std", "disp_qoq", "disp_slope4",
    "purch_roll4_mean", "offtake_roll4_mean", "share", "share_roll4",
    "q_since_active", "qnum", "region_seasonal_idx", "next_seasonal_idx", "monsoon_idx",
]


def dense_grid(panel):
    """Supplier x quarter grid from first activity onward (zeros are real)."""
    panel = panel.groupby(["supplier", "qidx", "fiscal_year", "fiscal_quarter"], as_index=False)[QTY_COLS].sum()
    qmax = int(panel["qidx"].max())
    frames = []
    for sup, g in panel.groupby("supplier"):
        qs = np.arange(int(g["qidx"].min()), qmax + 1)
        frames.append(pd.DataFrame({"supplier": sup, "qidx": qs}))
    grid = pd.concat(frames, ignore_index=True).merge(panel, on=["supplier", "qidx"], how="left")
    for c in QTY_COLS:
        grid[c] = grid[c].fillna(0.0)
    grid["fiscal_year"] = ((grid["qidx"] - 1) // 4).astype(int)
    grid["qnum"] = grid["qidx"] - grid["fiscal_year"] * 4
    grid["fiscal_quarter"] = "FQ" + grid["qnum"].astype(str)
    return grid.sort_values(["supplier", "qidx"]).reset_index(drop=True)


def build_features(grid, region_map, weather):
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
    # region seasonality learned from data
    reg_q = g.groupby(["region", "fiscal_quarter"])[d].mean()
    reg_m = g.groupby("region")[d].mean()
    seas = (reg_q / reg_m).rename("region_seasonal_idx").reset_index()
    g = g.merge(seas, on=["region", "fiscal_quarter"], how="left")
    g["region_seasonal_idx"] = g["region_seasonal_idx"].fillna(1.0)
    nxt = g["qnum"] % 4 + 1
    g["next_fq"] = "FQ" + nxt.astype(str)
    g = g.merge(seas.rename(columns={"fiscal_quarter": "next_fq",
                                     "region_seasonal_idx": "next_seasonal_idx"}),
                on=["region", "next_fq"], how="left")
    g["next_seasonal_idx"] = g["next_seasonal_idx"].fillna(1.0)
    # weather monsoon index (source 3)
    g = g.merge(weather[["region", "fiscal_quarter", "monsoon_idx"]],
                on=["region", "fiscal_quarter"], how="left")
    g["monsoon_idx"] = g["monsoon_idx"].fillna(1.0)
    return g


def run():
    banner("MODELS", "Gold layer — decline risk, forecast, opportunities, concentration")
    panel = pd.read_csv(os.path.join(SILVER, "supply_panel.csv"))
    weather = pd.read_csv(os.path.join(SILVER, "weather_quarterly.csv"))
    panel["qidx"] = panel["fiscal_year"] * 4 + panel["fiscal_quarter"].map(Q_NUM)
    region_map = (panel[panel["region"] != "UNKNOWN"].groupby("supplier")["region"]
                  .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "UNKNOWN"))

    cutoff = current_complete_quarter()
    grid = dense_grid(panel[(panel["fiscal_year"] >= 2021) & (panel["qidx"] <= cutoff)])
    g = build_features(grid, region_map, weather)
    latest_q = min(int(g.loc[g["mill_dispatched_MT"] > 0, "qidx"].max()), cutoff)
    latest_fy, latest_qn = (latest_q - 1) // 4, latest_q - ((latest_q - 1) // 4) * 4
    latest_complete_fy = latest_fy - 1 if latest_qn < 4 else latest_fy

    # ---- 1. decline risk
    grp = g.groupby("supplier")
    g["disp_next"] = grp["mill_dispatched_MT"].shift(-1)
    g["declined_next"] = (g["disp_next"] < 0.5 * g["disp_roll4_mean"]).astype(int)
    eligible = (g["disp_roll4_mean"] > 0) & g["disp_lag1"].notna()
    hist = g[eligible & g["disp_next"].notna() & (g["qidx"] < latest_q)]
    cut = hist["qidx"].max() - 4
    tr, va = hist[hist["qidx"] <= cut], hist[hist["qidx"] > cut]
    clf = LogisticModel().fit(tr[FEATURES].values, tr["declined_next"].values)
    auc = auc_score(va["declined_next"].values, clf.predict_proba(va[FEATURES].values))
    watch = g[eligible & (g["qidx"] == latest_q)].copy()
    watch["decline_risk"] = clf.predict_proba(watch[FEATURES].values)
    watch["risk_band"] = pd.cut(watch["decline_risk"], [0, 0.4, 0.7, 1.0],
                                labels=["Low", "Moderate", "Critical"]).astype(str)
    print(f"  decline model: AUC {auc:.3f} on held-out quarters "
          f"(base rate {hist['declined_next'].mean()*100:.1f}%)")

    # ---- 2. forecast (champion selection)
    reg = RidgeModel().fit(tr[FEATURES].values, np.log1p(tr["disp_next"].values))
    pred_va = np.clip(np.expm1(reg.predict(va[FEATURES].values)), 0, None)
    base = (va["disp_roll4_mean"] * va["next_seasonal_idx"]).clip(lower=0).values
    y = va["disp_next"].values
    wapes = {"ridge": np.abs(pred_va - y).sum() / max(y.sum(), 1) * 100,
             "seasonal_naive": np.abs(base - y).sum() / max(y.sum(), 1) * 100,
             "blend": np.abs(0.5 * pred_va + 0.5 * base - y).sum() / max(y.sum(), 1) * 100}
    champion = min(wapes, key=wapes.get)
    ml = np.clip(np.expm1(reg.predict(watch[FEATURES].values)), 0, None)
    naive = (watch["disp_roll4_mean"] * watch["next_seasonal_idx"]).clip(lower=0).values
    watch["forecast_next_q_MT"] = {"ridge": ml, "seasonal_naive": naive,
                                   "blend": 0.5 * ml + 0.5 * naive}[champion].round(1)
    # risk-adjusted expectation
    watch["expected_next_q_MT"] = (watch["decline_risk"] * 0.25 * watch["disp_roll4_mean"]
                                   + (1 - watch["decline_risk"]) * watch["forecast_next_q_MT"]).round(1)
    print(f"  forecast champion: {champion} (WAPE {wapes[champion]:.1f}%; "
          + ", ".join(f"{k} {v:.1f}%" for k, v in wapes.items()) + ")")

    watch_out = watch[["supplier", "region", "fiscal_year", "fiscal_quarter",
                       "mill_dispatched_MT", "disp_roll4_mean", "share",
                       "decline_risk", "risk_band", "forecast_next_q_MT",
                       "expected_next_q_MT"]].rename(columns={
        "mill_dispatched_MT": "latest_q_dispatch_MT", "disp_roll4_mean": "trailing_4q_avg_MT",
        "share": "our_share_pct"}).sort_values("decline_risk", ascending=False).round(3)
    watch_out.to_csv(os.path.join(GOLD, "watchlist_decline_risk.csv"), index=False)

    # ---- 3. opportunities (latest complete FY)
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
    opp = cur.sort_values("opportunity_score", ascending=False).round(2)
    opp.to_csv(os.path.join(GOLD, "sourcing_opportunities.csv"), index=False)
    print(f"  opportunities: {len(opp)} active mills ranked for FY{latest_complete_fy}")

    # ---- 4. concentration
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
    conc = pd.DataFrame(rows)
    conc.to_csv(os.path.join(GOLD, "concentration_risk.csv"), index=False)
    print(f"  concentration: top-5 share FY{conc.fiscal_year.iloc[0]} "
          f"{conc.top5_share_pct.iloc[0]}% -> FY{conc.fiscal_year.iloc[-1]} {conc.top5_share_pct.iloc[-1]}%")

    metrics = {"latest_quarter": f"FY{latest_fy} FQ{latest_qn}",
               "next_quarter": f"FY{latest_fy + (1 if latest_qn == 4 else 0)} FQ{1 if latest_qn == 4 else latest_qn + 1}",
               "latest_complete_fy": int(latest_complete_fy),
               "n_suppliers_scored": int(len(watch_out)),
               "n_critical": int((watch_out["risk_band"] == "Critical").sum()),
               "n_moderate": int((watch_out["risk_band"] == "Moderate").sum()),
               "decline_auc": round(auc, 3),
               "decline_base_rate_pct": round(float(hist["declined_next"].mean() * 100), 1),
               "forecast_champion": champion,
               "forecast_wape_pct": round(float(wapes[champion]), 1),
               "baseline_wape_pct": round(float(wapes["seasonal_naive"]), 1)}
    pd.Series(metrics).to_json(os.path.join(GOLD, "model_metrics.json"))
    return metrics


if __name__ == "__main__":
    print(run())
