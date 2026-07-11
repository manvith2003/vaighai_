"""ML DIAGNOSTICS (additive, read-only) — explainability report for the decline model.

The rolling-origin CV, precision/recall and calibration now run natively inside
`ml_train_score.py` (written to data/gold/model_metrics.json). This module adds the one
thing that belongs in a review pack but not in the scoring path: a **feature-importance
ranking** (standardized logistic coefficients) plus a **calibration/reliability table**,
so you can prove weather and share actually contribute.

Run AFTER a normal pipeline run:   python3 src/ml_diagnostics.py
Writes: docs/ml_diagnostics_report.md
"""
import json
import os

import numpy as np
import pandas as pd

from utils import (GOLD, DOCS, Q_NUM, LogisticModel, banner,
                   current_complete_quarter, wh_connect)
from mlx import PlattCalibrator, precision_recall_f1
from ml_train_score import (FEATURES, dense_grid, build_base_features,
                            add_fit_features, add_rain_next_q)


def _pool():
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
    base = build_base_features(grid, region_map, weather)
    latest_q = min(int(base.loc[base["mill_dispatched_MT"] > 0, "qidx"].max()), cutoff)
    base = add_rain_next_q(base, weather, weather_next, latest_q)
    g = base.groupby("supplier")
    base["disp_next"] = g["mill_dispatched_MT"].shift(-1)
    base["declined_next"] = (base["disp_next"] < 0.5 * base["disp_roll4_mean"]).astype(int)
    elig = (base["disp_roll4_mean"] > 0) & base["disp_lag1"].notna()
    pool = base[elig & base["disp_next"].notna() & (base["qidx"] < latest_q)].copy()
    return add_fit_features(pool, pool)


def run():
    banner("ML-DIAGNOSTICS", "Feature importance + reliability report")
    poolF = _pool()
    clf = LogisticModel().fit(poolF[FEATURES].values, poolF["declined_next"].values)
    imp = sorted(zip(FEATURES, clf.w[1:]), key=lambda t: -abs(t[1]))

    # in-sample reliability (bins) after Platt calibration
    raw = clf.predict_proba(poolF[FEATURES].values)
    cal = PlattCalibrator().fit(raw, poolF["declined_next"].values)
    p = cal.transform(raw)
    y = poolF["declined_next"].values
    prf = precision_recall_f1(y, p, thr=0.5)
    edges = np.linspace(0, 1, 6)
    rel = []
    for i in range(5):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == 4 else p < edges[i + 1])
        if m.sum():
            rel.append((f"{edges[i]:.1f}-{edges[i+1]:.1f}", int(m.sum()),
                        round(float(p[m].mean()), 3), round(float(y[m].mean()), 3)))

    metrics = {}
    mp = os.path.join(GOLD, "model_metrics.json")
    if os.path.exists(mp):
        metrics = json.load(open(mp))

    L = ["# Supply Radar — ML Diagnostics (explainability)", "",
         "## Cross-validated performance (from the pipeline run)", ""]
    if metrics:
        L += [f"- Scoring model: **{metrics.get('decline_scoring_model','?')}** · "
              f"CV AUC **{metrics.get('decline_auc','?')}** "
              f"(logistic {metrics.get('decline_auc_logistic','?')}, "
              f"lightgbm {metrics.get('decline_auc_lightgbm','n/a')})",
              f"- Precision/Recall/F1 @0.5: {metrics.get('decline_precision','?')} / "
              f"{metrics.get('decline_recall','?')} / {metrics.get('decline_f1','?')} · "
              f"Brier {metrics.get('decline_brier','?')} · CV folds {metrics.get('cv_folds','?')}",
              f"- Forecast champion: {metrics.get('forecast_champion','?')} "
              f"(WAPE {metrics.get('forecast_wape_pct','?')}%)"]
    else:
        L += ["- run `python3 src/pipeline.py data/raw` first to populate model_metrics.json"]

    L += ["", "## Feature importance (standardized logistic coefficients)", "",
          "Positive = raises decline risk; magnitude = strength. Confirms whether "
          "weather (`rain_*`, `monsoon_*`) and `share` features carry weight.", "",
          "| Feature | Std. coef |", "|---|---|"]
    L += [f"| {f} | {c:+.3f} |" for f, c in imp]
    L += ["", "## In-sample reliability (calibrated)", "",
          f"Precision/Recall/F1 @0.5: {prf['precision']}/{prf['recall']}/{prf['f1']}", "",
          "| Prob bin | n | mean predicted | observed decline |", "|---|---|---|---|"]
    L += [f"| {b} | {n} | {mp_} | {ob} |" for b, n, mp_, ob in rel]
    L += ["", "---", "*Read-only diagnostic — does not affect the production model or gold outputs.*"]

    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "ml_diagnostics_report.md"), "w") as f:
        f.write("\n".join(L))
    print(f"  -> docs/ml_diagnostics_report.md (top feature: {imp[0][0]} {imp[0][1]:+.3f})")


if __name__ == "__main__":
    run()
