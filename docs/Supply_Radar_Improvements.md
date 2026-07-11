# Supply Radar — Review Notes & Improvements

*Which ML models are used, how weather fits in, and where the local stack was improved
(models, training, architecture) — plus additional models worth trying.*

---

## 1. What ML models are used (names)

Everything is hand-written in **numpy only** (see `src/utils.py` and `src/mlx.py`).

| Purpose | Model / method | Where |
| --- | --- | --- |
| **Decline risk** | **L2 Logistic Regression** (gradient descent), **calibrated** (Platt); **optional LightGBM challenger** used only if it beats logistic on CV | `utils.py`, `mlx.py`, `ml_train_score.py` |
| **Volume forecast** | Champion of **Ridge Regression** / **Seasonal-naïve** / **Blend** / **Croston-TSB** by rolling-CV WAPE, with **P10/P50/P90** intervals | `utils.py`, `mlx.py`, `ml_train_score.py` |
| **Opportunity score** | Rule-based: size-percentile × share-headroom × growth | `ml_train_score.py` |
| **Concentration** | **HHI** + top-5 supplier share per FY | `ml_train_score.py` |

## 2. Weather

Weather is a first-class model input, not just context. `extract_weather_api.py` pulls Open-Meteo rainfall (climatology fallback offline); `transform_supply_panel.py` builds `rain_mm`, `monsoon_idx`, **`rain_lag1/2`**, **`monsoon_cum`**, **`extreme_wet`**, **`dry_spell`**; the ML stage also uses `rain_next_q` (a live 16-day forecast for the target quarter). The dashboard's **Weather-impact panel** overlays live rainfall on dispatch per region with a correlation readout, and `ml_diagnostics.py` prints feature importance so you can prove weather actually contributes.

## 3. Improvements implemented (this pass)

- **Leakage fix** — seasonal indices and weather climatology fallbacks are computed on the **training slice only** (per CV fold, and for scoring), so validation/scoring quarters never leak into the fitted stats. This makes the AUC honest.
- **Rolling-origin CV** — replaces the single held-out window; pooled metrics across up to 6 quarters (single-split fallback when quarters are few).
- **Calibration + precision/recall** — Platt-calibrated risk; precision/recall/F1/Brier reported, not just AUC.
- **LightGBM challenger** — optional, regularized, small; used for scoring only if it beats logistic on CV. Pipeline runs unchanged if the library is absent.
- **Croston/TSB + quantiles** — intermittent-demand forecast candidate + P10/P90 on the watchlist.
- **Richer weather + UNKNOWN-region fallback** — lagged rain, cumulative monsoon, extreme flags; UNKNOWN regions get a state-wide fallback instead of zeros.
- **Robust winsorizing** — log-scale median + 5·MAD cap (a log-scale *quantile* would be identical to p99.5 — quantiles are invariant under monotone transforms — so MAD is used) so genuinely large mills aren't flattened.
- **Tests** — `tests/` covers the new helpers, a transform integration test, and an import smoke test.
- **Dead-code archive** — the 8 modules the real pipeline never imports are moved to `_archive/`.

## 4. Further ideas (not yet done)

- Per-supplier prediction intervals via quantile regression rather than pooled residuals.
- Survival/hazard model for "quarters until a mill goes dormant."
- Humidity/temperature weather features (drying speed), not just rain.
- Restore the model-registry **promotion gate** around the new CV AUC (it was simplified in this pass — every run currently registers).

## 5. Caveat

MIR data starts ~FY2021, so there are relatively few quarters. Keep models modest —
LightGBM-with-regularization + good CV, nothing deep.
