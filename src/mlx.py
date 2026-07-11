"""ML EXTENSIONS (mlx) — self-contained, numpy-only helpers used by ml_train_score.

Kept in one module so the new modelling logic is easy to review and unit-test, and so
`utils.py` (shared everywhere) stays untouched. Nothing here has side effects on import.

Contents
--------
- precision_recall_f1        classification metrics at an operating threshold
- PlattCalibrator            turn raw scores into calibrated probabilities (1-D logistic)
- croston_tsb                intermittent-demand forecast (TSB variant) for zero-heavy series
- residual_interval          empirical P10/P50/P90 from backtest relative errors
- rolling_origin_folds       time-series CV fold selection (train on past, test next quarter)
- log_winsorize_cap          log-scale robust outlier cap (doesn't flatten genuinely large mills)
- fit_lgbm_classifier        OPTIONAL LightGBM challenger (returns None if lib absent)
"""
import numpy as np


# ---------------------------------------------------------------- metrics
def precision_recall_f1(y, p, thr=0.5):
    y = np.asarray(y, float); p = np.asarray(p, float)
    yhat = (p >= thr).astype(int)
    tp = float(((yhat == 1) & (y == 1)).sum()); fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum()); tn = float(((yhat == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    brier = float(np.mean((p - y) ** 2)) if len(y) else 0.0
    return dict(precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3),
                brier=round(brier, 3), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


# ---------------------------------------------------------------- calibration
class PlattCalibrator:
    """Platt scaling: 1-D logistic mapping raw score -> calibrated probability."""

    def fit(self, scores, y, lr=0.5, iters=800):
        s = np.asarray(scores, float)
        self.mu, self.sd = float(s.mean()), float(s.std() or 1.0)
        x = (s - self.mu) / self.sd
        y = np.asarray(y, float)
        a, b = 1.0, 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(a * x + b)))
            ga = float(np.mean((p - y) * x)); gb = float(np.mean(p - y))
            a -= lr * ga; b -= lr * gb
        self.a, self.b = a, b
        return self

    def transform(self, scores):
        x = (np.asarray(scores, float) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(self.a * x + self.b)))


# ---------------------------------------------------------------- intermittent demand
def croston_tsb(demand, alpha=0.15, beta=0.10):
    """TSB (Teunter–Syntetos–Babai) forecast for intermittent series with many zeros.
    Returns the next-period point forecast (demand-probability x demand-size)."""
    d = np.asarray(demand, float)
    if d.size == 0 or not np.any(d > 0):
        return 0.0
    nz = d[d > 0]
    p = 1.0 if d[0] > 0 else 0.0          # demand-occurrence probability
    z = float(nz[0])                       # demand size when it occurs
    for t in range(d.size):
        occurred = 1.0 if d[t] > 0 else 0.0
        p += alpha * (occurred - p)
        if occurred:
            z += beta * (d[t] - z)
    return max(0.0, p * z)


# ---------------------------------------------------------------- prediction intervals
def residual_interval(point, rel_errors, quantiles=(0.1, 0.5, 0.9)):
    """Empirical prediction interval: multiply the point forecast by backtest
    relative-error quantiles. `rel_errors` = (actual - pred)/max(pred,eps) samples."""
    point = np.asarray(point, float)
    if len(rel_errors) < 8:                # too few to estimate — degenerate to point
        return {q: np.round(point, 1) for q in quantiles}
    qs = np.quantile(np.asarray(rel_errors, float), quantiles)
    return {q: np.round(np.clip(point * (1 + qs[i]), 0, None), 1)
            for i, q in enumerate(quantiles)}


# ---------------------------------------------------------------- time-series CV
def rolling_origin_folds(quarters_sorted, k=6, min_train_quarters=4):
    """Return the test quarters for rolling-origin CV: each has >= min_train_quarters
    of history before it; take at most the last `k`."""
    qs = sorted(set(int(q) for q in quarters_sorted))
    eligible = qs[min_train_quarters:]
    return eligible[-k:] if k else eligible


# ---------------------------------------------------------------- robust capping
def log_winsorize_cap(positive_values, k=5.0):
    """Outlier cap on the log1p scale using median + k·MAD (robust, spread-aware).

    NOTE: a log-scale *quantile* would be pointless — quantiles are invariant under
    a monotonic transform, so expm1(quantile(log1p(x),q)) == quantile(x,q) exactly.
    A MAD-based threshold is genuinely different: on a heavy right tail it sits far
    above p99.5, so a real 9,000 MT mill isn't clipped toward the small-mill bulk;
    only values beyond k robust deviations (true fat-finger entries) are capped.
    """
    pos = np.asarray(positive_values, float)
    pos = pos[pos > 0]
    if pos.size < 4:
        return float(pos.max()) if pos.size else 0.0
    logv = np.log1p(pos)
    med = float(np.median(logv))
    mad = float(np.median(np.abs(logv - med)))
    if mad == 0:
        return float(np.expm1(logv.max()))  # no spread -> don't clip
    return float(np.expm1(med + k * 1.4826 * mad))


# ---------------------------------------------------------------- optional LightGBM
def fit_lgbm_classifier(X, y, seed=0):
    """Train a small, regularized LightGBM binary classifier as an optional challenger.
    Returns an object exposing predict_proba(X), or None if LightGBM isn't installed.
    Params are deliberately modest — data is small (MIR starts ~FY2021)."""
    try:
        import lightgbm as lgb
    except Exception:
        return None
    X = np.nan_to_num(np.asarray(X, float)); y = np.asarray(y, float)
    params = dict(objective="binary", num_leaves=15, min_data_in_leaf=20,
                  learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l1=1.0, lambda_l2=1.0, verbosity=-1, seed=seed)
    booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=200)

    class _Wrap:
        def __init__(self, b): self.b = b
        def predict_proba(self, Z):
            return self.b.predict(np.nan_to_num(np.asarray(Z, float)))
    return _Wrap(booster)
