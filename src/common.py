"""Shared constants, paths and the tiny numpy-only ML models."""
import os

import numpy as np
import pandas as pd

import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE = os.path.join(ROOT, "data", "bronze")
SILVER = os.path.join(ROOT, "data", "silver")
GOLD = os.path.join(ROOT, "data", "gold")
# SQLite needs a lock-friendly filesystem; work in temp, snapshot into the repo.
DB_PATH = os.path.join(tempfile.gettempdir(), "supply_radar.db")
DB_SNAPSHOT = os.path.join(ROOT, "data", "supply_radar.db")
DOCS = os.path.join(ROOT, "docs")
DASH = os.path.join(ROOT, "dashboard")

Q_NUM = {"FQ1": 1, "FQ2": 2, "FQ3": 3, "FQ4": 4}
QTY_COLS = ["mill_produced_MT", "mill_dispatched_MT", "vaighai_offtake_est_MT", "vaighai_purchased_MT"]

for d in (BRONZE, SILVER, GOLD, DASH):
    os.makedirs(d, exist_ok=True)


def current_complete_quarter():
    """Latest COMPLETE fiscal quarter index (fiscal year Apr-Mar, ending-year label)."""
    today = pd.Timestamp.today()
    fy = today.year + 1 if today.month >= 4 else today.year
    qn = {4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 1: 4, 2: 4, 3: 4}[today.month]
    return fy * 4 + qn - 1


def _standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


class LogisticModel:
    """L2-regularised logistic regression (numpy only) — production swap: Azure ML."""

    def fit(self, X, y, lr=0.3, iters=600, l2=1e-3):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        Xs, self.mu, self.sd = _standardize(X)
        Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
        y = np.asarray(y, dtype=float)
        w = np.zeros(Xs.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xs @ w))
            g = Xs.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
            w -= lr * g
        self.w = w
        return self

    def predict_proba(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        Xs = (X - self.mu) / self.sd
        Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
        return 1 / (1 + np.exp(-Xs @ self.w))


class RidgeModel:
    """Ridge regression, closed form (numpy only)."""

    def fit(self, X, y, l2=1.0):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        Xs, self.mu, self.sd = _standardize(X)
        Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
        A = Xs.T @ Xs + l2 * np.eye(Xs.shape[1])
        self.w = np.linalg.solve(A, Xs.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        Xs = (X - self.mu) / self.sd
        Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
        return Xs @ self.w


def auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)
    n1, n0 = y_true.sum(), (1 - y_true).sum()
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def banner(stage, msg):
    print(f"\n{'=' * 70}\n[{stage}] {msg}\n{'=' * 70}")
