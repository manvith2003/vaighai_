"""Shared paths, constants, warehouse connection and numpy-only ML models."""
import os
import tempfile

import numpy as np
import pandas as pd

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE = os.path.join(ROOT, "data", "bronze")
SILVER = os.path.join(ROOT, "data", "silver")
GOLD = os.path.join(ROOT, "data", "gold")
DOCS = os.path.join(ROOT, "docs")
DASH = os.path.join(ROOT, "dashboard")
# SQLite fallback warehouse: work in temp (lock-friendly), snapshot into repo
SQLITE_PATH = os.path.join(tempfile.gettempdir(), "supply_radar.db")
SQLITE_SNAPSHOT = os.path.join(ROOT, "data", "supply_radar.db")

Q_NUM = {"FQ1": 1, "FQ2": 2, "FQ3": 3, "FQ4": 4}
MONTH_TO_FQ = {4: "FQ1", 5: "FQ1", 6: "FQ1", 7: "FQ2", 8: "FQ2", 9: "FQ2",
               10: "FQ3", 11: "FQ3", 12: "FQ3", 1: "FQ4", 2: "FQ4", 3: "FQ4"}
QTY_COLS = ["mill_produced_MT", "mill_dispatched_MT", "vaighai_offtake_est_MT", "vaighai_purchased_MT"]

for d in (BRONZE, SILVER, GOLD, DASH, DOCS):
    os.makedirs(d, exist_ok=True)


def wh_connect():
    """Warehouse connection: Postgres if WAREHOUSE_URL set, else SQLite."""
    if config.WAREHOUSE_URL:
        from sqlalchemy import create_engine
        return create_engine(config.WAREHOUSE_URL), "postgres"
    import sqlite3
    return sqlite3.connect(SQLITE_PATH), "sqlite"


def wh_execute(con, backend, sql):
    if backend == "postgres":
        from sqlalchemy import text
        with con.begin() as c:
            c.execute(text(sql))
    else:
        con.execute(sql)
        con.commit()


def fiscal_of(ts):
    fy = ts.year + 1 if ts.month >= 4 else ts.year
    return fy, MONTH_TO_FQ[ts.month]


def current_complete_quarter():
    today = pd.Timestamp.today()
    fy, fq = fiscal_of(today)
    return fy * 4 + Q_NUM[fq] - 1


def _standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


class LogisticModel:
    """L2-regularised logistic regression (numpy only)."""

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
