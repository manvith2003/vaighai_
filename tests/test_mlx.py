"""Unit tests for the numpy-only ML extensions (mlx.py)."""
import numpy as np

from mlx import (precision_recall_f1, PlattCalibrator, croston_tsb, residual_interval,
                 rolling_origin_folds, log_winsorize_cap)


def test_precision_recall_perfect():
    m = precision_recall_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], thr=0.5)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_precision_recall_threshold():
    m = precision_recall_f1([0, 1, 1], [0.9, 0.9, 0.1], thr=0.5)
    assert m["fp"] == 1 and m["fn"] == 1 and 0 <= m["f1"] <= 1


def test_croston_all_zero():
    assert croston_tsb([0, 0, 0, 0]) == 0.0


def test_croston_constant_positive():
    f = croston_tsb([10, 10, 10, 10, 10])
    assert 5.0 < f <= 10.5


def test_croston_intermittent_between():
    f = croston_tsb([0, 10, 0, 10, 0, 10, 0])
    assert 0.0 < f < 10.0            # occurrence prob < 1 pulls the point forecast down


def test_log_winsorize_keeps_large_real_mill():
    # heavy right tail: a genuinely large 9000 MT mill should survive the cap
    vals = list(np.random.default_rng(0).gamma(2.0, 200, size=500)) + [9000.0]
    cap = log_winsorize_cap(vals, k=5.0)
    assert cap >= 9000.0                      # not clipped toward the small-mill bulk
    assert cap >= np.median(vals)


def test_log_winsorize_flags_fatfinger():
    vals = [10, 12, 11, 9, 10, 13, 8, 11, 10, 12, 500000]  # one absurd entry
    cap = log_winsorize_cap(vals, k=5.0)
    assert cap < 500000                       # the fat-finger value is above the cap


def test_rolling_origin_folds():
    qs = list(range(1, 11))
    assert rolling_origin_folds(qs, k=3, min_train_quarters=4) == [8, 9, 10]
    assert rolling_origin_folds(qs, k=0, min_train_quarters=4) == list(range(5, 11))


def test_platt_monotonic():
    rng = np.random.default_rng(1)
    scores = np.r_[rng.normal(0.2, 0.1, 200), rng.normal(0.8, 0.1, 200)]
    y = np.r_[np.zeros(200), np.ones(200)]
    cal = PlattCalibrator().fit(scores, y)
    assert cal.transform([0.9])[0] > cal.transform([0.1])[0]
    assert 0.0 <= cal.transform([0.5])[0] <= 1.0


def test_residual_interval_ordered():
    point = np.array([100.0, 50.0, 200.0])
    rel = list(np.linspace(-0.4, 0.4, 40))
    q = residual_interval(point, rel, (0.1, 0.5, 0.9))
    assert np.all(q[0.1] <= q[0.5]) and np.all(q[0.5] <= q[0.9])
    assert np.all(q[0.1] >= 0)


def test_residual_interval_degenerate():
    q = residual_interval(np.array([10.0]), [0.1, 0.2], (0.1, 0.5, 0.9))
    assert q[0.1][0] == q[0.9][0] == 10.0     # too few samples -> point forecast
