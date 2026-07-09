"""MODEL REGISTRY — lightweight local MLOps. Azure equivalent: Azure ML registry.

Every pipeline run:
  1. tunes hyperparameters on the held-out window,
  2. logs the run (params, metrics, data window) to data/models/registry.jsonl,
  3. saves versioned model weights (model_v{N}.json),
  4. PROMOTION GATE: the newly tuned challenger only replaces the current
     champion if it does not regress on the current validation window.

The `ml_runs` warehouse table (built from the registry) lets Metabase chart
model quality over time — the audit trail of "is the model improving?".
"""
import json
import os

import numpy as np
import pandas as pd

from utils import ROOT, LogisticModel

MODELS_DIR = os.path.join(ROOT, "data", "models")
REGISTRY = os.path.join(MODELS_DIR, "registry.jsonl")
os.makedirs(MODELS_DIR, exist_ok=True)


def history():
    if not os.path.exists(REGISTRY):
        return []
    with open(REGISTRY) as f:
        return [json.loads(line) for line in f if line.strip()]


def champion_version():
    """Latest promoted version, or None."""
    promoted = [h for h in history() if h.get("promoted")]
    return promoted[-1]["version"] if promoted else None


def load_model(version):
    with open(os.path.join(MODELS_DIR, f"model_v{version}.json")) as f:
        w = json.load(f)
    clf = LogisticModel()
    clf.w = np.array(w["w"])
    clf.mu = np.array(w["mu"])
    clf.sd = np.array(w["sd"])
    return clf, w.get("features", [])


def register_run(clf, features, params, metrics, promoted, note=""):
    version = len(history()) + 1
    with open(os.path.join(MODELS_DIR, f"model_v{version}.json"), "w") as f:
        json.dump({"w": clf.w.tolist(), "mu": clf.mu.tolist(), "sd": clf.sd.tolist(),
                   "features": features, "params": params}, f)
    record = {"version": version, "run_at": pd.Timestamp.today().isoformat(timespec="seconds"),
              "params": params, "promoted": bool(promoted), "note": note, **metrics}
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(record) + "\n")
    return version


def runs_table():
    """Registry as a flat DataFrame for the warehouse (ml_runs)."""
    h = history()
    if not h:
        return pd.DataFrame()
    rows = []
    for r in h:
        row = {k: v for k, v in r.items() if k != "params"}
        row.update({f"param_{k}": v for k, v in r.get("params", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)
