"""
Quick test for the flood susceptibility workflow (RF vs CART).

What it does
------------
Re-runs the modelling/validation step of the paper in about a minute, without
a Google Earth Engine account:

  * random 5-fold cross-validation,
  * spatial 5-fold cross-validation (grid blocks of the point coordinates,
    GroupKFold), as in CODE_COLAB_Tetouan.py,
  * an independent random 80/20 held-out test,

for Random Forest (200 trees) and CART, reporting AUROC, accuracy, precision,
sensitivity and F1.

Input data
----------
* If `training_data_with_coords.csv` exists next to this script (the file
  written by CODE_COLAB_Tetouan.py: one row per point, columns `label`, the 12
  conditioning factors and the point coordinates), it is used.
  A different file can be given as the first argument.
* Otherwise a small SYNTHETIC data set (210 flood / 115 non-flood points, 12
  factors) is generated, only to check that the environment works. In that
  case the numbers do not reproduce the paper.

Usage
-----
    pip install numpy pandas scikit-learn
    python quick_test.py                       # CSV if present, else synthetic
    python quick_test.py path/to/points.csv    # explicit CSV
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, KFold, cross_val_predict,
                                     train_test_split)
from sklearn.tree import DecisionTreeClassifier

SEED = 42
THRESHOLD = 0.5       # probability threshold for accuracy/precision/sensitivity/F1
CSV_DEFAULT = "training_data_with_coords.csv"
NON_FEATURES = {"label", "geometry", "longitude", "latitude", "random",
                "system:index", ".geo"}
SYNTH_FACTORS = ["DEM", "slope", "Aspect", "plan_curvature", "LULC",
                 "Drainage_Density", "Distance_River", "TWI", "TPI", "SPI",
                 "NDVI", "precipitation"]


# ---------------------------------------------------------------- data ----
def make_synthetic(n_flood=210, n_non=115, seed=SEED):
    rng = np.random.default_rng(seed)
    n = n_flood + n_non
    y = np.r_[np.ones(n_flood), np.zeros(n_non)].astype(int)
    X = rng.normal(size=(n, len(SYNTH_FACTORS)))
    X[:, 0] -= 0.8 * y     # lower elevation
    X[:, 1] -= 0.8 * y     # gentler slope
    X[:, 6] -= 0.6 * y     # closer to rivers
    X[:, 7] += 0.5 * y     # higher TWI
    df = pd.DataFrame(X, columns=SYNTH_FACTORS)
    df["label"] = y
    df["longitude"] = rng.uniform(-5.5, -5.3, n)
    df["latitude"] = rng.uniform(35.5, 35.7, n)
    return df


def load_csv(path):
    df = pd.read_csv(path)
    if "geometry" in df.columns and "longitude" not in df.columns:
        pts = df["geometry"].apply(lambda s: ast.literal_eval(s)["coordinates"])
        df["longitude"] = pts.apply(lambda c: c[0])
        df["latitude"] = pts.apply(lambda c: c[1])
    missing = {"label", "longitude", "latitude"} - set(df.columns)
    if missing:
        sys.exit(f"Missing column(s) in {path}: {sorted(missing)}")
    return df.dropna()


def grid_blocks(lon, lat, n=5):
    """Same grid blocks as in CODE_COLAB_Tetouan.py (n x n cells)."""
    bx = (lon.max() - lon.min()) / n
    by = (lat.max() - lat.min()) / n
    ix = ((lon - lon.min()) // bx).astype(int)
    iy = ((lat - lat.min()) // by).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(ix, iy)])


# ------------------------------------------------------------- metrics ----
def scores(y_true, proba, thr=THRESHOLD):
    pred = (proba >= thr).astype(int)
    return {"AUROC": roc_auc_score(y_true, proba),
            "Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Sensitivity": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred)}


def cv_auc(model, X, y, cv, groups=None):
    aucs = []
    for tr, te in cv.split(X, y, groups):
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


# ---------------------------------------------------------------- main ----
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    if os.path.exists(path):
        df = load_csv(path)
        source = f"real data: {path}"
    else:
        df = make_synthetic()
        source = "SYNTHETIC data (CSV not found): numbers do not reproduce the paper"
    features = [c for c in df.columns if c not in NON_FEATURES]
    X, y = df[features].to_numpy(float), df["label"].to_numpy(int)
    groups = grid_blocks(df["longitude"].to_numpy(), df["latitude"].to_numpy())

    print(f"Data: {source}")
    print(f"Points: {len(df)} (flood = {int(y.sum())}, non-flood = {int((1 - y).sum())})")
    print(f"Factors ({len(features)}): {', '.join(features)}\n")

    models = {
        "RF": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "CART": DecisionTreeClassifier(random_state=SEED),
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rows = {}
    for name, model in models.items():
        r = {"AUROC random 5-fold": cv_auc(model, X, y,
                                           KFold(5, shuffle=True, random_state=SEED)),
             "AUROC spatial 5-fold": cv_auc(model, X, y, GroupKFold(5), groups)}
        model.fit(Xtr, ytr)
        r.update({f"Held-out {k}": v
                  for k, v in scores(yte, model.predict_proba(Xte)[:, 1]).items()})
        rows[name] = r
    print(pd.DataFrame(rows).round(2).to_string())
    print(f"\n(threshold for accuracy/precision/sensitivity/F1 = {THRESHOLD})")
    print("Quick test finished successfully.")


if __name__ == "__main__":
    main()
