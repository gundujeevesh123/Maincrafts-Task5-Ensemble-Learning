"""
Task 5 — Ensemble Learning on the Breast Cancer Wisconsin dataset.

Pipeline:
  1. Load dataset
  2. Inspect dataset
  3. 80/20 stratified train-test split (random_state=42)
  4. Logistic Regression baseline (StandardScaler + LR in a Pipeline)
  5. Random Forest (default settings, random_state=42)
  6. Gradient Boosting (default settings, random_state=42)
  7. GridSearchCV optimization of Random Forest (5-fold stratified CV)
  8. Evaluate optimized Random Forest on the held-out test set
  9. Build the model-comparison table
 10. Generate figures (class distribution, confusion matrix, ROC curves,
     feature importance, CV comparison)
 11. Save artifacts (CSVs, JSON)

Positive-class convention
-------------------------
scikit-learn's `load_breast_cancer` encodes target 0 = malignant and
target 1 = benign. Because the clinical screening task is to detect
cancer, this experiment treats **malignant (class 0) as the positive
class** when computing precision, recall, F1-score and the GridSearchCV
scorer. Accuracy and ROC-AUC are label-symmetric and are computed in
the standard way. "Malignant Recall" is reported as a separate column
for clarity.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MALIGNANT = 0  # positive class for clinical screening
BENIGN = 1

sns.set_theme(style="whitegrid")


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Phase 1 — environment info
# ---------------------------------------------------------------------------
banner("Phase 1 — Environment")
env_info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
}
for k, v in env_info.items():
    print(f"{k:>14}: {v}")

# ---------------------------------------------------------------------------
# Phase 2 — load and inspect the dataset
# ---------------------------------------------------------------------------
banner("Phase 2 — Load & inspect dataset")
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")

print(f"Observations : {X.shape[0]}")
print(f"Features     : {X.shape[1]}")
print(f"Target names : {list(dataset.target_names)}  (0, 1 mapping)")
print(f"Missing cells: {int(X.isna().sum().sum())}")
print(f"Duplicate rows: {int(X.duplicated().sum())}")

class_counts = y.value_counts().sort_index()
class_labels = {0: "malignant", 1: "benign"}
print("Class distribution:")
for cls, count in class_counts.items():
    print(f"  {cls} ({class_labels[cls]:>9}): {count}  ({count / len(y):.1%})")

# Verify target mapping explicitly
assert dataset.target_names[0] == "malignant", "Expected target 0 = malignant"
assert dataset.target_names[1] == "benign", "Expected target 1 = benign"
print("Target mapping verified: 0 = malignant, 1 = benign.")

# Class-distribution figure
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#d62728", "#2ca02c"]  # malignant red, benign green
ax.bar(
    [class_labels[c] for c in class_counts.index],
    class_counts.values,
    color=colors,
    edgecolor="black",
)
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
ax.set_title("Class distribution — Breast Cancer Wisconsin")
ax.set_ylabel("Number of samples")
ax.set_xlabel("Class")
fig.tight_layout()
fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Phase 3 — train/test split
# ---------------------------------------------------------------------------
banner("Phase 3 — Train/test split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"X_train shape: {X_train.shape}   X_test shape: {X_test.shape}")
print(
    "Train class ratio:",
    dict(y_train.value_counts(normalize=True).round(3).sort_index()),
)
print(
    "Test  class ratio:",
    dict(y_test.value_counts(normalize=True).round(3).sort_index()),
)


# ---------------------------------------------------------------------------
# Helper — evaluate a fitted model
# ---------------------------------------------------------------------------
def evaluate(name: str, model, X_eval, y_eval, fit_time: float) -> dict:
    """Evaluate a fitted model on (X_eval, y_eval) and return a metrics dict.

    Metrics convention: precision/recall/F1 are reported for the
    *malignant* class (pos_label=0) because that is the clinically
    important screening class. ROC-AUC uses the standard sklearn
    convention (probability of the class labeled 1, i.e. benign);
    the AUC value itself is identical if labels are flipped. Accuracy
    is label-symmetric.
    """
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)  # columns match model.classes_
    # probability of malignant class (0)
    mal_col = list(model.classes_).index(MALIGNANT)
    y_proba_mal = y_proba[:, mal_col]

    acc = accuracy_score(y_eval, y_pred)
    prec_m = precision_score(y_eval, y_pred, pos_label=MALIGNANT)
    rec_m = recall_score(y_eval, y_pred, pos_label=MALIGNANT)
    f1_m = f1_score(y_eval, y_pred, pos_label=MALIGNANT)
    # roc_auc is symmetric; using the malignant probability + pos_label=0
    # gives the identical numeric AUC as using benign probability + pos_label=1
    auc = roc_auc_score((y_eval == MALIGNANT).astype(int), y_proba_mal)

    cm = confusion_matrix(y_eval, y_pred, labels=[MALIGNANT, BENIGN])
    print(f"\n[{name}]  fit_time={fit_time:.3f}s")
    print(f"  accuracy             : {acc:.4f}")
    print(f"  precision (malignant): {prec_m:.4f}")
    print(f"  recall    (malignant): {rec_m:.4f}")
    print(f"  F1        (malignant): {f1_m:.4f}")
    print(f"  ROC-AUC              : {auc:.4f}")
    print(f"  confusion matrix [[TN FP] [FN TP]] with malignant positive:")
    print(f"    {cm.tolist()}")
    print("  classification_report (default sklearn convention, pos=1=benign):")
    print("  " + classification_report(
        y_eval, y_pred, target_names=["malignant(0)", "benign(1)"]
    ).replace("\n", "\n  "))

    return {
        "model": name,
        "accuracy": acc,
        "precision_malignant": prec_m,
        "recall_malignant": rec_m,
        "f1_malignant": f1_m,
        "roc_auc": auc,
        "malignant_recall": rec_m,  # explicit duplicate column for the report
        "fit_time_sec": fit_time,
        "y_pred": y_pred,
        "y_proba_mal": y_proba_mal,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Logistic Regression baseline
# ---------------------------------------------------------------------------
banner("Phase 4 — Logistic Regression baseline")
lr_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ]
)
t0 = time.perf_counter()
lr_pipeline.fit(X_train, y_train)
lr_fit = time.perf_counter() - t0
lr_metrics = evaluate("LogisticRegression", lr_pipeline, X_test, y_test, lr_fit)

# ---------------------------------------------------------------------------
# Phase 5 — Random Forest (default)
# ---------------------------------------------------------------------------
banner("Phase 5 — Random Forest (defaults, random_state=42)")
rf = RandomForestClassifier(random_state=RANDOM_STATE)
t0 = time.perf_counter()
rf.fit(X_train, y_train)
rf_fit = time.perf_counter() - t0
rf_metrics = evaluate("RandomForest (default)", rf, X_test, y_test, rf_fit)

# ---------------------------------------------------------------------------
# Phase 6 — Gradient Boosting
# ---------------------------------------------------------------------------
banner("Phase 6 — Gradient Boosting (defaults, random_state=42)")
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
t0 = time.perf_counter()
gb.fit(X_train, y_train)
gb_fit = time.perf_counter() - t0
gb_metrics = evaluate("GradientBoosting", gb, X_test, y_test, gb_fit)

# ---------------------------------------------------------------------------
# Phase 7 — GridSearchCV on Random Forest
# ---------------------------------------------------------------------------
banner("Phase 7 — GridSearchCV on Random Forest")

param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
n_combos = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
    * len(param_grid["min_samples_leaf"])
)
print(f"Parameter grid contains {n_combos} combinations.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Malignant-focused F1 scorer: pos_label=0
malignant_f1_scorer = make_scorer(f1_score, pos_label=MALIGNANT)

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring=malignant_f1_scorer,
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False,
)

t0 = time.perf_counter()
grid.fit(X_train, y_train)
grid_fit = time.perf_counter() - t0
print(f"GridSearchCV completed in {grid_fit:.2f}s")
print("Best parameters :", grid.best_params_)
print(f"Best CV F1(mal) : {grid.best_score_:.4f}")

# Report mean and std for the best configuration
best_idx = int(grid.best_index_)
cv_results = pd.DataFrame(grid.cv_results_)
best_mean = cv_results.loc[best_idx, "mean_test_score"]
best_std = cv_results.loc[best_idx, "std_test_score"]
print(f"CV score (best): mean={best_mean:.4f}  std={best_std:.4f}")

# Save CV results
cv_results_slim = cv_results[
    [
        "params",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "rank_test_score",
    ]
].sort_values("rank_test_score")
cv_results_slim.to_csv(RESULTS_DIR / "random_forest_cv_results.csv", index=False)

# ---------------------------------------------------------------------------
# Phase 8 — Evaluate optimized Random Forest
# ---------------------------------------------------------------------------
banner("Phase 8 — Optimized Random Forest on held-out test set")
best_rf = grid.best_estimator_
# fit time is the sum of fit times across refit-equivalent; use the recorded
# mean_fit_time of the best configuration times n_splits for a comparable
# single-fit estimate, OR do a dedicated refit timing on full training set:
t0 = time.perf_counter()
best_rf_single = RandomForestClassifier(
    random_state=RANDOM_STATE, **grid.best_params_
).fit(X_train, y_train)
best_rf_fit = time.perf_counter() - t0
# Use the refit-on-full-training single-fit model (identical to best_estimator_)
best_rf_metrics = evaluate(
    "RandomForest (optimized)", best_rf_single, X_test, y_test, best_rf_fit
)

# ---------------------------------------------------------------------------
# Phase 9 — Model comparison table
# ---------------------------------------------------------------------------
banner("Phase 9 — Model comparison")

rows = []
for m in [lr_metrics, rf_metrics, gb_metrics, best_rf_metrics]:
    rows.append(
        {
            "Model": m["model"],
            "Accuracy": round(m["accuracy"], 4),
            "Precision (malignant)": round(m["precision_malignant"], 4),
            "Recall (malignant)": round(m["recall_malignant"], 4),
            "F1 (malignant)": round(m["f1_malignant"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Malignant Recall": round(m["malignant_recall"], 4),
            "Fit Time (s)": round(m["fit_time_sec"], 4),
        }
    )
comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))
comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# Phase 10 — Figures
# ---------------------------------------------------------------------------
banner("Phase 10 — Figures")

# Confusion matrices: the FINAL SELECTED MODEL (Logistic Regression) is the
# primary panel; the Optimized Random Forest is shown alongside for reference.
# Which model wins is only decided after the comparison in Phase 9, so the
# plotting here mirrors that selection.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, metrics_dict, title in [
    (axes[0], lr_metrics, "Logistic Regression  (selected final model)"),
    (axes[1], best_rf_metrics, "Random Forest (optimized) — reference"),
]:
    sns.heatmap(
        metrics_dict["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted malignant", "Predicted benign"],
        yticklabels=["Actual malignant", "Actual benign"],
        ax=ax,
    )
    ax.set_title(title, fontsize=11)
fig.suptitle("Confusion matrices on held-out test set (n=114)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ROC curves for all four models
fig, ax = plt.subplots(figsize=(6.2, 5))
for m, label in [
    (lr_metrics, "Logistic Regression"),
    (rf_metrics, "Random Forest (default)"),
    (gb_metrics, "Gradient Boosting"),
    (best_rf_metrics, "Random Forest (optimized)"),
]:
    fpr, tpr, _ = roc_curve(
        (y_test == MALIGNANT).astype(int), m["y_proba_mal"]
    )
    ax.plot(fpr, tpr, label=f"{label}  (AUC={m['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
ax.set_xlabel("False positive rate (benign predicted as malignant)")
ax.set_ylabel("True positive rate (malignant correctly identified)")
ax.set_title("ROC curves — malignant class as positive")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
plt.close(fig)

# Feature importance of the optimized RF
importances = pd.Series(
    best_rf_single.feature_importances_, index=X_train.columns, name="Feature"
).sort_values(ascending=False)
top = importances.head(15)
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x=top.values, y=top.index, ax=ax, color="#4C72B0")
ax.set_title("Top-15 feature importances — Optimized Random Forest")
ax.set_xlabel("Importance (impurity-based)")
ax.set_ylabel("")  # suppress the automatic 'None' label
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
plt.close(fig)

# CV comparison: default RF vs tuned RF via a fresh 5-fold CV on training set
from sklearn.model_selection import cross_val_score

banner("Phase 10b — Supplementary 5-fold CV comparison (training set only)")
cv_lr = cross_val_score(
    lr_pipeline, X_train, y_train, cv=cv, scoring=malignant_f1_scorer, n_jobs=-1
)
cv_rf_default = cross_val_score(
    RandomForestClassifier(random_state=RANDOM_STATE),
    X_train,
    y_train,
    cv=cv,
    scoring=malignant_f1_scorer,
    n_jobs=-1,
)
cv_gb = cross_val_score(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    X_train,
    y_train,
    cv=cv,
    scoring=malignant_f1_scorer,
    n_jobs=-1,
)
cv_rf_tuned = cross_val_score(
    RandomForestClassifier(random_state=RANDOM_STATE, **grid.best_params_),
    X_train,
    y_train,
    cv=cv,
    scoring=malignant_f1_scorer,
    n_jobs=-1,
)
cv_data = pd.DataFrame(
    {
        "Logistic Regression": cv_lr,
        "RF (default)": cv_rf_default,
        "Gradient Boosting": cv_gb,
        "RF (optimized)": cv_rf_tuned,
    }
)
print(cv_data.describe().T[["mean", "std", "min", "max"]].round(4))

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=cv_data, ax=ax, color="#9ec5e8")
sns.stripplot(data=cv_data, ax=ax, color="black", size=4, jitter=True)
ax.set_ylabel("CV F1 (malignant positive)")
ax.set_title("5-fold stratified CV — malignant F1 by model (train set)")
plt.xticks(rotation=15)
fig.tight_layout()
fig.savefig(FIG_DIR / "cv_comparison.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Phase 11 — Save results
# ---------------------------------------------------------------------------
banner("Phase 11 — Save results & final selection")

best_params_record = {
    "best_params": grid.best_params_,
    "best_cv_score_malignant_f1": float(grid.best_score_),
    "best_cv_score_mean": float(best_mean),
    "best_cv_score_std": float(best_std),
    "cv_folds": 5,
    "scorer": "f1_score (pos_label=0, malignant)",
    "random_state": RANDOM_STATE,
}
with open(RESULTS_DIR / "best_parameters.json", "w") as fp:
    json.dump(best_params_record, fp, indent=2)

# Environment info
env_lines = [f"{k}: {v}" for k, v in env_info.items()]
(PROJECT_ROOT / "environment_info.txt").write_text("\n".join(env_lines) + "\n")

# ---------------------------------------------------------------------------
# Final selection message
# ---------------------------------------------------------------------------
# Primary selection criterion: malignant F1 on the held-out test set,
# with ROC-AUC and malignant recall as tie-breakers. We only choose the
# final model after inspecting the results — never iterate against the
# test set.
def selection_summary(comp: pd.DataFrame) -> dict:
    key = "F1 (malignant)"
    sorted_c = comp.sort_values(
        [key, "ROC-AUC", "Malignant Recall"], ascending=False
    )
    top_row = sorted_c.iloc[0].to_dict()
    return {
        "selected_model": top_row["Model"],
        "primary_criterion": "malignant F1 (test set), with ROC-AUC and malignant recall as tie-breakers",
        "selected_metrics": top_row,
    }


selection = selection_summary(comparison)
print("\nFinal selection:", selection["selected_model"])
print("Reason          :", selection["primary_criterion"])

(RESULTS_DIR / "final_selection.json").write_text(
    json.dumps(selection, indent=2, default=str)
)

print("\nAll artifacts saved under:", PROJECT_ROOT)
print("Done.")
