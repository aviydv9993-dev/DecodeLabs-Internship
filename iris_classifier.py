"""
DecodeLabs | AI Internship | Project 2: Data Classification Using AI
=====================================================================
Pipeline (IPO):
    INPUT   -> Load Iris, inspect, stratified train/test split
    PROCESS -> Scale features, tune K with cross-validation, train KNN
    OUTPUT  -> Accuracy, confusion matrix, precision/recall/F1, plots

Extras beyond the brief:
    * Scaler lives INSIDE a Pipeline, so test data never leaks into fitting.
    * K is chosen by 5-fold cross-validation on the training set only.
    * KNN is benchmarked against 3 other simple classifiers.
    * A predict_flower() helper classifies brand-new measurements.

Run:  python iris_classifier.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.20
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def banner(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# --------------------------------------------------------------------------
# 1. INPUT: load and understand the dataset
# --------------------------------------------------------------------------
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, iris.feature_names, list(iris.target_names)


def explore(df: pd.DataFrame, features: list) -> None:
    banner("STEP 1 | DATASET OVERVIEW")
    print(f"Samples: {len(df)} | Features: {len(features)} | "
          f"Classes: {df['species'].nunique()}")
    print(f"Missing values: {int(df[features].isna().sum().sum())}\n")
    print("Class balance:")
    print(df["species"].value_counts().to_string(), "\n")
    print("Per-class feature means (cm):")
    print(df.groupby("species")[features].mean().round(2).to_string())


# --------------------------------------------------------------------------
# 2. PROCESS: split, tune K, train
# --------------------------------------------------------------------------
def make_knn(k: int) -> Pipeline:
    """Scaling + KNN bundled together so scaling is learned from train only."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def tune_k(X_train, y_train, k_range=range(1, 31)):
    """Pick K by 5-fold CV on the TRAINING data only (test set stays locked)."""
    scores = [
        cross_val_score(make_knn(k), X_train, y_train, cv=5).mean()
        for k in k_range
    ]
    errors = [1 - s for s in scores]
    # Smallest K that achieves the best CV score -> simplest good model
    best_k = list(k_range)[int(np.argmax(scores))]

    plt.figure(figsize=(8, 4.5))
    plt.plot(list(k_range), errors, marker="o", color="#003f7f")
    plt.scatter([best_k], [1 - max(scores)], s=260, facecolors="none",
                edgecolors="#e8590c", linewidths=3, label=f"Best K = {best_k}")
    plt.xlabel("K (number of neighbours)")
    plt.ylabel("Cross-validated error rate")
    plt.title("Tuning the engine: choosing K")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "k_tuning.png", dpi=150)
    plt.close()
    return best_k, max(scores)


def benchmark(X_train, y_train, X_test, y_test, best_k):
    """Compare KNN against other simple classifiers on the same split."""
    candidates = {
        f"KNN (k={best_k})": make_knn(best_k),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC()),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    }
    rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
        })
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False)


# --------------------------------------------------------------------------
# 3. OUTPUT: validation
# --------------------------------------------------------------------------
def evaluate(model, X_test, y_test, class_names):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    banner("STEP 4 | FINAL MODEL EVALUATION (held-out test set)")
    print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"Macro F1 : {f1_score(y_test, pred, average='macro'):.3f}\n")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print("\nPer-class report:")
    print(classification_report(y_test, pred, target_names=class_names,
                                digits=3))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix | Iris KNN")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def predict_flower(model, class_names, sepal_l, sepal_w, petal_l, petal_w):
    """Classify a brand-new flower and show the model's vote confidence."""
    sample = pd.DataFrame(
        [[sepal_l, sepal_w, petal_l, petal_w]],
        columns=model.feature_names_in_,
    )
    label = str(class_names[int(model.predict(sample)[0])])
    proba = model.predict_proba(sample)[0]
    return label, {str(c): round(float(p), 2) for c, p in zip(class_names, proba)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df, features, class_names = load_data()
    explore(df, features)

    X, y = df[features], df["target"]

    banner("STEP 2 | STRATIFIED TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"(stratified, shuffled, seed={SEED})")

    banner("STEP 3 | TUNING K WITH 5-FOLD CROSS-VALIDATION")
    best_k, cv_acc = tune_k(X_train, y_train)
    print(f"Best K = {best_k} (CV accuracy {cv_acc:.3f}) "
          f"-> plot saved to {OUT_DIR / 'k_tuning.png'}")

    model = make_knn(best_k).fit(X_train, y_train)
    evaluate(model, X_test, y_test, class_names)

    banner("STEP 5 | BENCHMARK vs OTHER ALGORITHMS")
    table = benchmark(X_train, y_train, X_test, y_test, best_k)
    print(table.round(3).to_string(index=False))

    banner("STEP 6 | PREDICTING UNSEEN FLOWERS")
    demos = {
        "Small petals": (5.0, 3.4, 1.5, 0.2),
        "Medium petals": (6.0, 2.8, 4.5, 1.4),
        "Large petals": (6.9, 3.1, 5.8, 2.1),
    }
    for name, meas in demos.items():
        label, conf = predict_flower(model, class_names, *meas)
        print(f"{name:14s} {meas} -> {label}  votes: {conf}")


if __name__ == "__main__":
    main()
