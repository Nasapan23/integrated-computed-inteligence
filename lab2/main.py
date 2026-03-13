from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "outputs"


def load_wine_data() -> pd.DataFrame:
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    red["wine_type"] = "red"

    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    white["wine_type"] = "white"

    data = pd.concat([red, white], ignore_index=True)
    data["wine_type"] = data["wine_type"].map({"red": 0, "white": 1}).astype(int)
    return data


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
            ]
        ),
        "SVM (RBF Kernel)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Naive Bayes (Gaussian)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_wine_data()
    data["target"] = (data["quality"] >= 6).astype(int)

    X = data.drop(columns=["quality", "target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print("Dataset shape:", data.shape)
    print("Target distribution (0=bad, 1=good):")
    print(y.value_counts(normalize=True).sort_index().round(4))
    print("-" * 70)

    models = build_models()
    metrics_rows: list[dict[str, float | str]] = []
    confusion_results: dict[str, list[list[int]]] = {}
    reports: dict[str, str] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["bad", "good"])

        metrics_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cross_entropy_loss": log_loss(y_test, y_prob),
            }
        )

        confusion_results[name] = cm.tolist()
        reports[name] = report

        print(name)
        print(report)
        print("-" * 70)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1", ascending=False)
    print("Summary metrics (sorted by F1):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    metrics_df.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)

    with (OUTPUT_DIR / "classification_reports.txt").open("w", encoding="utf-8") as file:
        for model_name, report_text in reports.items():
            file.write(f"{model_name}\n")
            file.write(report_text)
            file.write("\n" + "-" * 70 + "\n")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (model_name, cm) in enumerate(confusion_results.items()):
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred bad", "Pred good"],
            yticklabels=["True bad", "True good"],
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices - Wine Quality Classification", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"- {OUTPUT_DIR / 'classification_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'classification_reports.txt'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
