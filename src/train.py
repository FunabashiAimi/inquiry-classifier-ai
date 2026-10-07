from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_features import KeywordFeatureExtractor, load_keyword_map


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "inquiries.csv"
KEYWORD_PATH = ROOT_DIR / "data" / "category_keywords.csv"
MODEL_PATH = ROOT_DIR / "models" / "inquiry_classifier.joblib"
REPORT_DIR = ROOT_DIR / "reports"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"text", "category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["text"] != ""]
    return df


def build_pipeline(keyword_path: Path = KEYWORD_PATH) -> Pipeline:
    keyword_map = load_keyword_map(keyword_path)
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                        ("keywords", KeywordFeatureExtractor(keyword_map=keyword_map)),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    C=3.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def save_confusion_matrix(y_true: pd.Series, y_pred: list[str], labels: list[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Inquiry Category Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inquiry category classifier.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--keywords", type=Path, default=KEYWORD_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=0.25,
        random_state=42,
        stratify=df["category"],
    )

    model = build_pipeline(args.keywords)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(df["category"].unique())
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_DIR / "classification_report.csv", index=True)
    save_confusion_matrix(y_test, y_pred, labels)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "metrics.csv", index=False)
    joblib.dump(model, args.model)

    print("Training completed.")
    print(f"Model: {args.model}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
