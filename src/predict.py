from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "inquiry_classifier.joblib"
LABEL_PATH = ROOT_DIR / "data" / "category_labels.csv"


def load_category_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"category", "label"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing label columns: {', '.join(sorted(missing))}")
        return {
            row["category"].strip(): row["label"].strip()
            for row in reader
            if row["category"].strip() and row["label"].strip()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict inquiry category.")
    parser.add_argument("text", help="問い合わせ文")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=LABEL_PATH)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run `python src/train.py` first."
        )

    model = joblib.load(args.model)
    category_labels = load_category_labels(args.labels)
    predicted = model.predict([args.text])[0]

    print(f"入力: {args.text}")
    print(f"予測カテゴリ: {category_labels.get(predicted, predicted)} ({predicted})")

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba([args.text])[0]
        classes = model.named_steps["classifier"].classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        print("確信度:")
        for category, score in ranked[:3]:
            label = category_labels.get(category, category)
            print(f"- {label}: {score:.3f}")


if __name__ == "__main__":
    main()
