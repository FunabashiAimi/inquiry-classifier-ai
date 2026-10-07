from __future__ import annotations

import argparse
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "inquiry_classifier.joblib"

CATEGORY_LABELS = {
    "account": "アカウント",
    "billing": "請求・支払い",
    "bug": "不具合",
    "cancel": "解約・退会",
    "feature": "機能要望",
    "sales": "導入相談",
    "usage": "使い方",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict inquiry category.")
    parser.add_argument("text", help="問い合わせ文")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run `python src/train.py` first."
        )

    model = joblib.load(args.model)
    predicted = model.predict([args.text])[0]

    print(f"入力: {args.text}")
    print(f"予測カテゴリ: {CATEGORY_LABELS.get(predicted, predicted)} ({predicted})")

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba([args.text])[0]
        classes = model.named_steps["classifier"].classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        print("確信度:")
        for category, score in ranked[:3]:
            label = CATEGORY_LABELS.get(category, category)
            print(f"- {label}: {score:.3f}")


if __name__ == "__main__":
    main()
