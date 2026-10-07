from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


def load_keyword_map(path: Path) -> dict[str, list[str]]:
    keyword_map: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"category", "keyword"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing keyword columns: {', '.join(sorted(missing))}")

        for row in reader:
            category = row["category"].strip()
            keyword = row["keyword"].strip()
            if not category or not keyword:
                continue
            keyword_map.setdefault(category, []).append(keyword)

    if not keyword_map:
        raise ValueError(f"No keywords loaded from {path}")
    return keyword_map


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """Create keyword count features from an externally supplied keyword map."""

    def __init__(self, keyword_map: dict[str, list[str]]) -> None:
        self.keyword_map = keyword_map

    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "KeywordFeatureExtractor":
        self.categories_ = sorted(self.keyword_map.keys())
        self.normalized_keyword_map_ = {
            category: [keyword.lower() for keyword in self.keyword_map[category]]
            for category in self.categories_
        }
        return self

    def transform(self, X: Iterable[str]) -> sparse.csr_matrix:
        rows: list[list[float]] = []
        for text in X:
            text_value = str(text).lower()
            rows.append(
                [
                    sum(
                        1
                        for keyword in self.normalized_keyword_map_[category]
                        if keyword in text_value
                    )
                    for category in self.categories_
                ]
            )
        return sparse.csr_matrix(rows, dtype=float)
