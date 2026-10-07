from __future__ import annotations

from typing import Iterable

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


CATEGORY_KEYWORDS = {
    "account": [
        "ログイン",
        "パスワード",
        "認証",
        "アカウント",
        "メールアドレス",
        "招待",
        "ユーザー",
        "二段階",
    ],
    "billing": [
        "請求",
        "料金",
        "支払い",
        "決済",
        "領収書",
        "カード",
        "振込",
        "契約",
        "金額",
    ],
    "bug": [
        "エラー",
        "不具合",
        "表示されません",
        "保存されません",
        "落ちます",
        "止まります",
        "真っ白",
        "タイムアウト",
    ],
    "cancel": [
        "解約",
        "退会",
        "キャンセル",
        "停止",
        "削除",
        "終了",
        "無料プラン",
        "更新前",
    ],
    "feature": [
        "追加",
        "機能",
        "要望",
        "できるように",
        "連携",
        "一括",
        "並べ替え",
        "下書き",
    ],
    "sales": [
        "導入",
        "見積",
        "資料",
        "デモ",
        "トライアル",
        "法人",
        "相談",
        "契約前",
        "他社",
    ],
    "usage": [
        "方法",
        "手順",
        "使い方",
        "設定",
        "どこ",
        "教えて",
        "確認したい",
        "操作",
        "インポート",
    ],
}


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """Create simple domain keyword count features for inquiry categories."""

    def __init__(self, keyword_map: dict[str, list[str]] | None = None) -> None:
        self.keyword_map = keyword_map or CATEGORY_KEYWORDS
        self.categories_ = list(self.keyword_map.keys())

    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "KeywordFeatureExtractor":
        return self

    def transform(self, X: Iterable[str]) -> sparse.csr_matrix:
        rows: list[list[float]] = []
        for text in X:
            text_value = str(text)
            features = []
            for category in self.categories_:
                keywords = self.keyword_map[category]
                features.append(sum(1 for keyword in keywords if keyword in text_value))
            rows.append(features)
        return sparse.csr_matrix(rows, dtype=float)
