# 多言語問い合わせ文カテゴリ分類AI

日本語・英語・中国語の問い合わせ文を、内容に応じてカテゴリ分類する機械学習プロジェクトです。

## 目的

カスタマーサポートに届く問い合わせ文を自動分類し、担当部署への振り分けや対応優先度の整理を効率化することを想定して実装しました。

## 分類カテゴリ

- `account`: アカウント
- `billing`: 請求・支払い
- `bug`: 不具合
- `cancel`: 解約・退会
- `feature`: 機能要望
- `sales`: 導入相談
- `usage`: 使い方

## 使用技術

- Python
- pandas
- scikit-learn
- TF-IDF
- キーワード特徴量
- Logistic Regression
- matplotlib / seaborn

## 実装内容

- 問い合わせ文とカテゴリのCSVデータ作成
- 欠損値・空文字の除去
- 文字N-gramによるTF-IDF特徴量化
- 日本語・英語・中国語のドメインキーワード出現数を特徴量として追加
- Logistic Regressionによるカテゴリ分類モデルの学習
- Accuracy、F1-score、混同行列による評価
- 学習済みモデルの保存
- CLIから新しい問い合わせ文を分類する予測機能

単語分割には言語ごとの前処理が必要になる場合がありますが、このプロジェクトでは環境構築を軽くするため、文字単位のN-gramを使っています。さらに「請求」「invoice」「发票」など、カテゴリ判定に効きやすい日本語・英語・中国語のドメインキーワードの出現数も特徴量化し、少量データでも分類しやすいようにしました。

カテゴリ別キーワードはコードに直接書かず、`data/category_keywords.csv` で管理しています。後から他の言語を追加する場合は、同じカテゴリに対して `language` と `keyword` を追記し、必要に応じて `data/inquiries.csv` に学習例を追加して再学習します。

カテゴリの表示名も `data/category_labels.csv` に分離しているため、表示名の変更やカテゴリ追加時のコード修正を抑えられます。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 学習と評価

```bash
python src/train.py
```

実行後、以下が生成されます。

- `models/inquiry_classifier.joblib`
- `reports/metrics.csv`
- `reports/classification_report.csv`
- `reports/confusion_matrix.png`

現在の多言語サンプルデータでは、固定の分割条件で以下の評価結果になりました。

| 指標 | 値 |
| --- | ---: |
| Accuracy | 0.923 |
| Macro F1 | 0.921 |
| Weighted F1 | 0.923 |

## 予測デモ

```bash
python src/predict.py "請求書の宛名を変更したいです"
```

英語・中国語の問い合わせ文も分類できます。

```bash
python src/predict.py "I cannot log in after resetting my password"
python src/predict.py "请提供10个用户的报价"
```

## 言語・キーワードの追加方法

例として韓国語の請求カテゴリを追加する場合:

```csv
category,language,keyword
billing,ko,청구서
billing,ko,결제
billing,ko,요금
```

その後、必要に応じて `data/inquiries.csv` に韓国語の問い合わせ例を追加し、再学習します。

```bash
python src/train.py
```

出力例:

```text
入力: 請求書の宛名を変更したいです
予測カテゴリ: 請求・支払い (billing)
確信度:
- 請求・支払い: 0.800
- 不具合: 0.087
- 導入相談: 0.039
```

## 工夫した点

- 少量データでも動作を確認しやすいよう、シンプルな教師あり分類タスクとして設計しました。
- 日本語・英語・中国語テキストに対して、追加辞書や形態素解析器なしで動く文字N-gram TF-IDFを採用しました。
- TF-IDFだけでは分類しづらいカテゴリに対応するため、問い合わせ業務で重要になりやすい多言語キーワード特徴量を追加しました。
- キーワードやカテゴリ表示名をCSVに分離し、言語追加時にコードを変更しなくても拡張できる構成にしました。
- 単に分類するだけでなく、評価指標と混同行列を出力し、どのカテゴリを誤分類しやすいか確認できるようにしました。
- 予測時に上位カテゴリの確信度も表示し、運用時に人が確認しやすい形にしました。

## 今後の改善案

- 実データを追加してカテゴリごとの表現の偏りを減らす
- LinearSVC、Random Forest、LightGBMなどとの比較
- Streamlitでブラウザから試せるデモ画面を追加
- 誤分類データを分析して学習データやカテゴリ定義を改善
- 多言語Embeddingモデルを使い、未知の表現や学習データが少ない言語への対応力を高める
