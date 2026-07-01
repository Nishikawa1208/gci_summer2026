import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')

# 2. 前処理（ベースラインと同じ）
median_income = train['annual_income'].median()
train['annual_income'] = train['annual_income'].fillna(median_income)

train['registration_date'] = pd.to_datetime(train['registration_date'])
train['reg_year'] = train['registration_date'].dt.year
train['reg_month'] = train['registration_date'].dt.month
train = train.drop(columns=['registration_date'])

train_processed = pd.get_dummies(train, columns=['education_level', 'marital_status'], drop_first=True)

# 3. 特徴量とターゲットの分割
X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']

# 4. クロスバリデーション（5分割）の実行
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

# 各Foldのスコアを保存するリスト
auc_scores = []

print(f"--- {folds}分割交差検証を開始します ---")

# データを5つに分け、4つで学習、1つで検証を5回繰り返す
for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    # 学習データと検証データに分割
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

    # モデルの学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 検証データに対する予測（1になる確率）
    valid_preds = model.predict_proba(X_valid)[:, 1]

    # AUCスコアの計算
    auc = roc_auc_score(y_valid, valid_preds)
    auc_scores.append(auc)

    print(f"Fold {fold + 1} AUC: {auc:.4f}")

# 全Foldの平均スコアを出力
print("-" * 30)
print(f"平均 AUC (CV Score): {np.mean(auc_scores):.4f}")
