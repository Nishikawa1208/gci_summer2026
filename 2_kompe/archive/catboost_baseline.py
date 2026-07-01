import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ==========================================
# 【極秘テクニック】外れ値（異常値）の除去
# 異常に高い年収や、古すぎる生まれ年のデータはモデルを混乱させるため学習データから除外します
# ==========================================
train = train[train['annual_income'] < 200000].copy() # 年収20万以上の異常値を除外（存在すれば）
train = train[train['birth_year'] > 1920].copy()      # 生まれ年1920年以前の異常値を除外（存在すれば）
train.reset_index(drop=True, inplace=True) # インデックスの振り直し

# 2. 特徴量作成（最強だったV1ベース）
def create_features(df):
    df = df.copy()
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
    df['age'] = 2026 - df['birth_year']

    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    df['total_spend'] = df[spend_cols].sum(axis=1)
    df['necessity_spend'] = df[['spend_meat', 'spend_fish', 'spend_fruits']].sum(axis=1)
    df['luxury_spend'] = df[['spend_wines', 'spend_sweets', 'spend_gold']].sum(axis=1)
    df['luxury_ratio'] = df['luxury_spend'] / (df['total_spend'] + 1)

    purchase_cols = ['web_purchases', 'catalog_purchases', 'store_purchases']
    df['total_purchases'] = df[purchase_cols].sum(axis=1)
    df['avg_spend_per_purchase'] = df['total_spend'] / (df['total_purchases'] + 1)

    df['total_children'] = df['num_children'] + df['num_teenagers']
    df['is_married'] = df['marital_status'].apply(lambda x: 1 if x in ['Married', 'Together'] else 0)
    df['family_size'] = 1 + df['is_married'] + df['total_children']
    df['income_per_person'] = df['annual_income'] / df['family_size']

    df = df.drop(columns=['registration_date'])
    return df

print("特徴量作成と外れ値処理を実行中...")
train_processed = create_features(train)
test_processed = create_features(test)

# CatBoostはカテゴリ変数をそのまま扱えるため、文字列型にしておきます
cat_features = ['education_level', 'marital_status']
for col in cat_features:
    train_processed[col] = train_processed[col].astype(str)
    test_processed[col] = test_processed[col].astype(str)

X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# 3. CatBoostの学習と交差検証
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

print(f"\n--- CatBoost {folds}分割交差検証を開始 ---")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # CatBoostモデルの定義（デフォルトでも強力ですが、少し調整しています）
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=50
    )

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / folds
    print(f"Fold {fold+1} 完了")

cv_auc = roc_auc_score(y, oof_preds)
print("\n=== 結果 ===")
print(f"CV Score (AUC): {cv_auc:.4f} <-- どうなりましたか！？")

submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': test_preds})
submission.to_csv('submission_catboost.csv', index=False)
print("提出ファイル 'submission_catboost.csv' を作成しました！")
