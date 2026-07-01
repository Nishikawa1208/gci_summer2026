import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 2. 非常に安定していたV1特徴量
def create_features_v1(df):
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

    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['reg_year'] = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month
    df = df.drop(columns=['registration_date'])

    df = pd.get_dummies(df, columns=['education_level', 'marital_status'], drop_first=True)
    return df

print("特徴量を作成中...")
train_processed = create_features_v1(train)
test_processed = create_features_v1(test)

train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# 3. 交差検証の設定
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

# Level 1の予測結果を格納する配列
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_rf = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_rf = np.zeros(len(X_test))

# バージョン違いによるエラーを防ぐため、Early Stoppingに頼らず手堅いパラメータで固定
model_lgb = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=88, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
model_xgb = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', verbosity=0)
model_rf = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42, n_jobs=-1)

print("\n--- Level 1: ベースモデル3種（専門家）の学習を開始 ---")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    # LightGBM
    model_lgb.fit(X_tr, y_tr)
    oof_lgb[va_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / folds

    # XGBoost
    model_xgb.fit(X_tr, y_tr)
    oof_xgb[va_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / folds

    # Random Forest
    model_rf.fit(X_tr, y_tr)
    oof_rf[va_idx] = model_rf.predict_proba(X_va)[:, 1]
    test_rf += model_rf.predict_proba(X_test)[:, 1] / folds

    print(f"Fold {fold+1} 完了")

print("\n[Level 1 単体モデルの CV AUC]")
print(f"LightGBM    : {roc_auc_score(y, oof_lgb):.4f}")
print(f"XGBoost     : {roc_auc_score(y, oof_xgb):.4f}")
print(f"RandomForest: {roc_auc_score(y, oof_rf):.4f}")

# 4. Level 2（メタモデル）の学習と予測
print("\n--- Level 2: メタモデル（マネージャー）による統合 ---")
# 各専門家の予測結果を新しい「特徴量」としてデータフレーム化
X_level2 = pd.DataFrame({'lgb': oof_lgb, 'xgb': oof_xgb, 'rf': oof_rf})
X_test_level2 = pd.DataFrame({'lgb': test_lgb, 'xgb': test_xgb, 'rf': test_rf})

oof_meta = np.zeros(len(y))
test_meta = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_level2, y)):
    X_tr_m, y_tr_m = X_level2.iloc[tr_idx], y.iloc[tr_idx]
    X_va_m, y_va_m = X_level2.iloc[va_idx], y.iloc[va_idx]

    meta_model = LogisticRegression()
    meta_model.fit(X_tr_m, y_tr_m)

    oof_meta[va_idx] = meta_model.predict_proba(X_va_m)[:, 1]
    test_meta += meta_model.predict_proba(X_test_level2)[:, 1] / folds

final_auc = roc_auc_score(y, oof_meta)
print("\n" + "="*40)
print(f"★★★ 最終スタッキング CV Score (AUC): {final_auc:.4f} ★★★")
print("="*40)

# 5. 提出ファイルの作成
submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': test_meta})
submission.to_csv('submission_stacking.csv', index=False)
print("\n提出ファイル 'submission_stacking.csv' を作成しました！")
