import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 2. V1特徴量
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

print("特徴量を作成し、上位24個の精鋭に絞り込みます...")
train_processed = create_features_v1(train)
test_processed = create_features_v1(test)

train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

X_all = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test_all = test_processed.drop(columns=['customer_id'])

# --- 上位24個の特徴量を再計算して抽出 ---
skf_temp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
temp_params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'random_state': 42}
feature_importances = np.zeros(X_all.shape[1])

for train_idx, valid_idx in skf_temp.split(X_all, y):
    model = lgb.train(temp_params, lgb.Dataset(X_all.iloc[train_idx], y.iloc[train_idx]),
                      valid_sets=[lgb.Dataset(X_all.iloc[valid_idx], y.iloc[valid_idx])],
                      num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
    feature_importances += model.feature_importance(importance_type='gain')

imp_df = pd.DataFrame({'feature': X_all.columns, 'importance': feature_importances})
top_24_features = imp_df.sort_values('importance', ascending=False).head(24)['feature'].tolist()

X_selected = X_all[top_24_features]
X_test_selected = X_test_all[top_24_features]

# 3. Optunaで見つけた「24変数専用」のベストパラメータ
best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'learning_rate': 0.05757442930097058,
    'num_leaves': 92,
    'max_depth': 8,
    'feature_fraction': 0.8291206962733898,
    'bagging_fraction': 0.5352225283496226,
    'bagging_freq': 6,
    'min_child_samples': 31,
}

# 4. シードアベレージング & 10-Fold CV で最強の安定化
SEEDS = [42, 77, 2026]
FOLDS = 10
oof_preds = np.zeros(len(X_selected))
test_preds = np.zeros(len(X_test_selected))

print(f"\n--- {FOLDS}分割交差検証 × {len(SEEDS)}シード ({FOLDS * len(SEEDS)}モデル) の学習を開始 ---")

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    params = best_params.copy()
    params['random_state'] = seed

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_selected, y)):
        X_tr, y_tr = X_selected.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_selected.iloc[valid_idx], y.iloc[valid_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_valid = lgb.Dataset(X_va, y_va, reference=lgb_train)

        model = lgb.train(
            params, lgb_train, valid_sets=[lgb_valid],
            num_boost_round=1500, callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        valid_pred = model.predict(X_va)
        oof_preds[valid_idx] += valid_pred / len(SEEDS)
        test_preds += model.predict(X_test_selected) / (FOLDS * len(SEEDS))

final_auc = roc_auc_score(y, oof_preds)
print("\n" + "="*40)
print(f"★★★ 最終集大成 CV Score (AUC): {final_auc:.4f} ★★★")
print("="*40)

# 5. 提出ファイルの作成
submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': test_preds})
submission.to_csv('submission_final_24feat.csv', index=False)
print("\n提出ファイル 'submission_final_24feat.csv' を作成しました！")
