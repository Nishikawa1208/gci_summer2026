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

# 2. 最高スコアを叩き出したV1特徴量
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

print("V1特徴量を作成中...")
train_processed = create_features_v1(train)
test_processed = create_features_v1(test)

train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# 3. 最強パラメータ
best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'learning_rate': 0.03214991300054569,
    'num_leaves': 88,
    'max_depth': 5,
    'feature_fraction': 0.8076714735823799,
    'bagging_fraction': 0.6070547458788909,
    'bagging_freq': 3,
    'min_child_samples': 30,
}

# 4. シードアベレージング & 10-Fold CV
SEEDS = [42, 77, 2026] # 3つの異なる乱数シード
FOLDS = 10

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"\n--- {FOLDS}分割交差検証 × {len(SEEDS)}シード ({FOLDS * len(SEEDS)}モデル) のアンサンブルを開始 ---")

for seed in SEEDS:
    print(f"\n>> Seed {seed} の学習を開始...")
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

    # パラメータのシードも更新
    params = best_params.copy()
    params['random_state'] = seed

    seed_oof = np.zeros(len(X))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        model = lgb.train(
            params, lgb_train, valid_sets=[lgb_valid],
            num_boost_round=1500, callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # OOF予測とテスト予測を合算
        valid_pred = model.predict(X_valid)
        seed_oof[valid_idx] = valid_pred
        oof_preds[valid_idx] += valid_pred / len(SEEDS)

        test_preds += model.predict(X_test) / (FOLDS * len(SEEDS))

    seed_auc = roc_auc_score(y, seed_oof)
    print(f"Seed {seed} 単体のCV AUC: {seed_auc:.4f}")

# 5. 最終結果
final_auc = roc_auc_score(y, oof_preds)
print("\n" + "="*40)
print(f"★★★ 最終アンサンブル CV Score (AUC): {final_auc:.4f} ★★★")
print("="*40)

submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': test_preds})
submission.to_csv('submission_seed_avg.csv', index=False)
print("\n提出ファイル 'submission_seed_avg.csv' を作成しました！")
