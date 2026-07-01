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

# 2. V1特徴量（ここまでは同じ）
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
X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']

# Optunaで見つけたベストパラメータ
params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'verbose': -1, 'random_state': 42,
    'learning_rate': 0.03214991300054569, 'num_leaves': 88, 'max_depth': 5,
    'feature_fraction': 0.8076714735823799, 'bagging_fraction': 0.6070547458788909,
    'bagging_freq': 3, 'min_child_samples': 30,
}

# 3. まず全特徴量で学習して、重要度（Importance）を算出する
print("\n--- 全特徴量での重要度を計算中 ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
feature_importances = np.zeros(X.shape[1])

for train_idx, valid_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]

    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), valid_sets=[lgb.Dataset(X_va, y_va)],
                      num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
    feature_importances += model.feature_importance(importance_type='gain') / 5

# 重要度をデータフレームにまとめる
imp_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

# 4. 下位の特徴量を徐々に削ってCVスコアの変化を見る
print("\n--- 特徴量削減によるスコア変化の検証 ---")
best_auc = 0
best_num_features = 0
best_features = []

# 下位から5個ずつ削っていくテスト
for drop_num in [0, 2, 5, 8, 12, 15]:
    if drop_num == 0:
        current_features = imp_df['feature'].tolist()
    else:
        # 下位 drop_num 個を除外した特徴量リスト
        current_features = imp_df['feature'].tolist()[:-drop_num]

    X_subset = X[current_features]
    oof_preds = np.zeros(len(X_subset))

    for train_idx, valid_idx in skf.split(X_subset, y):
        X_tr, y_tr = X_subset.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_subset.iloc[valid_idx], y.iloc[valid_idx]

        model = lgb.train(params, lgb.Dataset(X_tr, y_tr), valid_sets=[lgb.Dataset(X_va, y_va)],
                          num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds[valid_idx] = model.predict(X_va)

    current_auc = roc_auc_score(y, oof_preds)
    print(f"下位 {drop_num:2d} 個を削除 (残り {len(current_features):2d} 特徴量) -> CV AUC: {current_auc:.4f}")

    if current_auc > best_auc:
        best_auc = current_auc
        best_num_features = len(current_features)
        best_features = current_features

print("\n" + "="*40)
print(f"★ ベストな特徴量数: {best_num_features}個 (CV AUC: {best_auc:.4f})")
print("="*40)
