import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 2. V1特徴量の作成（スコアが良かったもの）
def create_features_v1(df):
    df = df.copy()
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())

    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['reg_year'] = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month
    df = df.drop(columns=['registration_date'])

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
    df['is_parent'] = (df['total_children'] > 0).astype(int)
    df['is_married'] = df['marital_status'].apply(lambda x: 1 if x in ['Married', 'Together'] else 0)
    df['family_size'] = 1 + df['is_married'] + df['total_children']
    df['income_per_person'] = df['annual_income'] / df['family_size']

    df = pd.get_dummies(df, columns=['education_level', 'marital_status'], drop_first=True)
    return df

train_processed = create_features_v1(train)
test_processed = create_features_v1(test)

train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# 3. Optunaで見つけたベストパラメータ
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
    # === Optunaの出力結果をここにセット ===
    'learning_rate': 0.03214991300054569,
    'num_leaves': 88,
    'max_depth': 5,
    'feature_fraction': 0.8076714735823799,
    'bagging_fraction': 0.6070547458788909,
    'bagging_freq': 3,
    'min_child_samples': 30,
}

# 4. K-Fold CVによる学習と予測（アンサンブル）
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_processed))

print("最適化されたパラメータで最終モデルを学習中...")

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # 5回の学習それぞれでTestデータを予測し、結果を足し合わせる
    test_preds += model.predict(X_test) / folds

# 5. 提出ファイルの作成
submission = pd.DataFrame({
    'customer_id': test['customer_id'],
    'target': test_preds
})
submission.to_csv('submission_optuna.csv', index=False)
print("提出ファイル 'submission_optuna.csv' を作成しました！")
