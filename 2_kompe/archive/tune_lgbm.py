import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

def create_features_v1(df):
    """最もスコアが良かったV1の特徴量エンジニアリング"""
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
X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']

# 2. Optunaの目的関数（探索ルール）を定義
def objective(trial):
    # 探索するパラメータの範囲を指定
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    auc_scores = []

    # K-Fold CVでこのパラメータ設定でのスコアを計算
    for train_index, valid_index in skf.split(X, y):
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

        valid_preds = model.predict(X_valid)
        auc_scores.append(roc_auc_score(y_valid, valid_preds))

    return np.mean(auc_scores)

# 3. Optunaで最適化の実行
print("Optunaによるハイパーパラメータ探索を開始します（数十回試行します）...")
study = optuna.create_study(direction='maximize') # AUCなので「最大化」を目指す
study.optimize(objective, n_trials=50) # 50パターンのパラメータを試す

# 4. 結果の表示
print("\n=== チューニング完了！ ===")
print(f"ベストCVスコア: {study.best_value:.4f}")
print("ベストパラメータ:")
for key, value in study.best_params.items():
    print(f"  '{key}': {value},")
