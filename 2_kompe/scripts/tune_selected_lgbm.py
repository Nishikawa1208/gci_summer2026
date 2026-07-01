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

# 2. V1特徴量の作成
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
X_all = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']

# 3. 前回の結果に基づき、重要度を計算して上位24個を抽出
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
temp_params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'random_state': 42}
feature_importances = np.zeros(X_all.shape[1])

for train_idx, valid_idx in skf.split(X_all, y):
    model = lgb.train(temp_params, lgb.Dataset(X_all.iloc[train_idx], y.iloc[train_idx]),
                      valid_sets=[lgb.Dataset(X_all.iloc[valid_idx], y.iloc[valid_idx])],
                      num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
    feature_importances += model.feature_importance(importance_type='gain')

imp_df = pd.DataFrame({'feature': X_all.columns, 'importance': feature_importances})
# 上位24個の特徴量名を取得
top_24_features = imp_df.sort_values('importance', ascending=False).head(24)['feature'].tolist()

# データを精鋭24個に絞る
X_selected = X_all[top_24_features]
print(f"絞り込み完了。選ばれた24個のデータでOptuna最適化を開始します！\n")

# 4. Optunaによる再チューニング
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10), # 変数が減ったので少し浅めも探索
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    auc_scores = []
    for train_idx, valid_idx in skf.split(X_selected, y):
        X_tr, y_tr = X_selected.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_selected.iloc[valid_idx], y.iloc[valid_idx]

        model = lgb.train(params, lgb.Dataset(X_tr, y_tr), valid_sets=[lgb.Dataset(X_va, y_va)],
                          num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
        auc_scores.append(roc_auc_score(y_va, model.predict(X_va)))

    return np.mean(auc_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("\n=== 再チューニング完了！ ===")
print(f"ベストCVスコア: {study.best_value:.4f}")
print("ベストパラメータ:")
for key, value in study.best_params.items():
    print(f"  '{key}': {value},")
