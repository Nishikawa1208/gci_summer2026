import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 2. V3特徴量エンジニアリング（割合を全展開）
def create_features_v3(df):
    df = df.copy()

    # 欠損値
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())

    # 年代
    df['age'] = 2026 - df['birth_year']

    # 既存の基本特徴量
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    purchase_cols = ['web_purchases', 'catalog_purchases', 'store_purchases']

    df['total_spend'] = df[spend_cols].sum(axis=1)
    df['total_purchases'] = df[purchase_cols].sum(axis=1)
    df['avg_spend_per_purchase'] = df['total_spend'] / (df['total_purchases'] + 1)

    df['is_married'] = df['marital_status'].apply(lambda x: 1 if x in ['Married', 'Together'] else 0)
    df['total_children'] = df['num_children'] + df['num_teenagers']
    df['family_size'] = 1 + df['is_married'] + df['total_children']
    df['income_per_person'] = df['annual_income'] / df['family_size']

    # 各カテゴリの消費割合（Share of Wallet）
    for col in spend_cols:
        df[f'{col}_ratio'] = df[col] / (df['total_spend'] + 1)

    # 各チャネルの利用割合
    for col in purchase_cols:
        df[f'{col}_ratio'] = df[col] / (df['total_purchases'] + 1)

    # 不要な日付を落としてダミー化
    df = df.drop(columns=['registration_date'])
    df = pd.get_dummies(df, columns=['education_level', 'marital_status'], drop_first=True)

    return df

print("V3特徴量を作成中...")
train_processed = create_features_v3(train)
test_processed = create_features_v3(test)

train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# 3. アンサンブル評価の設定
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

# 各モデルの予測を保存する配列（OOF: Out Of Fold予測）
lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))

# 手動で見つけた強めのパラメータ
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03,
    'num_leaves': 64, 'max_depth': 6, 'feature_fraction': 0.8,
    'random_state': 42, 'verbose': -1
}

xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.03,
    'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'verbosity': 0
}

print(f"\n--- LightGBM & XGBoost {folds}分割交差検証を開始 ---")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    # 【修正箇所】valid_index を valid_idx に修正しました
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # --- LightGBMの学習 ---
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    lgb_model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_valid],
                          num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])

    lgb_oof[valid_idx] = lgb_model.predict(X_valid)
    lgb_test_preds += lgb_model.predict(X_test) / folds

    # --- XGBoostの学習 ---
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
    xgb_model = xgb.train(xgb_params, xgb_train, evals=[(xgb_valid, 'valid')],
                          num_boost_round=1000, early_stopping_rounds=50, verbose_eval=False)

    xgb_oof[valid_idx] = xgb_model.predict(xgb.DMatrix(X_valid))
    xgb_test_preds += xgb_model.predict(xgb.DMatrix(X_test)) / folds

    print(f"Fold {fold+1} 完了")

# 4. スコアの計算
lgb_auc = roc_auc_score(y, lgb_oof)
xgb_auc = roc_auc_score(y, xgb_oof)

# 両方の予測を50%ずつ混ぜる（アンサンブル）
ensemble_oof = (lgb_oof * 0.5) + (xgb_oof * 0.5)
ensemble_auc = roc_auc_score(y, ensemble_oof)

print("\n=== クロスバリデーション結果 ===")
print(f"LightGBM 単体 AUC : {lgb_auc:.4f}")
print(f"XGBoost  単体 AUC : {xgb_auc:.4f}")
print(f"アンサンブル  AUC : {ensemble_auc:.4f}  <-- ここが重要！")

# 5. 提出ファイルの作成
ensemble_test_preds = (lgb_test_preds * 0.5) + (xgb_test_preds * 0.5)
submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': ensemble_test_preds})
submission.to_csv('submission_ensemble_v3.csv', index=False)
print("\n提出ファイル 'submission_ensemble_v3.csv' を作成しました！")
