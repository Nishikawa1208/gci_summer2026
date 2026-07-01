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

# 2. V1ベースの特徴量（ターゲットエンコーディングの準備のため、ダミー化はまだしない）
def create_base_features(df):
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

    df = df.drop(columns=['registration_date']) # 今回は使わない
    return df

print("ベース特徴量を作成中...")
train_base = create_base_features(train)
test_base = create_base_features(test)

X = train_base.drop(columns=['customer_id', 'target'])
y = train_base['target']
X_test = test_base.drop(columns=['customer_id'])

# 3. K-Fold CV & ターゲットエンコーディングの実行
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

# CVごとの予測を保存
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# Optunaで見つけたベストパラメータ
params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'random_state': 42, 'verbose': -1,
    'learning_rate': 0.03214991300054569,
    'num_leaves': 88,
    'max_depth': 5,
    'feature_fraction': 0.8076714735823799,
    'bagging_fraction': 0.6070547458788909,
    'bagging_freq': 3,
    'min_child_samples': 30,
}

print(f"\n--- ターゲットエンコーディング + LightGBM {folds}分割交差検証 ---")

# ターゲットエンコーディングを行うカラム
te_cols = ['education_level', 'marital_status']

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx].copy(), y.iloc[valid_idx]
    X_test_fold = X_test.copy()

    # 【ターゲットエンコーディング処理】
    # ※ データリーク（未来の情報をカンニングすること）を防ぐため、
    # 必ず「学習データ（train_idx）のみ」を使って平均値を計算し、それを検証データとテストデータに適用します。
    for col in te_cols:
        # 学習データのカテゴリごとの平均値（反応率）を計算
        target_mean = y_train.groupby(X_train[col]).mean()

        # 学習データ、検証データ、テストデータに適用
        X_train[col + '_te'] = X_train[col].map(target_mean)
        X_valid[col + '_te'] = X_valid[col].map(target_mean)
        X_test_fold[col + '_te'] = X_test_fold[col].map(target_mean)

        # マッピングできなかった場合（テストデータにしかないカテゴリなど）は全体の平均で埋める
        overall_mean = y_train.mean()
        X_valid[col + '_te'] = X_valid[col + '_te'].fillna(overall_mean)
        X_test_fold[col + '_te'] = X_test_fold[col + '_te'].fillna(overall_mean)

    # 元のカテゴリ変数（文字列）は削除
    X_train = X_train.drop(columns=te_cols)
    X_valid = X_valid.drop(columns=te_cols)
    X_test_fold = X_test_fold.drop(columns=te_cols)

    # --- LightGBMの学習 ---
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    model = lgb.train(params, lgb_train, valid_sets=[lgb_valid],
                      num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])

    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test_fold) / folds
    print(f"Fold {fold+1} 完了")

cv_auc = roc_auc_score(y, oof_preds)
print("\n=== 結果 ===")
print(f"CV Score (AUC): {cv_auc:.4f}")

submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': test_preds})
submission.to_csv('submission_te.csv', index=False)
print("提出ファイル 'submission_te.csv' を作成しました！")
