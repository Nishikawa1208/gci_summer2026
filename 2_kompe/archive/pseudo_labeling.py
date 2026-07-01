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

print("特徴量を作成中...")
train_processed = create_features_v1(train)
test_processed = create_features_v1(test)

train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

# 3. Step 1: まずは通常通り学習して、テストデータを予測する
print("\n--- Step 1: 初期モデルによるテストデータの予測 ---")
X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# Optunaで見つけたパラメータ（元の5分割で最も良かった設定）
params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'verbose': -1, 'random_state': 42,
    'learning_rate': 0.03214991300054569, 'num_leaves': 88, 'max_depth': 5,
    'feature_fraction': 0.8076714735823799, 'bagging_fraction': 0.6070547458788909,
    'bagging_freq': 3, 'min_child_samples': 30,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
initial_test_preds = np.zeros(len(X_test))
initial_oof = np.zeros(len(X))

for train_idx, valid_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]

    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), valid_sets=[lgb.Dataset(X_va, y_va)],
                      num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
    initial_oof[valid_idx] = model.predict(X_va)
    initial_test_preds += model.predict(X_test) / 5

print(f"初期モデルの CV AUC: {roc_auc_score(y, initial_oof):.4f}")

# 4. Step 2: 自信がある予測結果を抽出して「擬似ラベル」を作る
# 閾値（90%以上は1、10%以下は0とみなす）
pseudo_indices_1 = np.where(initial_test_preds > 0.90)[0]
pseudo_indices_0 = np.where(initial_test_preds < 0.10)[0]

print(f"\n--- Step 2: 擬似ラベルの抽出 ---")
print(f"自信あり(1) と判定した件数: {len(pseudo_indices_1)} 件")
print(f"自信あり(0) と判定した件数: {len(pseudo_indices_0)} 件")

# テストデータから抽出したデータを訓練用データフレームに変換
pseudo_1_df = train_processed.iloc[:0].copy() # カラム構造をコピー
pseudo_0_df = train_processed.iloc[:0].copy()

# X_testの該当行を取り出し、ターゲット変数（1または0）を付与
temp_test_1 = test_processed.iloc[pseudo_indices_1].copy()
temp_test_1['target'] = 1
temp_test_0 = test_processed.iloc[pseudo_indices_0].copy()
temp_test_0['target'] = 0

# 元のTrainデータと結合して、データ増量！
augmented_train = pd.concat([train_processed, temp_test_1, temp_test_0], axis=0).reset_index(drop=True)
print(f"元のデータ数: {len(train_processed)} -> 擬似ラベル追加後: {len(augmented_train)}")

# 5. Step 3: 水増ししたデータで再学習
print("\n--- Step 3: 水増しデータによる最終モデルの学習 ---")
X_aug = augmented_train.drop(columns=['customer_id', 'target'])
y_aug = augmented_train['target']

final_test_preds = np.zeros(len(X_test))
skf_aug = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, valid_idx in skf_aug.split(X_aug, y_aug):
    X_tr, y_tr = X_aug.iloc[train_idx], y_aug.iloc[train_idx]
    X_va, y_va = X_aug.iloc[valid_idx], y_aug.iloc[valid_idx]

    final_model = lgb.train(params, lgb.Dataset(X_tr, y_tr), valid_sets=[lgb.Dataset(X_va, y_va)],
                            num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])

    final_test_preds += final_model.predict(X_test) / 5

# 提出ファイルの作成
submission = pd.DataFrame({
    'customer_id': test['customer_id'],
    'target': final_test_preds
})
submission.to_csv('submission_pseudo_label.csv', index=False)
print("\n提出ファイル 'submission_pseudo_label.csv' を作成しました！")
