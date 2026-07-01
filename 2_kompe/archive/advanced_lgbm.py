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

def create_features(df):
    """強力な特徴量エンジニアリングを行う関数"""
    df = df.copy()

    # 【1】欠損値の処理
    # ※ 本来はtrainの中央値等を使うのが厳密ですが、ここでは簡易的に処理します
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())

    # 【2】日付系の特徴量
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['reg_year'] = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month
    df = df.drop(columns=['registration_date'])

    # 【3】年齢に関する特徴量（現在は2026年想定）
    df['age'] = 2026 - df['birth_year']

    # 【4】購買行動（消費額）に関する特徴量
    # 全カテゴリの合計消費額
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    df['total_spend'] = df[spend_cols].sum(axis=1)

    # 必需品（肉・魚・果物）と嗜好品（ワイン・菓子・金）の割合
    df['necessity_spend'] = df[['spend_meat', 'spend_fish', 'spend_fruits']].sum(axis=1)
    df['luxury_spend'] = df[['spend_wines', 'spend_sweets', 'spend_gold']].sum(axis=1)
    df['luxury_ratio'] = df['luxury_spend'] / (df['total_spend'] + 1) # 0割り防止

    # 【5】購買チャネル（回数）に関する特徴量
    # 合計購入回数
    purchase_cols = ['web_purchases', 'catalog_purchases', 'store_purchases']
    df['total_purchases'] = df[purchase_cols].sum(axis=1)

    # 1回あたりの平均消費額（顧客単価）
    df['avg_spend_per_purchase'] = df['total_spend'] / (df['total_purchases'] + 1)

    # 【6】家族構成に関する特徴量
    df['total_children'] = df['num_children'] + df['num_teenagers']
    df['is_parent'] = (df['total_children'] > 0).astype(int)

    # 1人あたりの世帯年収（ざっくりとした生活水準の指標）
    # 配偶者なし=1, あり=2 として世帯人数を概算
    df['is_married'] = df['marital_status'].apply(lambda x: 1 if x in ['Married', 'Together'] else 0)
    df['family_size'] = 1 + df['is_married'] + df['total_children']
    df['income_per_person'] = df['annual_income'] / df['family_size']

    # 【7】カテゴリ変数のダミー化（One-Hot Encoding）
    df = pd.get_dummies(df, columns=['education_level', 'marital_status'], drop_first=True)

    return df

# 特徴量の作成
print("特徴量エンジニアリングを実行中...")
train_processed = create_features(train)
test_processed = create_features(test)

# TrainとTestでカラムを揃える
train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

# 3. 特徴量とターゲットの分割
X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']

# 4. クロスバリデーションの実行
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

auc_scores = []
models = []

print(f"--- LightGBMによる {folds}分割交差検証を開始します ---")

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # LightGBMのパラメータ（少しチューニングを入れています）
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    valid_preds = model.predict(X_valid)
    auc = roc_auc_score(y_valid, valid_preds)
    auc_scores.append(auc)
    models.append(model)

    print(f"Fold {fold + 1} AUC: {auc:.4f}")

print("-" * 30)
print(f"平均 AUC (CV Score): {np.mean(auc_scores):.4f}")

# ==========================================
# 5. テストデータに対する予測（アンサンブル）と提出ファイルの作成
# ==========================================
print("\nテストデータの予測を行っています...")
# 5回のCVで学習した5つのモデルそれぞれの予測値の「平均」をとります（アンサンブル学習）
test_preds = np.zeros(len(test_processed))
X_test = test_processed.drop(columns=['customer_id'])

for model in models:
    test_preds += model.predict(X_test) / folds

# 提出用データフレームの作成
submission = pd.DataFrame({
    'customer_id': test['customer_id'],
    'target': test_preds
})
submission.to_csv('submission_lgbm_feat.csv', index=False)
print("提出ファイル 'submission_lgbm_feat.csv' を作成しました！")

# ==========================================
# 6. 特徴量の重要度（Feature Importance）の確認
# ==========================================
print("\n--- 特徴量の重要度 トップ15 ---")
importance_df = pd.DataFrame()

# 5つのモデルの特徴量重要度を収集
for i, model in enumerate(models):
    fold_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain'), # gain: 分岐による精度向上の貢献度
        'fold': i + 1
    })
    importance_df = pd.concat([importance_df, fold_importance], axis=0)

# 全Foldの平均をとって、重要度が高い順に並び替え
mean_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

# 上位15個の表示
for i, row in mean_importance.head(15).iterrows():
    print(f"{i+1:2d}位: {row['feature']:<25} ({row['importance']:.1f})")
