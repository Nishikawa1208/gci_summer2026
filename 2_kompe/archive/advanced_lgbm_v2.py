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
    df = df.copy()

    # 欠損値の処理
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())

    # 【NEW】登録からの経過日数（より精緻なロイヤリティ指標）
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    # データ内の最新日を基準（現在）として、何日経過しているか計算
    base_date = pd.to_datetime('2026-01-01')
    df['days_since_registration'] = (base_date - df['registration_date']).dt.days
    df['reg_year'] = df['registration_date'].dt.year
    df = df.drop(columns=['registration_date'])

    # 年齢に関する特徴量
    df['age'] = 2026 - df['birth_year']

    # 消費額に関する特徴量
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    df['total_spend'] = df[spend_cols].sum(axis=1)
    df['necessity_spend'] = df[['spend_meat', 'spend_fish', 'spend_fruits']].sum(axis=1)
    df['luxury_spend'] = df[['spend_wines', 'spend_sweets', 'spend_gold']].sum(axis=1)
    df['luxury_ratio'] = df['luxury_spend'] / (df['total_spend'] + 1)

    # 購入チャネルに関する特徴量
    purchase_cols = ['web_purchases', 'catalog_purchases', 'store_purchases']
    df['total_purchases'] = df[purchase_cols].sum(axis=1)
    df['avg_spend_per_purchase'] = df['total_spend'] / (df['total_purchases'] + 1)

    # 【NEW】Webコンバージョン率（訪問したうち何回買ったか）
    df['web_conversion_rate'] = df['web_purchases'] / (df['monthly_web_visits'] + 1)

    # 【NEW】割引に惹かれる度合い（Deal Hunter指標）
    df['deal_ratio'] = df['deals_purchases'] / (df['total_purchases'] + 1)

    # 家族構成に関する特徴量
    df['total_children'] = df['num_children'] + df['num_teenagers']
    df['is_married'] = df['marital_status'].apply(lambda x: 1 if x in ['Married', 'Together'] else 0)
    df['family_size'] = 1 + df['is_married'] + df['total_children']
    df['income_per_person'] = df['annual_income'] / df['family_size']

    # 【NEW】カテゴリ変数の処理（LightGBMにそのまま渡すために 'category' 型に変換）
    # ※ pd.get_dummies() はもう使いません
    cat_cols = ['education_level', 'marital_status']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    return df

print("特徴量エンジニアリング（V2）を実行中...")
train_processed = create_features(train)
test_processed = create_features(test)

X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

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

# 予測と提出ファイルの作成
test_preds = np.zeros(len(test_processed))
for model in models:
    test_preds += model.predict(X_test) / folds

submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': test_preds})
submission.to_csv('submission_lgbm_v2.csv', index=False)
print("\n提出ファイル 'submission_lgbm_v2.csv' を作成しました！")
