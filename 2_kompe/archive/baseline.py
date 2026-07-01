import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 2. 前処理（欠損値の補完）
# annual_incomeの欠損値を、全体の「中央値」で埋めます
median_income = train['annual_income'].median()
train['annual_income'] = train['annual_income'].fillna(median_income)
test['annual_income'] = test['annual_income'].fillna(median_income)

# 3. 前処理（日付と文字列データの数値化）
def preprocess(df):
    df = df.copy()
    # registration_dateを日付型に変換し、登録「年」と「月」の数値データとして抽出
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['reg_year'] = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month

    # 元の文字列のカラムは削除
    df = df.drop(columns=['registration_date'])

    # カテゴリ変数（文字列）をダミー変数（0と1のフラグ）に変換（One-Hot Encoding）
    df = pd.get_dummies(df, columns=['education_level', 'marital_status'], drop_first=True)
    return df

train_processed = preprocess(train)
test_processed = preprocess(test)

# TrainとTestでカラム（列）を揃える処理
train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target']) # Testにはtargetがないので合わせる過程で出来たものを消す

# 4. モデルの学習
# 学習に使う特徴量（customer_idとtarget以外）と、予測したい正解ラベル（target）を分ける
X_train = train_processed.drop(columns=['customer_id', 'target'])
y_train = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# ランダムフォレストモデルの定義（コンペルールの「再現性の確保」のため random_state=42 を指定）
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 学習の実行
print("モデルを学習しています...")
model.fit(X_train, y_train)

# 5. 予測と提出ファイルの作成
# コンペの評価指標が「AUC」なので、0か1かのラベルではなく「1になる確率（0〜1）」を予測します
print("テストデータの予測を行っています...")
preds = model.predict_proba(X_test)[:, 1]

# 提出用データフレームの作成
submission = pd.DataFrame({
    'customer_id': test['customer_id'],
    'target': preds
})

# CSVファイルとして出力
submission.to_csv('submission.csv', index=False)
print("提出ファイル 'submission.csv' を作成しました！")
