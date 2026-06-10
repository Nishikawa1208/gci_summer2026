import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

# データ読み込み
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 特徴量
for df in [train, test]:
    df["Age_missing"] = df["Age"].isnull().astype(int)
    df["Sprint_missing"] = df["Sprint_40yd"].isnull().astype(int)
    df["Vertical_missing"] = df["Vertical_Jump"].isnull().astype(int)
    df["Bench_missing"] = df["Bench_Press_Reps"].isnull().astype(int)
    df["Broad_missing"] = df["Broad_Jump"].isnull().astype(int)
    df["Agility_missing"] = df["Agility_3cone"].isnull().astype(int)
    df["Shuttle_missing"] = df["Shuttle"].isnull().astype(int)

# BMI（簡易）
for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# 頻度エンコーディング
cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
for col in cat_cols:
    freq = train[col].value_counts(dropna=False)
    freq_map = (freq / len(train)).to_dict()
    train[col + '_freq'] = train[col].map(freq_map).fillna(0)
    test[col + '_freq'] = test[col].map(freq_map).fillna(0)

# 目的変数と特徴量
y = train['Drafted']
X = train.drop(columns=['Drafted', 'Id'])
X_test = test.drop(columns=['Id'])

numeric_features = [
    'Year','Age','Height','Weight','Sprint_40yd','Vertical_Jump','Bench_Press_Reps',
    'Broad_Jump','Agility_3cone','Shuttle','Age_missing','Sprint_missing','Vertical_missing',
    'Bench_missing','Broad_missing','Agility_missing','Shuttle_missing','BMI',
    'School_freq','Player_Type_freq','Position_Type_freq','Position_freq'
]

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric_features)
])

clf = Pipeline([
    ('pre', preprocessor),
    ('est', ExtraTreesClassifier(
        n_estimators=2000,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print('AUC improved test:', scores)
print('Mean AUC:', scores.mean())
