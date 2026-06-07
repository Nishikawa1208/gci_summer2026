import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 目的変数
y = train["Drafted"]

# Idは特徴量から除外
X = train.drop(columns=["Drafted", "Id"])
X_test = test.drop(columns=["Id"])

numeric_features = [
    "Year",
    "Age",
    "Height",
    "Weight",
    "Sprint_40yd",
    "Vertical_Jump",
    "Bench_Press_Reps",
    "Broad_Jump",
    "Agility_3cone",
    "Shuttle",
]

categorical_features = [
    "School",
    "Player_Type",
    "Position_Type",
    "Position",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

clf.fit(X, y)

pred = clf.predict(X_test)

submission = pd.DataFrame({
    "Id": test["Id"],
    "Drafted": pred
})

submission.to_csv("submission.csv", index=False)

print("submission.csv を作成しました")
