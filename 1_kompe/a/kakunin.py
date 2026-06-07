import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv("data/train.csv")

y = train["Drafted"]
X = train.drop(columns=["Drafted", "Id"])

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

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat",
     Pipeline([
         ("imputer", SimpleImputer(strategy="most_frequent")),
         ("encoder", OneHotEncoder(handle_unknown="ignore"))
     ]),
     categorical_features)
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=500,
        random_state=42
    ))
])

scores = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

print("各fold AUC:", scores)
print("平均AUC:", scores.mean())
print(train["Drafted"].value_counts())
print(train["Drafted"].value_counts(normalize=True))
