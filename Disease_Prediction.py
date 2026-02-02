import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


def load_disease_data(disease_type):
    if disease_type == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df, "Breast Cancer", "target"

    elif disease_type == "diabetes":
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = ['pregnancies','glucose','blood_pressure','skin_thickness',
                'insulin','bmi','diabetes_pedigree','age','target']
        df = pd.read_csv(url, names=cols)
        return df, "Diabetes", "target"

    elif disease_type == "heart_disease":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal','target']
        df = pd.read_csv(url, names=cols, na_values='?')
        df["target"] = (df["target"] > 0).astype(int)
        return df, "Heart Disease", "target"


def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    cat_cols = X.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c])

    if X.shape[1] > 10:
        selector = SelectKBest(f_classif, k=10)
        X = pd.DataFrame(selector.fit_transform(X, y),
                         columns=X.columns[selector.get_support()])

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, scaler, X.columns.tolist()

def train_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)

    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred),
            "f1": f1_score(yte, pred),
            "cv": cross_val_score(model, X, y, cv=5, scoring="f1").mean()
        }


    return results
