import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from processing.preprocessing_utils import preprocess_dataframe

def train_model(df: pd.DataFrame):

    if "cardio" not in df.columns:
        raise ValueError("Expected a 'cardio' column in the data")

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            "C": [0.1, 1, 10]
        }),
        "RandomForest": (RandomForestClassifier(random_state=42), {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1]
        }),
        "SVM": (SVC(probability=True), {  # ✅ SVM بسرعة
            "C": [1],  
            "kernel": ["rbf"],  
            "gamma": ["scale"]  
        })
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    for name, (model, params) in models.items():
        print(f"\n🔍 Training: {name}")
        cv_splits = 3 if name == "SVM" else 5  # ✅ SVM أقل عشان السرعة
        grid = GridSearchCV(
            model, params,
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42),
            scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
        print("📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("📌 Classification Report:\n", classification_report(y_test, y_pred))

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = grid.best_estimator_
            best_model_name = name

    print(f"\n✅ Best Model: {best_model_name} with ROC-AUC: {best_score:.4f}")

    # Save the best model
    model_path = os.path.join("modeling", "heart_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Save feature names
    feature_path = os.path.join("modeling", "features.pkl")
    joblib.dump(X.columns.tolist(), feature_path)
    print(f"💾 Features saved to: {feature_path}")

    return best_model
