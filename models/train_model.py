"""
train_model.py
---------------
Trains the Disease Prediction Model (the "AI Used -> Disease Prediction
Model" component of the project) using scikit-learn.

Algorithm: RandomForestClassifier over a binary symptom-vector input.
Random Forest is used (instead of a single Decision Tree) because it
handles the correlated/overlapping symptom patterns across diseases
better and gives us feature_importances_ for explainability in the
project viva ("which symptoms mattered most").

Run:
    python models/train_model.py

Outputs:
    models/disease_model.pkl   -> trained sklearn Pipeline
    models/symptom_list.pkl    -> ordered list of the 30 symptom columns
    models/label_classes.pkl   -> disease class names (for reference)
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "models", "symptom_list.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "models", "label_classes.pkl")


def main():
    df = pd.read_csv(DATA_PATH)

    symptom_cols = [c for c in df.columns if c != "disease"]
    X = df[symptom_cols]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature importance (for viva / report explainability section)
    importances = pipeline.named_steps["clf"].feature_importances_
    top_features = sorted(zip(symptom_cols, importances), key=lambda x: -x[1])[:10]
    print("\nTop 10 most predictive symptoms:")
    for name, score in top_features:
        print(f"  {name:25s} {score:.4f}")

    # Persist artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(SYMPTOMS_PATH, "wb") as f:
        pickle.dump(symptom_cols, f)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(sorted(y.unique().tolist()), f)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved symptom list -> {SYMPTOMS_PATH}")


if __name__ == "__main__":
    main()
