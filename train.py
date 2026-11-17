from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
TARGET = "churned"

# The model is deliberately trained on only these 8 dashboard inputs.
FEATURE_COLUMNS = [
    "subscription_type",
    "payment_plan",
    "num_subscription_pauses",
    "customer_service_inquiries",
    "weekly_hours",
    "song_skip_rate",
    "weekly_songs_played",
    "notifications_clicked",
]


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Copy train.csv into the project's data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found: {list(df.columns)}")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    df = df.drop_duplicates().copy()
    df = df[df[TARGET].notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError("The 'churned' target must contain only 0 and 1.")
    if y.nunique() != 2:
        raise ValueError("The target must contain both classes: 0 (stayed) and 1 (churned).")

    # Explicitly select only the 8 features; no other dataset columns enter training.
    X = df[FEATURE_COLUMNS].copy()
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("string")

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }

    results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        probabilities = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, probabilities)
        print(f"\n===== {name} (8 features) =====")
        print(classification_report(y_valid, predictions, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_valid, predictions))
        print(f"ROC-AUC: {auc:.4f}")
        results.append((auc, name, model))

    results.sort(key=lambda item: item[0], reverse=True)
    best_auc, best_name, best_model = results[0]

    # Refit the selected model on all labelled rows, still using only 8 features.
    best_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET,
        "model_name": best_name,
        "validation_roc_auc": float(best_auc),
        "class_counts": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "training_rows": int(len(df)),
        "excluded_columns": sorted(set(df.columns) - set(FEATURE_COLUMNS) - {TARGET}),
    }
    joblib.dump(bundle, MODEL_DIR / "churn_model.pkl")
    print(f"\nSaved model bundle to {MODEL_DIR / 'churn_model.pkl'}")
    print(f"Selected model: {best_name} | validation ROC-AUC: {best_auc:.4f}")
    print("Prediction feature count: 8")
    print("Validation metrics are from the holdout split; the saved model is refitted on all labelled rows.")


if __name__ == "__main__":
    main()
