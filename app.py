from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.pkl"

EXPECTED_FEATURES = [
    "subscription_type",
    "payment_plan",
    "num_subscription_pauses",
    "customer_service_inquiries",
    "weekly_hours",
    "song_skip_rate",
    "weekly_songs_played",
    "notifications_clicked",
]

app = FastAPI(
    title="Smart Music Streaming Churn API",
    description="Predicts subscriber churn risk using eight subscriber and engagement features.",
    version="2.0.0",
)

bundle = None
if MODEL_PATH.exists():
    bundle = joblib.load(MODEL_PATH)


class SubscriberProfile(BaseModel):
    features: dict[str, Any]


@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "Smart Music Streaming Churn API is active.",
        "model_loaded": bundle is not None,
        "prediction_input_count": len(EXPECTED_FEATURES),
    }


@app.get("/metadata")
def metadata():
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run python train.py first.")

    return {
        "feature_columns": bundle["feature_columns"],
        "numeric_features": bundle["numeric_features"],
        "categorical_features": bundle["categorical_features"],
        "model_name": bundle["model_name"],
        "validation_roc_auc": bundle["validation_roc_auc"],
    }


@app.post("/predict")
def predict_churn(profile: SubscriberProfile):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run python train.py first.")

    try:
        columns = bundle["feature_columns"]
        if set(columns) != set(EXPECTED_FEATURES):
            raise HTTPException(
                status_code=503,
                detail="The saved model does not use the expected 8 features. Run python train.py again.",
            )

        missing = [col for col in EXPECTED_FEATURES if col not in profile.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required input fields: {missing}",
            )

        # Keep the model input in the exact order used during training.
        row = {col: profile.features[col] for col in columns}
        frame = pd.DataFrame([row], columns=columns)
        probability = float(bundle["model"].predict_proba(frame)[0][1])
        risk_pct = round(probability * 100, 2)

        if risk_pct >= 70:
            level = "HIGH RISK"
            recommendation = (
                "Consider a personal retention check-in and review recent engagement "
                "or support issues."
            )
        elif risk_pct >= 35:
            level = "MEDIUM RISK"
            recommendation = (
                "Monitor engagement and consider a relevant playlist, discovery, "
                "or feedback campaign."
            )
        else:
            level = "LOW RISK"
            recommendation = (
                "Continue regular engagement; consider suitable playlist discovery "
                "or plan benefits."
            )

        return {
            "churn_probability": round(probability, 4),
            "risk_score_percentage": risk_pct,
            "risk_level": level,
            "recommendation": recommendation,
            "model_name": bundle["model_name"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score profile: {exc}") from exc
