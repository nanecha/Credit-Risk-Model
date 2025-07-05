from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.api.pydantic_models import PredictionRequest, PredictionResponse

# Initialize FastAPI app
app = FastAPI(title="Credit Risk Prediction API", version="1.0")

# Load environment variables
load_dotenv()

# Load model from MLflow Model Registry
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "GradientBoosting")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    raise Exception(f"Failed to load model from MLflow: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk probability for a customer.

    Parameters:
    - request: PredictionRequest containing customer features

    Returns:
    - PredictionResponse with risk probability and high-risk status
    """
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.model_dump()])

        # Predict probability
        prob = model.predict_proba(data)[:, 1][0]

        # Determine high-risk status (threshold = 0.5)
        is_high_risk = int(prob >= 0.5)

        return PredictionResponse(
            probability=prob,
            is_high_risk=is_high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
