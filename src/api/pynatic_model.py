from pydantic import BaseModel
# from typing import List


class PredictionRequest(BaseModel):
    """
    Pydantic model for prediction request data.
    Adjust feature names to match Transformedfinal_data.csv
    (excluding RFM columns).
    """
    TransactionId: str
    BatchId: str
    Amount: float
    ProductCategory: str
    TransactionHour: float
    # Add other features from Transformedfinal_data.csv as needed


class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response.
    """
    probability: float
    is_high_risk: int
