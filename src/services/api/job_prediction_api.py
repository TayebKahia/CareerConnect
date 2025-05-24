"""
FastAPI implementation for job prediction using the Dual Ensemble model.
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modeling.predict_dual import get_predictor
# Initialize FastAPI app
app = FastAPI(
    title="Job Role Prediction API (Dual Ensemble)",
    description="API for predicting job roles using the Dual Ensemble (ML+NN) model.",
    version="2.0.0"
)

# Predictor singleton
predictor = get_predictor()

# Define request and response models
class TextInput(BaseModel):
    text: str
    top_k: Optional[int] = 3


class TechnologyInput(BaseModel):
    technologies: List[str]
    top_k: Optional[int] = 3


class PredictionDetail(BaseModel):
    role: str
    probability: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionDetail]
    extracted_technologies: List[str]
    processing_time: float


@app.get("/health")
def health_check():
    """Health check endpoint to verify if the API is running."""
    return {"status": "ok"}


@app.post("/predict/text", response_model=PredictionResponse)
def predict_from_text(input_data: TextInput):
    """
    Predict job roles from the provided text.
    
    Returns the top predictions and extracted technologies.
    """
    import time
    start_time = time.time()
    try:
        result = predictor.predict_from_text(input_data.text, top_k=input_data.top_k)
        processing_time = time.time() - start_time
        return {
            "predictions": result["ensemble_predictions"],
            "extracted_technologies": result["extracted_technologies"],
            "processing_time": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/technologies", response_model=PredictionResponse)
def predict_from_technologies(input_data: TechnologyInput):
    """
    Predict job roles from a list of technologies.
    
    Returns the top predictions.
    """
    import time
    start_time = time.time()
    try:
        result = predictor.predict_from_technologies(input_data.technologies, top_k=input_data.top_k)
        processing_time = time.time() - start_time
        return {
            "predictions": result["ensemble_predictions"],
            "extracted_technologies": result["extracted_technologies"],
            "processing_time": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For development/debugging
if __name__ == "__main__":
    import uvicorn
    
    # Run with hot reload during development
    uvicorn.run("job_prediction_api:app", host="0.0.0.0", port=8001, reload=True)
