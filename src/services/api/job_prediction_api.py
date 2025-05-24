"""
FastAPI implementation for job prediction using the Hybrid Ensemble model.
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import ensemble model
from modeling.hybrid_ensemble import HybridEnsembleClassifier

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Initialize FastAPI app
app = FastAPI(
    title="Job Role Prediction API",
    description="API for predicting job roles based on technologies using a Hybrid Ensemble model",
    version="1.0.0"
)

# Define request and response models
class TextInput(BaseModel):
    text: str


class TechnologyInput(BaseModel):
    technologies: List[str]


class PredictionDetail(BaseModel):
    role: str
    probability: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionDetail]
    extracted_technologies: List[str]
    processing_time: float


class DetailedPredictionResponse(BaseModel):
    ensemble_predictions: List[PredictionDetail]
    extracted_technologies: List[str]
    component_predictions: Dict[str, Any]
    ensemble_weights: Dict[str, float]
    processing_time: float


# Global variable for the ensemble model
ensemble_model = None


@app.on_event("startup")
async def startup_event():
    """Initialize the Hybrid Ensemble model when the API starts."""
    global ensemble_model
    try:
        # Path to saved model
        model_path = os.path.join(MODEL_DIR, 'hybrid_ensemble')
        
        # Check if model exists
        if os.path.exists(model_path):
            print(f"Loading ensemble model from {model_path}")
            ensemble_model = HybridEnsembleClassifier()
            ensemble_model.load_model(model_path)
        else:
            print("Ensemble model not found. Initializing new model.")
            # Initialize a new model, but don't train it yet
            ensemble_model = HybridEnsembleClassifier(
                concept_matcher_path=os.path.join(DATA_PROCESSED_DIR, 'technologies_with_abbreviations.csv')
            )
    except Exception as e:
        print(f"Error initializing Hybrid Ensemble model: {e}")
        # Continue without failing, but log error
        # In production, you might want to raise a startup exception


@app.get("/health")
async def health_check():
    """Health check endpoint to verify if the API is running."""
    global ensemble_model
    
    model_status = "loaded" if ensemble_model is not None else "not_loaded"
    technologies_extracted = None
    
    # Try to extract technologies from a simple text to check if ConceptMatcher is working
    if ensemble_model is not None and ensemble_model.concept_matcher is not None:
        try:
            technologies_extracted = ensemble_model.extract_technologies("Python JavaScript")
        except Exception as e:
            technologies_extracted = f"Error: {str(e)}"
    
    return {
        "status": "ok",
        "model_status": model_status,
        "technologies_extracted": technologies_extracted
    }


@app.post("/api/predict-job-role", response_model=PredictionResponse)
async def predict_job_role(input_data: TextInput):
    """
    Predict job roles from the provided text.
    
    Returns the top predictions and extracted technologies.
    """
    global ensemble_model
    
    if ensemble_model is None:
        raise HTTPException(status_code=500, detail="Ensemble model is not initialized")
    
    start_time = time.time()
    
    try:
        # Make prediction
        result = ensemble_model.predict(input_text=input_data.text, top_k=3)
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "predictions": result["ensemble_predictions"],
            "extracted_technologies": result["extracted_technologies"],
            "processing_time": processing_time
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting job role: {str(e)}")


@app.post("/api/predict-job-role-from-technologies", response_model=PredictionResponse)
async def predict_job_role_from_technologies(input_data: TechnologyInput):
    """
    Predict job roles from a list of technologies.
    
    Returns the top predictions.
    """
    global ensemble_model
    
    if ensemble_model is None:
        raise HTTPException(status_code=500, detail="Ensemble model is not initialized")
    
    start_time = time.time()
    
    try:
        # Make prediction
        result = ensemble_model.predict(technologies_list=input_data.technologies, top_k=3)
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "predictions": result["ensemble_predictions"],
            "extracted_technologies": result["extracted_technologies"],
            "processing_time": processing_time
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting job role: {str(e)}")


@app.post("/api/predict-job-role-detailed", response_model=DetailedPredictionResponse)
async def predict_job_role_detailed(input_data: TextInput):
    """
    Predict job roles from the provided text with detailed component predictions.
    
    Returns predictions from all component models and ensemble weights.
    """
    global ensemble_model
    
    if ensemble_model is None:
        raise HTTPException(status_code=500, detail="Ensemble model is not initialized")
    
    start_time = time.time()
    
    try:
        # Make prediction
        result = ensemble_model.predict(input_text=input_data.text, top_k=3)
        processing_time = time.time() - start_time
        
        # Add processing time to result
        result["processing_time"] = processing_time
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting job role: {str(e)}")


# For development/debugging
if __name__ == "__main__":
    import uvicorn
    
    # Run with hot reload during development
    uvicorn.run("job_prediction_api:app", host="0.0.0.0", port=8001, reload=True)
