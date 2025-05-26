from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import time

# Add the parent directory to sys.path to be able to import ConceptMatcher
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from features import ConceptMatcher

app = FastAPI(
    title="ConceptMatcher API",
    description="API for detecting technology concepts in text",
    version="1.0.0",
)


# Define detailed models for the response
class PhraseMatch(BaseModel):
    text: str
    score: float


class ThresholdInfo(BaseModel):
    ngram_threshold: float
    filter_similarity_threshold: float


class TechnologyDetail(BaseModel):
    name: str
    similarity_score: float
    thresholds: ThresholdInfo
    type: str
    phrases: List[PhraseMatch]


# Define request and response models
class TextInput(BaseModel):
    text: str


class TechnologyResponse(BaseModel):
    technologies: List[TechnologyDetail]
    processing_time: float


# For backward compatibility, add a simple response model option
class SimpleTechnologyResponse(BaseModel):
    technologies: List[str]
    processing_time: float


# Global variables for the ConceptMatcher singleton
concept_matcher = None
# Calculate the absolute path to the CSV file based on the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
csv_path = os.path.join(
    project_root, "data", "processed", "technologies_with_abbreviations.csv"
)


@app.on_event("startup")
async def startup_event():
    """Initialize the ConceptMatcher singleton when the API starts."""
    global concept_matcher
    try:
        concept_matcher = ConceptMatcher(csv_path=csv_path)
        concept_matcher.load_concepts()
        concept_matcher.generate_concept_embeddings(load_if_exists=True)
    except Exception as e:
        print(f"Error initializing ConceptMatcher: {e}")
        # Continue without failing, but log error
        # In production, you might want to raise a startup exception


@app.get("/health")
async def health_check():
    """Health check endpoint to verify if the API is running."""
    return {
        "status": "ok",
        "model": concept_matcher.model_name if concept_matcher else None,
    }


@app.post("/api/recognize-technologies", response_model=TechnologyResponse)
async def recognize_technologies(input_data: TextInput):
    """
    Extract technology concepts from the provided text.

    Returns detailed information including similarity scores and thresholds.
    """
    if not concept_matcher:
        raise HTTPException(status_code=500, detail="ConceptMatcher is not initialized")

    start_time = time.time()

    # Process text to recognize technologies with detailed information
    try:
        technologies = concept_matcher.process_text(input_data.text)
        processing_time = time.time() - start_time

        return {"technologies": technologies, "processing_time": processing_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.post("/api/recognize-technologies-simple", response_model=SimpleTechnologyResponse)
async def recognize_technologies_simple(input_data: TextInput):
    """
    Extract technology concepts from the provided text.

    Returns a simplified response with just the technology names.
    """
    if not concept_matcher:
        raise HTTPException(status_code=500, detail="ConceptMatcher is not initialized")

    start_time = time.time()

    # Process text to recognize technologies
    try:
        detailed_technologies = concept_matcher.process_text(input_data.text)
        # Extract just the names for the simple response
        technology_names = [tech["name"] for tech in detailed_technologies]
        processing_time = time.time() - start_time

        return {"technologies": technology_names, "processing_time": processing_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


# For development/debugging
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
