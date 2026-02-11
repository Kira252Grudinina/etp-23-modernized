"""
FastAPI REST API for sexism detection.

Endpoints:
- POST /predict - Single text prediction
- POST /predict/batch - Batch predictions
- GET /health - Health check
- GET / - API documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import time
from pathlib import Path

from ..models.classifier import SexismClassifier
from ..config import CHECKPOINTS_DIR


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Single text prediction request."""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to classify")
    return_probabilities: bool = Field(default=True, description="Include class probabilities")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    return_probabilities: bool = Field(default=True, description="Include class probabilities")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        if not all(text.strip() for text in v):
            raise ValueError('All texts must be non-empty')
        return v


class PredictionResponse(BaseModel):
    """Prediction response."""
    text: str
    label: str
    confidence: float
    probabilities: Optional[List[float]] = None
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="Sexism Detection API",
    description="REST API for detecting sexist language in text using RoBERTa",
    version="2.0.0",
    contact={
        "name": "Kira Grudinina",
        "email": "kira.grudinina7@etu.univ-lorraine.fr"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
classifier = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global classifier
    
    model_path = CHECKPOINTS_DIR / "roberta_base_binary.pt"
    
    try:
        if model_path.exists():
            print(f"Loading model from {model_path}...")
            classifier = SexismClassifier.from_pretrained(
                str(model_path),
                num_classes=2
            )
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model not found at {model_path}")
            print("API will run but predictions will fail until model is available")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will run but predictions will fail")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sexism Detection API",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        },
        "model": {
            "architecture": "RoBERTa-base",
            "performance": "F1: 0.827, Accuracy: 87.85%",
            "dataset": "EDOS (20,000 samples)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    import torch
    
    return HealthResponse(
        status="healthy" if classifier is not None else "model_not_loaded",
        model_loaded=classifier is not None,
        model_path=str(CHECKPOINTS_DIR / "roberta_base_binary.pt") if classifier else None,
        device=str(classifier.device) if classifier else "unknown"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict sexism for a single text.
    
    - **text**: Input text to classify (1-1000 characters)
    - **return_probabilities**: Include probability scores (default: true)
    
    Returns prediction with label, confidence, and optional probabilities.
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        result = classifier.predict(
            request.text,
            return_probs=request.return_probabilities
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResponse(
            text=result['text'],
            label=result['label'],
            confidence=result['confidence'],
            probabilities=result.get('probabilities') if request.return_probabilities else None,
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sexism for multiple texts in batch.
    
    - **texts**: List of texts to classify (1-100 texts)
    - **return_probabilities**: Include probability scores (default: true)
    
    Returns predictions for all texts with processing time.
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        results = classifier.predict(
            request.texts,
            return_probs=request.return_probabilities
        )
        
        predictions = []
        for result in results:
            predictions.append(PredictionResponse(
                text=result['text'],
                label=result['label'],
                confidence=result['confidence'],
                probabilities=result.get('probabilities') if request.return_probabilities else None,
                processing_time_ms=0  # Individual times not tracked in batch
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=round(total_time, 2),
            count=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Optional: Add example endpoint for demo
@app.get("/examples", tags=["Demo"])
async def get_examples():
    """Get example texts for testing."""
    return {
        "sexist_examples": [
            "Women should stay in the kitchen",
            "Girls are too emotional for leadership",
            "She's just a pretty face"
        ],
        "not_sexist_examples": [
            "Women are wonderful people",
            "Everyone deserves equal opportunities",
            "Respect people regardless of gender"
        ]
    }
