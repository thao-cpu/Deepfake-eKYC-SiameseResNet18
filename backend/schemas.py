from pydantic import BaseModel
from typing import Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class PredictResponse(BaseModel):
    session_id: str
    filename: str
    file_type: str 
    is_deepfake: bool
    confidence_score: float
    aggregation_stats: Optional[Dict[str, Any]] = None 
    message: str
    error: Optional[str] = None