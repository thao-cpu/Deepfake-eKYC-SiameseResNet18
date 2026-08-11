from pydantic import BaseModel
from typing import Optional

# Output Schema cho endpoint /health
class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

# Output Schema cho endpoint /predict
class PredictResponse(BaseModel):
    session_id: str
    filename: str
    is_deepfake: bool
    confidence_score: float
    message: str
    error: Optional[str] = None



from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class HealthResponse(BaseModel):
    status: str
    message: str

class PredictResponse(BaseModel):
    session_id: str
    filename: str
    file_type: str # image hoặc video
    is_deepfake: bool
    confidence_score: float
    
    # Dành riêng cho video aggregation
    aggregation_stats: Optional[Dict[str, Any]] = None 
    
    message: str
    error: Optional[str] = None