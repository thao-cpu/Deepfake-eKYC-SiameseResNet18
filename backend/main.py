from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid
import random
import time

# Import các schema vừa định nghĩa
from schemas import HealthResponse, PredictResponse

# Khởi tạo app FastAPI
app = FastAPI(
    title="Deepfake Detection API",
    description="API hệ thống eKYC - Mockup chưa tích hợp model AI",
    version="1.0.0"
)

# /health
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():

    return HealthResponse(
        status="OK",
        message="Backend FastAPI đang hoạt động bình thường.",
        version="1.0.0"
    )

# /predict
@app.post("/predict", response_model=PredictResponse, tags=["AI Inference"])
async def predict_deepfake(file: UploadFile = File(...)):
    
    # Chấp nhận mọi định dạng bắt đầu bằng "image/" hoặc "video/"
    if not (file.content_type.startswith("image/") or file.content_type.startswith("video/")):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file định dạng ảnh (JPG, PNG...) hoặc video (MP4, AVI...).")
    
    # Tạo dữ liệu Mock
    mock_score = random.uniform(0.0, 1.0)
    is_fake = mock_score > 0.5
    
    # Trả về kết quả theo đúng chuẩn Schema đã chốt
    return PredictResponse(
        session_id=str(uuid.uuid4()), # Sinh ID phiên ngẫu nhiên
        filename=file.filename,
        is_deepfake=is_fake,
        confidence_score=round(mock_score, 4),
        message="[MOCK] Phát hiện giả mạo!" if is_fake else "[MOCK] Khuôn mặt hợp lệ."
    )