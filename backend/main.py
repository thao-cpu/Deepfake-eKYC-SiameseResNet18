from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import asyncio
import time

from schemas import HealthResponse, PredictResponse
from inference import DeepfakeEngine

engine = None

# Lifespan load model 1 lần duy nhất lúc startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Sẽ tự động Mock nếu không có file này trong thư mục models/
    engine = DeepfakeEngine("models/best_model.pth")
    yield
    engine = None

app = FastAPI(
    title="eKYC Deepfake Backend API",
    description="Hệ thống phát hiện giả mạo khuôn mặt (Phase 3)",
    version="1.0.0",
    lifespan=lifespan
)

# KẾT NỐI FRONTEND (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="OK", 
        message="API đang chạy, model/mock đã được nạp sẵn sàng.",
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse, tags=["AI Inference"])
async def predict_deepfake(file: UploadFile = File(...)):
    start_time = time.time()
    session_id = str(uuid.uuid4())
    
    file_bytes = await file.read() 
    
    if file.content_type.startswith("image/"):
        try:
            score = await asyncio.to_thread(engine.predict_image, file_bytes)
            
            # LOGIC ĐÚNG: fake=0, real=1
            is_fake = score < 0.5
            # Nếu fake, độ tự tin = (1 - score). Nếu real, độ tự tin = score
            final_confidence = (1.0 - score) if is_fake else score
            
            return PredictResponse(
                session_id=session_id,
                filename=file.filename,
                file_type="image",
                is_deepfake=is_fake,
                confidence_score=round(final_confidence, 4),
                message="Xử lý ảnh thành công."
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    elif file.content_type.startswith("video/"):
        try:
            result = await asyncio.to_thread(engine.predict_video, file_bytes)
            avg_score = result["final_score"]
            
            is_fake = avg_score < 0.5
            final_confidence = (1.0 - avg_score) if is_fake else avg_score
            
            return PredictResponse(
                session_id=session_id,
                filename=file.filename,
                file_type="video",
                is_deepfake=is_fake,
                confidence_score=round(final_confidence, 4),
                aggregation_stats=result["stats"],
                message="Xử lý luồng video thành công."
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    else:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file 'image/' hoặc 'video/'.")