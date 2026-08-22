import cv2
import numpy as np
import onnxruntime as ort
import tempfile
import os

class DeepfakeEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_mock = not os.path.exists(model_path)
        
        if self.is_mock:
            print(f" CẢNH BÁO: Không tìm thấy model tại '{model_path}'. Hệ thống tự động chuyển sang chế độ MOCK.")
        else:
            # Chỉ nạp model ONNX khi file thực sự tồn tại
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            print(" Đã nạp thành công mô hình ONNX.")

    def preprocess(self, image_np):
        
        # Tiền xử lý: Resize -> Normalize -> Chuyển thành Tensor
    
        try:
            face_array = cv2.resize(image_np, (224, 224)) 
            face_array = face_array.astype(np.float32) / 255.0
            face_tensor = np.transpose(face_array, (2, 0, 1))
            face_tensor = np.expand_dims(face_tensor, axis=0)
            return face_tensor
        except Exception as e:
            raise ValueError(f"Lỗi tiền xử lý: {str(e)}")

    def predict_image(self, image_bytes: bytes) -> float:
        # Dự đoán 1 tấm ảnh
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể đọc được file ảnh. File lỗi hoặc sai định dạng.")
        
        tensor = self.preprocess(img)
        
        if self.is_mock:
            # Chạy giả lập nếu chưa có model
            return float(np.random.uniform(0.0, 1.0))
        else:
            # Chạy Inference thật khi đã có file ONNX
            outputs = self.session.run(None, {self.input_name: tensor})
            return float(outputs[0][0])

    def predict_video(self, video_bytes: bytes) -> dict:
        
        # Xử lý video và Aggregation
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            raise ValueError("File video bị hỏng hoặc không đúng định dạng.")

        scores = []
        frame_count = 0
        error_frames = 0
        frame_skip = 10 # Chỉ test 1 frame mỗi 10 frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                try:
                    tensor = self.preprocess(frame)
                    
                    if self.is_mock:
                        score = float(np.random.uniform(0.0, 1.0))
                    else:
                        outputs = self.session.run(None, {self.input_name: tensor})
                        score = float(outputs[0][0])
                        
                    scores.append(score)
                except:
                    error_frames += 1
            
            frame_count += 1
            
            # Giới hạn 50 frames để tránh treo máy lúc dev
            if len(scores) >= 50:
                break

        cap.release()
        os.remove(temp_video_path)

        if len(scores) == 0:
            raise ValueError("Video quá ngắn hoặc không tìm thấy khuôn mặt trong bất kỳ frame nào.")

        avg_score = sum(scores) / len(scores)
        
        return {
            "final_score": avg_score,
            "stats": {
                "total_frames_processed": len(scores),
                "error_frames": error_frames,
                "max_score": max(scores),
                "min_score": min(scores)
            }
        }