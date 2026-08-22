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
            print(f"CẢNH BÁO: Không tìm thấy file model tại '{model_path}'.")
            print("TỰ ĐỘNG KÍCH HOẠT CHẾ ĐỘ MOCK.")
        else:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            print(f"Đã nạp thành công mô hình ONNX từ '{model_path}'.")

    def preprocess(self, image_np):
        """
        [CHỜ GHÉP CODE]: Chỗ này nhắc Kiều My viết thuật toán MTCNN/RetinaFace cắt mặt.
        Tạm thời dùng resize cơ bản để test luồng.
        """
        try:
            face_array = cv2.resize(image_np, (224, 224)) 
            face_array = face_array.astype(np.float32) / 255.0
            face_tensor = np.transpose(face_array, (2, 0, 1))
            face_tensor = np.expand_dims(face_tensor, axis=0)
            return face_tensor
        except Exception as e:
            raise ValueError(f"Lỗi tiền xử lý (Crop/Resize): {str(e)}")

    def predict_image(self, image_bytes: bytes) -> float:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("File hỏng hoặc không đúng chuẩn hình ảnh.")
        
        tensor = self.preprocess(img)
        
        if self.is_mock:
            return float(np.random.uniform(0.0, 1.0))
        else:
            outputs = self.session.run(None, {self.input_name: tensor})
            return float(outputs[0][0])

    def predict_video(self, video_bytes: bytes) -> dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            raise ValueError("Không thể đọc video. File hỏng hoặc sai định dạng.")

        scores = []
        frame_count = 0
        error_frames = 0
        frame_skip = 10 # Chỉ bóc 1 frame sau mỗi 10 frames để tối ưu
        
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
            if len(scores) >= 30: # Giới hạn tối đa test 30 frames
                break

        cap.release()
        os.remove(temp_video_path)

        if len(scores) == 0:
            raise ValueError("Không tìm thấy khuôn mặt nào trong video.")

        avg_score = sum(scores) / len(scores)
        
        return {
            "final_score": avg_score,
            "stats": {
                "total_frames_processed": len(scores),
                "error_frames": error_frames,
                "max_score": round(max(scores), 4),
                "min_score": round(min(scores), 4)
            }
        }