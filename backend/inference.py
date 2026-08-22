import cv2
import numpy as np
import torch
import tempfile
import os
from facenet_pytorch import MTCNN
from PIL import Image

from model_arch import ResNet18Baseline

class DeepfakeEngine:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # KHỞI TẠO MTCNN
        print("Đang khởi tạo bộ dò khuôn mặt MTCNN...")
        self.mtcnn = MTCNN(keep_all=False, device=self.device, min_face_size=20)
        
        # KHỞI TẠO MÔ HÌNH AI
        self.is_mock = not os.path.exists(model_path)
        
        if self.is_mock:
            print(f"CẢNH BÁO: Không có file '{model_path}'. Chạy chế độ MOCK.")
        else:
            print("Đang nạp mô hình ResNet18 Baseline lên RAM...")
            self.model = ResNet18Baseline(pretrained=False).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval() 
            print(f"Đã nạp thành công Baseline lên {self.device}.")

    def preprocess(self, image_np, target_size=224):
    
        try:
            # TÌM & CẮT MẶT
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Phát hiện khuôn mặt
            boxes, probs = self.mtcnn.detect(img_pil)
            
            # Nếu không tìm thấy mặt, trả về None để bỏ qua frame này
            if boxes is None or probs[0] <= 0.70:
                return None
                
            box = boxes[0]
            x1, y1, x2, y2 = box
            
            # Tính kích thước & đệm viền (Padding)
            w = x2 - x1
            h = y2 - y1
            padding_ratio = 0.3
            pad_w = int(w * padding_ratio)
            pad_h = int(h * padding_ratio)
            
            # Tọa độ mới có viền đen
            new_x1 = int(x1 - pad_w)
            new_y1 = int(y1 - pad_h * 1.5)
            new_x2 = int(x2 + pad_w)
            new_y2 = int(y2 + pad_h)
            
            img_h, img_w, _ = image_np.shape
            new_x1_safe = max(0, new_x1)
            new_y1_safe = max(0, new_y1)
            new_x2_safe = min(img_w, new_x2)
            new_y2_safe = min(img_h, new_y2)
            
            # Cắt cái mặt ra khỏi khung hình gốc
            face_crop = image_np[new_y1_safe:new_y2_safe, new_x1_safe:new_x2_safe]
            
            if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                return None

            # Ép về khung hình vuông 224x224, thêm viền đen nếu cần
            crop_h, crop_w, _ = face_crop.shape
            aspect_ratio = crop_w / crop_h
            
            if crop_w > crop_h:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            else:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            
            face_resized_temp = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            face_final_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            face_final_square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = face_resized_temp

            # CHUẨN HÓA MÀU IMAGENET
            # face_final_square hiện đang là BGR (ảnh cắt từ OpenCV)
            face_rgb = cv2.cvtColor(face_final_square, cv2.COLOR_BGR2RGB)
            face_array = face_rgb.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            face_array = (face_array - mean) / std
            
            face_tensor = np.transpose(face_array, (2, 0, 1))
            face_tensor = np.expand_dims(face_tensor, axis=0)
            
            return face_tensor
            
        except Exception as e:
            # Nếu lỗi thuật toán cắt mặt, bỏ qua frame này luôn
            return None

    def predict_image(self, image_bytes: bytes) -> float:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("File hỏng hoặc không đúng chuẩn hình ảnh.")
        
        numpy_tensor = self.preprocess(img)
        
        if numpy_tensor is None:
            raise ValueError("Không tìm thấy khuôn mặt rõ ràng trong ảnh để phân tích.")
        
        if self.is_mock:
            return float(np.random.uniform(0.0, 1.0))
        else:
            torch_tensor = torch.from_numpy(numpy_tensor).to(self.device)
            with torch.no_grad():
                logit = self.model(torch_tensor)
                score = torch.sigmoid(logit).item() 
            return score

    def predict_video(self, video_bytes: bytes) -> dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            raise ValueError("Không thể đọc video.")

        scores = []
        frame_count = 0
        error_frames = 0
        frame_skip = 10 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            if frame_count % frame_skip == 0:
                numpy_tensor = self.preprocess(frame)
                
                # Chỉ tính điểm nếu MTCNN dò ra mặt
                if numpy_tensor is not None:
                    if self.is_mock:
                        score = float(np.random.uniform(0.0, 1.0))
                    else:
                        torch_tensor = torch.from_numpy(numpy_tensor).to(self.device)
                        with torch.no_grad():
                            logit = self.model(torch_tensor)
                            score = torch.sigmoid(logit).item()
                    scores.append(score)
                else:
                    error_frames += 1
            
            frame_count += 1
            if len(scores) >= 30: break

        cap.release()
        os.remove(temp_video_path)

        if len(scores) == 0:
            raise ValueError("Không tìm thấy khuôn mặt nào trong toàn bộ video.")

        return {
            "final_score": sum(scores) / len(scores),
            "stats": {
                "total_frames_processed": len(scores),
                "frames_no_face_detected": error_frames,
                "max_score": round(max(scores), 4),
                "min_score": round(min(scores), 4)
            }
        }