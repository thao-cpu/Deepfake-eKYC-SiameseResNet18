import cv2
import os
import numpy as np
import glob
import random
from facenet_pytorch import MTCNN
from PIL import Image

# Cố định 
random.seed(42)

# Khởi tạo mô hình nhận diện khuôn mặt MTCNN
# min_face_size=20
mtcnn = MTCNN(keep_all=False, device='cpu', min_face_size=20) 

def process_single_video(video_path, output_dir, frame_skip=15, target_size=224):
    """
    trích xuất, nhận diện, crop về 224x224 và lưu ảnh.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    
    frame_idx = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Chỉ lấy ảnh theo khoảng cách nhất định
        if frame_idx % frame_skip == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Phát hiện khuôn mặt
            boxes, probs = mtcnn.detect(img_pil)
            
            # Chỉ xử lý nếu có khuôn mặt và độ tin cậy > 0.70
            if boxes is not None and probs[0] > 0.70:
                box = boxes[0] 
                x1, y1, x2, y2 = box
                
                # Tính kích thước
                w = x2 - x1
                h = y2 - y1
                
                # Tính padding
                padding_ratio = 0.3 
                pad_w = int(w * padding_ratio)
                pad_h = int(h * padding_ratio)
                
                #Tạo tọa độ mới
                new_x1 = int(x1 - pad_w)
                new_y1 = int(y1 - pad_h * 1.5) 
                new_x2 = int(x2 + pad_w)
                new_y2 = int(y2 + pad_h)
                
            
                img_h, img_w, _ = frame.shape
                new_x1_safe = max(0, new_x1)
                new_y1_safe = max(0, new_y1)
                new_x2_safe = min(img_w, new_x2)
                new_y2_safe = min(img_h, new_y2)
                
                # Cắt ảnh
                face_crop = frame[new_y1_safe:new_y2_safe, new_x1_safe:new_x2_safe]
                
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    
                    # Thêm viền đen
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
                    
                
                    save_name = f"{video_name}_f{frame_idx}.jpg"
                    save_path = os.path.join(output_dir, save_name)
                    cv2.imwrite(save_path, face_final_square)
                    saved_count += 1
                    
        frame_idx += 1
        
    cap.release()
    return saved_count

def build_dataset(real_dir, fake_dir, output_root, split_ratio=(0.7, 0.15, 0.15), frame_skip=15):
    """
    quét video, chia tập (No Leakage) và trích xuất ảnh 
    """
    # Quét danh sách video
    real_videos = glob.glob(os.path.join(real_dir, "*.mp4"))
    fake_videos = glob.glob(os.path.join(fake_dir, "*.mp4"))
    
    print(f" Tìm thấy {len(real_videos)} video REAL và {len(fake_videos)} video FAKE")
    
    
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    # mốc phân chia
    def get_split_indices(n, ratios):
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])
        return train_end, val_end
        
    r_train_end, r_val_end = get_split_indices(len(real_videos), split_ratio)
    f_train_end, f_val_end = get_split_indices(len(fake_videos), split_ratio)
    
    datasets = {
        'train': {
            'real': real_videos[:r_train_end],
            'fake': fake_videos[:f_train_end]
        },
        'val': {
            'real': real_videos[r_train_end:r_val_end],
            'fake': fake_videos[f_train_end:f_val_end]
        },
        'test': {
            'real': real_videos[r_val_end:],
            'fake': fake_videos[f_val_end:]
        }
    }
    
    # xử lý từng thư mục
    for split_name, categories in datasets.items():
        for cls_name, video_list in categories.items():
            
            # Tạo thư mục tương ứng
            out_dir = os.path.join(output_root, split_name, cls_name)
            os.makedirs(out_dir, exist_ok=True)
            
            print(f"\nĐang xử lý tập {split_name.upper()} - Lớp {cls_name.upper()} ({len(video_list)} videos)...")
            
            total_frames = 0
            for idx, vid_path in enumerate(video_list):
                print(f"   [{idx+1}/{len(video_list)}] Đang cắt: {os.path.basename(vid_path)}", end=" -> ")
                frames_extracted = process_single_video(vid_path, out_dir, frame_skip=frame_skip)
                total_frames += frames_extracted
                print(f"{frames_extracted} ảnh")
                
            print(f"Hoàn tất tập {split_name}/{cls_name}Tổng số ảnh: {total_frames}")

if __name__ == "__main__":
    #trỏ tới 2 thư mục real và fake 
    REAL_VIDEO_DIR = r"D:\python\video_dataset_balanced\video_dataset_balanced\real"
    FAKE_VIDEO_DIR = r"D:\python\video_dataset_balanced\video_dataset_balanced\fake"
    
    # Thư mục chứa dữ liệu đầu ra
    OUTPUT_ROOT_DIR = r"D:\python\output_dataset"
    
    build_dataset(REAL_VIDEO_DIR, FAKE_VIDEO_DIR, OUTPUT_ROOT_DIR, split_ratio=(0.7, 0.15, 0.15), frame_skip=15)
    print("\nXong")