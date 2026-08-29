"""
Filter giả lập deepfake THẬT — tác động trực tiếp lên vùng mặt
(chỉnh hình dáng, màu da, blend boundary) thay vì chỉ vẽ phụ kiện.
Mục tiêu: tạo artifact giống deepfake thật để test model.
"""

import cv2
import numpy as np


def _get_face_mask(landmarks, img_w, img_h, expand=1.2):
    """
    Tạo mask vùng mặt từ landmarks (hull lồi).
    expand: phóng to mask ra bao gồm cả cằm/cổ một chút.
    """
    # Các điểm tạo thành viền mặt
    face_outline_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]
    
    # Chuyển landmarks thành điểm ảnh
    points = []
    for idx in face_outline_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            points.append([x, y])
    
    if len(points) < 3:
        return None
    
    points = np.array(points, dtype=np.int32)
    
    # Tạo mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    
    # Làm mờ cạnh mask để blend tự nhiên hơn
    if expand > 1.0:
        kernel_size = int(min(img_w, img_h) * 0.02 * (expand - 1) * 10)
        if kernel_size > 0 and kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            mask = (mask > 127).astype(np.uint8) * 255
    
    return mask


def _get_eyes_mouth_mask(landmarks, img_w, img_h):
    """
    Tạo mask vùng mắt và miệng — giữ nguyên, không làm mờ.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # Mắt trái
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # Mắt phải  
    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Miệng
    mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    
    for indices in [left_eye_indices, right_eye_indices, mouth_indices]:
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append([int(lm.x * img_w), int(lm.y * img_h)])
        if len(pts) >= 3:
            cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 255)
    
    # Mở rộng mask một chút để không bị sóng ở cạnh
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask


def apply_deepfake_face_smooth(img, landmarks, img_w, img_h):
    """
    Giả lập da GAN: làm mịn CHỈ vùng da mặt, giữ nguyên mắt/miệng/mũi.
    Đây là artifact phổ biến nhất của deepfake — da quá mượt, mất texture.
    """
    face_mask = _get_face_mask(landmarks, img_w, img_h, expand=1.15)
    if face_mask is None:
        return img
    
    eyes_mouth_mask = _get_eyes_mouth_mask(landmarks, img_w, img_h)
    
    # Mask cuối cùng: mặt trừ mắt/miệng
    final_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(eyes_mouth_mask))
    
    # Làm mịn da (bilateral filter giữ biên)
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Blend: chỉ áp dụng smooth ở vùng mask
    final_mask_3ch = cv2.merge([final_mask, final_mask, final_mask]).astype(np.float32) / 255.0
    result = (img.astype(np.float32) * (1 - final_mask_3ch) + 
              smoothed.astype(np.float32) * final_mask_3ch).astype(np.uint8)
    
    return result


def apply_deepfake_color_shift(img, landmarks, img_w, img_h):
    """
    Giả lập lỗi blend màu: vùng mặt có tông màu khác biệt so với
    cổ/vai/xung quanh — artifact rất phổ biến ở DeepFaceLab.
    """
    face_mask = _get_face_mask(landmarks, img_w, img_h, expand=1.1)
    if face_mask is None:
        return img
    
    # Làm mờ mask để blend dần
    blur_size = int(min(img_w, img_h) * 0.03)
    if blur_size % 2 == 0:
        blur_size += 1
    soft_mask = cv2.GaussianBlur(face_mask, (blur_size, blur_size), 0)
    soft_mask = soft_mask.astype(np.float32) / 255.0
    
    # Tạo ảnh mặt với màu lệch (ấm hơn/hơi xanh)
    face_only = img.copy().astype(np.float32)
    
    # Lệch màu nhẹ: bớt đỏ, thêm xanh — tạo cảm giác "khác vùng da"
    face_only[:, :, 0] = np.clip(face_only[:, :, 0] - 8, 0, 255)  # B - 
    face_only[:, :, 1] = np.clip(face_only[:, :, 1] + 3, 0, 255)   # G +
    face_only[:, :, 2] = np.clip(face_only[:, :, 2] + 10, 0, 255)  # R +
    
    # Blend
    mask_3ch = cv2.merge([soft_mask, soft_mask, soft_mask])
    result = (img.astype(np.float32) * (1 - mask_3ch) + 
              face_only * mask_3ch).astype(np.uint8)
    
    return result


def apply_deepfake_boundary_artifact(img, landmarks, img_w, img_h):
    """
    Giả lập lỗi ở biên mặt — vùng giao giữa face-swap và background
    có hiện tượng "seam" (đường may) hoặc mismatch.
    """
    face_mask = _get_face_mask(landmarks, img_w, img_h, expand=1.0)
    if face_mask is None:
        return img
    
    # Tìm edge của mask (biên mặt)
    edges = cv2.Canny(face_mask, 100, 200)
    
    # Làm dày edge một chút
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Tạo artifact ở edge: nhiễu + lệch màu nhẹ
    edge_mask = edges.astype(np.float32) / 255.0
    
    # Nhiễu ở biên
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    
    # Lệch màu ở biên
    color_shift = np.zeros_like(img, dtype=np.float32)
    color_shift[:, :, 2] = 15  # thêm đỏ ở biên
    
    artifact = noise + color_shift
    
    # Chỉ áp dụng ở vùng edge
    mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask])
    result = (img.astype(np.float32) + artifact * mask_3ch).astype(np.uint8)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_deepfake_warp(img, landmarks, img_w, img_h):
    """
    Giả lập biến dạng hình dáng mặt nhẹ — deepfake đôi khi làm
    mặt hơi dài/hẹp khác so với gốc, hoặc không khớp hoàn toàn.
    """
    face_mask = _get_face_mask(landmarks, img_w, img_h, expand=1.1)
    if face_mask is None:
        return img
    
    # Tính center của mặt
    moments = cv2.moments(face_mask)
    if moments["m00"] == 0:
        return img
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    # Tạo map biến dạng nhẹ (kéo giãn mặt theo trục Y)
    map_x = np.copy(np.float32(np.arange(img_w))[np.newaxis, :] * np.ones((img_h, 1)))
    map_y = np.copy(np.float32(np.arange(img_h)[:, np.newaxis]) * np.ones((1, img_w)))
    
    # Kéo giãn nhẹ theo Y (mặt dài hơn một chút)
    scale_y = 1.03
    map_y = cy + (map_y - cy) * scale_y
    
    # Thu hẹp nhẹ theo X (mặt hẹp hơn)
    scale_x = 0.98
    map_x = cx + (map_x - cx) * scale_x
    
    # Chỉ warp vùng mặt
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Blend
    soft_mask = cv2.GaussianBlur(face_mask, (21, 21), 0).astype(np.float32) / 255.0
    mask_3ch = cv2.merge([soft_mask, soft_mask, soft_mask])
    result = (img.astype(np.float32) * (1 - mask_3ch) + 
              warped.astype(np.float32) * mask_3ch).astype(np.uint8)
    
    return result


def apply_deepfake_asymmetric(img, landmarks, img_w, img_h):
    """
    Giả lập bất đối xứng — mặt trái/phải có chất lượng khác nhau,
    rất hay gặp khi face-swap chỉ khớp một nửa.
    """
    face_mask = _get_face_mask(landmarks, img_w, img_h, expand=1.1)
    if face_mask is None:
        return img
    
    # Chia mask làm 2 nửa
    nose_x = int(landmarks[1].x * img_w)  # mũi
    
    left_mask = face_mask.copy()
    left_mask[:, nose_x:] = 0
    
    right_mask = face_mask.copy()
    right_mask[:, :nose_x] = 0
    
    # Nửa trái: mịn hơn (giống GAN tốt)
    left_smooth = cv2.bilateralFilter(img, d=12, sigmaColor=100, sigmaSpace=100)
    
    # Nửa phải: giữ nguyên + thêm nhiễu nhẹ (giống blend kém)
    right_noisy = img.copy().astype(np.float32)
    noise = np.random.normal(0, 5, img.shape)
    right_noisy = np.clip(right_noisy + noise, 0, 255).astype(np.uint8)
    
    # Blend từng nửa
    left_mask_3ch = cv2.merge([left_mask, left_mask, left_mask]).astype(np.float32) / 255.0
    right_mask_3ch = cv2.merge([right_mask, right_mask, right_mask]).astype(np.float32) / 255.0
    
    result = img.astype(np.float32)
    result = result * (1 - left_mask_3ch) + left_smooth.astype(np.float32) * left_mask_3ch
    result = result * (1 - right_mask_3ch) + right_noisy.astype(np.float32) * right_mask_3ch
    result = result.astype(np.uint8)
    
    return result


def apply_deepfake_eye_glitch(img, landmarks, img_w, img_h):
    """
    Giả lập lỗi ở mắt — vùng mắt có thể bị mờ/lệch,
    vì mắt là phần khó blend nhất trong deepfake.
    """
    # Mắt trái
    left_eye_center = landmarks[468] if len(landmarks) > 468 else landmarks[159]
    left_eye_x = int(left_eye_center.x * img_w)
    left_eye_y = int(left_eye_center.y * img_h)
    
    # Mắt phải
    right_eye_center = landmarks[473] if len(landmarks) > 473 else landmarks[386]
    right_eye_x = int(right_eye_center.x * img_w)
    right_eye_y = int(right_eye_center.y * img_h)
    
    # Kích thước vùng mắt
    eye_radius = int(min(img_w, img_h) * 0.04)
    
    result = img.copy()
    
    for ex, ey in [(left_eye_x, left_eye_y), (right_eye_x, right_eye_y)]:
        # Tạo mask tròn cho mắt
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(mask, (ex, ey), eye_radius, 255, -1)
        soft_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        soft_mask = soft_mask.astype(np.float32) / 255.0
        
        # Làm mờ vùng mắt
        eye_region = img[max(0, ey-eye_radius):ey+eye_radius, 
                        max(0, ex-eye_radius):ex+eye_radius]
        if eye_region.size > 0:
            blurred = cv2.GaussianBlur(eye_region, (5, 5), 0)
            
            # Thêm lệch màu nhẹ
            blurred = blurred.astype(np.float32)
            blurred[:, :, 0] += 5  # B
            blurred[:, :, 1] -= 3  # G  
            blurred = np.clip(blurred, 0, 255).astype(np.uint8)
            
            # Đặt lại vào result
            y1, y2 = max(0, ey-eye_radius), ey+eye_radius
            x1, x2 = max(0, ex-eye_radius), ex+eye_radius
            h, w = y2-y1, x2-x1
            mask_crop = soft_mask[y1:y2, x1:x2]
            mask_3ch = cv2.merge([mask_crop, mask_crop, mask_crop])
            
            result[y1:y2, x1:x2] = (
                result[y1:y2, x1:x2].astype(np.float32) * (1 - mask_3ch) + 
                blurred.astype(np.float32) * mask_3ch
            ).astype(np.uint8)
    
    return result


# ============================================================
# COMBO FILTERS — Kết hợp nhiều artifact
# ============================================================

def apply_deepfake_realistic(img, landmarks, img_w, img_h):
    """
    Combo thực tế nhất: Da mịn + Lệch màu + Biên artifact
    Giống deepfake từ DeepFaceLab/ROOP chất lượng trung bình.
    """
    img = apply_deepfake_face_smooth(img, landmarks, img_w, img_h)
    img = apply_deepfake_color_shift(img, landmarks, img_w, img_h)
    img = apply_deepfake_boundary_artifact(img, landmarks, img_w, img_h)
    return img


def apply_deepfake_subtle(img, landmarks, img_w, img_h):
    """
    Combo tinh vi: Chỉ warp nhẹ + da mịn
    Giống deepfake chất lượng cao, khó phát hiện bằng mắt.
    """
    img = apply_deepfake_warp(img, landmarks, img_w, img_h)
    img = apply_deepfake_face_smooth(img, landmarks, img_w, img_h)
    return img


def apply_deepfake_obvious(img, landmarks, img_w, img_h):
    """
    Combo rõ ràng: Tất cả artifact + bất đối xứng + lỗi mắt
    Giống deepfake kém chất lượng, dễ phát hiện.
    """
    img = apply_deepfake_face_smooth(img, landmarks, img_w, img_h)
    img = apply_deepfake_color_shift(img, landmarks, img_w, img_h)
    img = apply_deepfake_boundary_artifact(img, landmarks, img_w, img_h)
    img = apply_deepfake_asymmetric(img, landmarks, img_w, img_h)
    img = apply_deepfake_eye_glitch(img, landmarks, img_w, img_h)
    return img


# ============================================================
# MAIN ENTRY
# ============================================================

DEEPFAKE_FACE_FILTERS = {
    "realistic": ("🤥 Realistic (da mịn+lệch màu+biên)", apply_deepfake_realistic),
    "subtle": ("👁️ Subtle (warp+da mịn, khó thấy)", apply_deepfake_subtle),
    "obvious": ("⚠️ Obvious (tất cả artifact, rõ ràng)", apply_deepfake_obvious),
    "face_smooth": ("✨ Da mịn GAN (chỉ smooth da mặt)", apply_deepfake_face_smooth),
    "color_shift": ("🎨 Lệch màu blend (màu da khác nền)", apply_deepfake_color_shift),
    "boundary": ("📍 Biên artifact (seam tại viền mặt)", apply_deepfake_boundary_artifact),
    "warp": ("📐 Biến dạng (mặt dài/hẹp khác)", apply_deepfake_warp),
    "asymmetric": ("⚖️ Bất đối xứng (2 nửa mặt khác nhau)", apply_deepfake_asymmetric),
    "eye_glitch": ("👁 Lỗi mắt (mắt mờ/lệch)", apply_deepfake_eye_glitch),
}


def apply_deepfake_face_filter(img, landmarks, filter_name="realistic"):
    """
    Hàm chính — gọi từ LivenessProcessor.
    
    Args:
        img: BGR image
        landmarks: MediaPipe face landmarks (list)
        filter_name: tên filter trong DEEPFAKE_FACE_FILTERS
    
    Returns:
        img đã áp dụng artifact deepfake lên vùng mặt
    """
    if landmarks is None:
        return img
    
    img_h, img_w = img.shape[:2]
    
    filter_entry = DEEPFAKE_FACE_FILTERS.get(filter_name)
    if filter_entry is None:
        return img
    
    _, filter_func = filter_entry
    return filter_func(img, landmarks, img_w, img_h)