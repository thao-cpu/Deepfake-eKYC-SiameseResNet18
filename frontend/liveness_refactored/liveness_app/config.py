"""
Cấu hình chung cho toàn bộ app: page config của Streamlit + các hằng số
(URL backend, ngưỡng deepfake, danh sách challenge, từ khoá virtual camera).
Tách riêng để mọi module khác chỉ cần `from liveness_app import config`
thay vì hard-code URL/hằng số rải rác nhiều nơi.
"""

import streamlit as st


def configure_page():
    """Gọi 1 lần duy nhất, ở đầu app.py, trước khi vẽ bất kỳ widget nào."""
    st.set_page_config(
        page_title="Liveness Detection",
        page_icon="🔐",
        layout="wide",
    )


# ============================================================
# BACKEND API
# ============================================================

# === THAY URL NÀY CHO KHỚP VỚI BACKEND CỦA BẠN ===
BACKEND_URL = "https://a3ea12646af62e.lhr.life/predict"

# Endpoint dùng để check deepfake theo từng frame trong lúc Active Liveness
# (dùng chung backend passive /predict, chỉ đổi ảnh gửi lên)
DEEPFAKE_CHECK_URL = "https://a3ea12646af62e.lhr.life/predict"
DEEPFAKE_CHECK_INTERVAL = 1.5  # giây giữa mỗi lần gọi API (KHÔNG gọi mỗi frame)
DEEPFAKE_FAKE_THRESHOLD = 0.5  # confidence_score trung bình > ngưỡng này => coi là fake


# ============================================================
# ACTIVE LIVENESS CHALLENGES
# ============================================================

CHALLENGES = [
    "blink",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "tilt_left",
    "tilt_right",
]


# ============================================================
# VIRTUAL CAMERA / DEEPFAKE-INJECTION DETECTION
# ============================================================
# Deepfake real-time attack (DeepFaceLive, DeepFaceLab-live, ...) thường
# hoạt động bằng cách cài 1 "virtual camera" giả lập, rồi chọn nó làm
# nguồn input thay cho webcam thật. Trình duyệt cho phép liệt kê danh
# sách camera device (tên) trước khi bắt đầu stream -> quét tên đó để
# cảnh báo nếu khớp phần mềm virtual cam phổ biến.

VIRTUAL_CAM_KEYWORDS = [
    "obs", "virtual", "manycam", "droidcam", "snap camera",
    "xsplit", "camtwist", "e2esoft", "iriun", "epoccam",
    "ndi", "chromacam", "youcam", "camo", "deepfacelive",
    "vcam", "webcamoid", "ivcam", "reincubate",
]
