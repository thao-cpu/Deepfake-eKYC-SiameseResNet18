"""
Virtual camera / deepfake-injection detection.

Deepfake real-time attack (DeepFaceLive, DeepFaceLab-live, ...) thường
hoạt động bằng cách cài 1 "virtual camera" giả lập, rồi chọn nó làm
nguồn input thay cho webcam thật. Trình duyệt cho phép liệt kê danh
sách camera device (tên) trước khi bắt đầu stream -> quét tên đó để
cảnh báo nếu khớp phần mềm virtual cam phổ biến.
"""

import streamlit as st

from liveness_app.config import VIRTUAL_CAM_KEYWORDS

try:
    from streamlit_js_eval import streamlit_js_eval
    _JS_EVAL_AVAILABLE = True
except ImportError:
    # Thiếu package streamlit-js-eval (pip install streamlit-js-eval).
    # Không crash toàn app — chỉ tự tắt bước check virtual camera.
    _JS_EVAL_AVAILABLE = False


def check_virtual_camera_and_get_permission():
    """
    Vẽ UI cảnh báo (nếu cần) và trả về `camera_start_allowed: bool`
    quyết định có cho phép khởi động webrtc_streamer hay không.
    """
    if not _JS_EVAL_AVAILABLE:
        # Thiếu package streamlit-js-eval -> bỏ qua bước check, không chặn app.
        # Cài `pip install streamlit-js-eval` để bật lại tính năng này.
        st.caption(
            "ℹ️ Bỏ qua kiểm tra virtual camera (thiếu package `streamlit-js-eval`, "
            "chạy `pip install streamlit-js-eval` để bật)."
        )
        return True

    camera_devices = streamlit_js_eval(
        js_expressions="""
        (async () => {
            try {
                // Xin quyền trước, nếu không label sẽ bị trình duyệt ẩn (rỗng)
                const s = await navigator.mediaDevices.getUserMedia({video: true});
                s.getTracks().forEach(t => t.stop());
                const devices = await navigator.mediaDevices.enumerateDevices();
                return devices
                    .filter(d => d.kind === 'videoinput')
                    .map(d => d.label || 'unknown-camera');
            } catch (e) {
                return [];
            }
        })()
        """,
        key="camera_device_list",
    )

    suspicious_devices = []
    if camera_devices:
        for label in camera_devices:
            low = str(label).lower()
            if any(kw in low for kw in VIRTUAL_CAM_KEYWORDS):
                suspicious_devices.append(label)

    if camera_devices is None:
        # streamlit_js_eval trả None trong lúc chờ JS phản hồi (hoặc ngay
        # sau khi có rerun khác xen vào, vd bấm nút toggle deepfake).
        # KHÔNG khoá camera trong lúc chờ — chỉ cảnh báo nếu sau đó phát
        # hiện thật sự có thiết bị đáng ngờ (nhánh elif bên dưới).
        st.caption("🔍 Đang kiểm tra danh sách camera trên máy...")
        return True

    elif suspicious_devices:
        st.error(
            "🚨 Phát hiện thiết bị camera nghi là VIRTUAL CAMERA (có thể dùng để "
            "bơm hình ảnh deepfake real-time thay cho webcam thật): "
            + ", ".join(f"`{d}`" for d in suspicious_devices)
        )
        st.warning(
            "Khi bắt đầu camera bên dưới, hãy bấm nút **SELECT DEVICE** (do "
            "streamlit-webrtc hiển thị) và chọn đúng webcam vật lý — "
            "không chọn thiết bị nêu trên."
        )
        return st.checkbox(
            "✅ Tôi xác nhận đã/sẽ chọn đúng webcam vật lý thật, "
            "không dùng virtual camera"
        )

    else:
        st.caption(
            f"✅ Không phát hiện virtual camera đáng ngờ trong "
            f"{len(camera_devices)} thiết bị camera tìm thấy."
        )
        return True
