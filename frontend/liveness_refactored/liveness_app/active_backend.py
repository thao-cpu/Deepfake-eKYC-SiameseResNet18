"""
Gọi Backend AI để xác nhận kết quả sau khi Active Liveness PASS challenge,
và tổng hợp kết luận eKYC cuối cùng (Active Challenge + Deepfake Filter +
Backend Passive Model).
"""

import cv2
import requests
import streamlit as st

from liveness_app.config import BACKEND_URL


def _crop_face_224(img, landmarks, padding=0.3):
    """Crop vùng mặt rồi resize về 224x224."""
    if landmarks is None:
        return None
    
    img_h, img_w = img.shape[:2]
    
    x_coords = [lm.x * img_w for lm in landmarks]
    y_coords = [lm.y * img_h for lm in landmarks]
    
    x_min = int(min(x_coords))
    x_max = int(max(x_coords))
    y_min = int(min(y_coords))
    y_max = int(max(y_coords))
    
    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)
    
    x_min = max(0, x_min - pad_x)
    x_max = min(img_w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(img_h, y_max + pad_y)
    
    face_crop = img[y_min:y_max, x_min:x_max]
    
    if face_crop.size == 0:
        return None
    
    return cv2.resize(face_crop, (224, 224))


def call_active_backend(processor):
    """
    Gửi frame (crop mặt 224x224) cho Backend để xác nhận REAL/FAKE.

    Trả về (backend_is_real, api_result_status):
        backend_is_real: True/False nếu backend trả lời được, None nếu lỗi.
        api_result_status: chuỗi mô tả trạng thái gọi API (hiển thị UI).
    """
    api_result_status = "🔄 Đang gọi..."
    backend_is_real = None

    with st.spinner("🔄 Đang gọi Backend AI để xác nhận kết quả..."):
        try:
            # 1. Lấy ảnh sạch, crop mặt, resize 224x224
            frame_to_send = processor.last_frame_clean
            face_224 = _crop_face_224(frame_to_send, processor.last_landmarks)
            
            if face_224 is None:
                face_224 = cv2.resize(frame_to_send, (224, 224))
            
            ok, buf = cv2.imencode(".jpg", face_224, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if not ok:
                st.error("Lỗi mã hóa ảnh")
                st.stop()

            frame_bytes = buf.tobytes()
            files = {
                "file": ("active_frame.jpg", frame_bytes, "image/jpeg")
            }

            # 2. Data đi kèm
            data = {
                "liveness_type": "active",
                "status": "pass",
                "verification_method": "oval_align_then_challenge",
                "hold_seconds": str(processor.oval_hold_required),
                "challenge_passed": processor.current_challenge or "",
            }

            # 3. Gọi API
            response = requests.post(
                BACKEND_URL,
                files=files,
                data=data,
                timeout=10
            )

            if response.status_code == 200:
                try:
                    result_data = response.json()

                    is_deepfake = result_data.get("is_deepfake", False)
                    score = result_data.get("confidence_score", 0)
                    message = result_data.get("message", "Xác minh thành công")

                    if is_deepfake:
                        result_color = "#f85149"
                        result_text = "🚨 CẢNH BÁO: FAKE"
                        sub_text = f"Backend xác nhận: {message}"
                    else:
                        result_color = "#3fb950"
                        result_text = "✅ AN TOÀN: REAL"
                        sub_text = f"Backend xác nhận: {message}"

                    st.markdown(f"""
                    <div class="result-box" style="border-color: {result_color};">
                        <div class="card-title">Backend Verification Result</div>
                        <div style="font-size: 2rem; font-weight: 700; color: {result_color};">
                            {result_text}
                        </div>
                        <div style="color: #9aa4b2; margin-top: 8px;">
                            {sub_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    dfv = st.session_state.get("deepfake_verdict")
                    if dfv and dfv["n_samples"] > 0:
                        verdict_note = (
                            "nghi ngờ deepfake" if dfv["is_suspected_fake"]
                            else "không nghi ngờ"
                        )
                        st.caption(
                            f"🛡️ Deepfake filter trong lúc Active Liveness: "
                            f"{dfv['n_samples']} lần check, "
                            f"điểm fake trung bình {dfv['avg_score']*100:.1f}% "
                            f"({verdict_note})."
                        )

                    with st.expander("🔍 Xem chi tiết JSON trả về (Dùng để test)"):
                        st.json(result_data)

                    api_result_status = "✅ Đã gọi xong (200 OK)"

                except requests.exceptions.JSONDecodeError:
                    st.warning(f"Backend trả về 200 nhưng không phải JSON. Nội dung: `{response.text}`")
                    api_result_status = "⚠️ 200 OK nhưng không phải JSON"
            else:
                try:
                    error_msg = response.json().get('detail', 'Lỗi không xác định')
                except Exception:
                    error_msg = response.text
                st.error(f"❌ Backend lỗi (Mã {response.status_code}): {error_msg}")
                api_result_status = f"⚠️ Backend lỗi (mã {response.status_code})"

        except requests.exceptions.ConnectionError:
            st.error("❌ Không thể kết nối tới Backend.")
            api_result_status = "❌ Không kết nối được Backend"
        except Exception as e:
            st.error(f"❌ Lỗi hệ thống: {str(e)}")
            api_result_status = "❌ Lỗi hệ thống khi gọi API"

    return backend_is_real, api_result_status


def render_overall_verdict(n_samples, is_suspected_fake, backend_is_real):
    """
    Gộp 3 tín hiệu: Active Challenge + Deepfake Filter (active session)
    + Backend Passive Model — để demo cách các lớp phối hợp với nhau,
    thay vì chỉ nhìn 1 con số PASS/FAIL đơn lẻ.
    """
    signals = [("Active Challenge (blink/turn)", True)]
    if n_samples > 0:
        signals.append(
            ("Deepfake Filter (Active session)", not is_suspected_fake)
        )
    if backend_is_real is not None:
        signals.append(("Backend Passive Model", backend_is_real))

    overall_ok = all(ok for _, ok in signals)
    verdict_color = "#3fb950" if overall_ok else "#f85149"
    verdict_text = (
        "✅ XÁC THỰC THÀNH CÔNG (REAL)"
        if overall_ok
        else "🚫 NGHI NGỜ GIẢ MẠO — TỪ CHỐI"
    )
    signal_rows = "".join(
        f'<div class="status-row">'
        f'<span class="status-label">{name}: </span>'
        f'<b style="color:{"#3fb950" if ok else "#f85149"};">'
        f'{"✅ OK" if ok else "❌ Không đạt"}</b></div>'
        for name, ok in signals
    )
    st.markdown(f"""
    <div class="result-box" style="border-color:{verdict_color}; margin-top:16px;">
        <div class="card-title">🎯 KẾT LUẬN eKYC TỔNG HỢP (DEMO)</div>
        <div style="font-size:1.8rem;font-weight:700;color:{verdict_color};">
            {verdict_text}
        </div>
        <div style="margin-top:10px;">
            {signal_rows}
        </div>
    </div>
    """, unsafe_allow_html=True)