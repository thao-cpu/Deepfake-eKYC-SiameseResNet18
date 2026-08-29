"""
Passive Liveness: upload ảnh/video, hiển thị trạng thái xác minh,
gọi backend AI và hiển thị kết quả (có xử lý lỗi JSON đầy đủ).
"""

import requests
import streamlit as st

from liveness_app.config import BACKEND_URL


def render_passive_liveness():
    """Vẽ toàn bộ khối Passive Liveness (upload + status + kết quả backend)."""
    left, right = st.columns([1.25, 1])

    uploaded_file = _render_upload_panel(left)
    _render_verification_status_panel(right, uploaded_file)
    _render_result_and_call_backend(uploaded_file)

    return uploaded_file


# ------------------------------------------------------------
# LEFT: INPUT
# ------------------------------------------------------------
def _render_upload_panel(container):
    with container:
        st.subheader("📁 Passive Liveness")

        st.caption(
            "Upload an image or video for passive liveness detection."
        )

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[
                "jpg",
                "jpeg",
                "png",
                "mp4",
                "avi",
                "mov"
            ],
        )

        if uploaded_file is None:

            st.info(
                "Upload an image or video to start "
                "passive liveness detection."
            )

        else:

            st.success(
                f"✓ File received: {uploaded_file.name}"
            )

            st.write(
                f"**Type:** `{uploaded_file.type}`"
            )

            st.write(
                f"**Size:** `{uploaded_file.size / 1024:.1f} KB`"
            )

            if uploaded_file.type.startswith("image"):

                st.image(
                    uploaded_file,
                    caption="Input image",
                    use_container_width=True
                )

            elif uploaded_file.type.startswith("video"):

                st.video(uploaded_file)

    return uploaded_file


# ------------------------------------------------------------
# RIGHT: VERIFICATION STATUS
# ------------------------------------------------------------
def _render_verification_status_panel(container, uploaded_file):
    with container:
        st.subheader("🔎 Verification Status")
        st.divider()

        rows = [
            ("System", "● Ready", "success"),
            ("Input",
              "● Received" if uploaded_file is not None else "● Waiting",
              "success" if uploaded_file is not None else "warning"),

            ("Passive Model", "● Waiting for API", "warning"),
            ("Result", "—", "info"),
        ]

        for label, value, status in rows:
            col1, col2 = st.columns([1, 1.6])

            with col1:
                st.markdown(f"**{label}**")

            with col2:
                if status == "success":
                    st.success(value)
                elif status == "warning":
                    st.warning(value)
                else:
                    st.info(value)


# ============================================================
# RESULT & API INTEGRATION (ĐÃ SỬA LỖI JSON)
# ============================================================
def _render_result_and_call_backend(uploaded_file):
    if uploaded_file is None:
        return

    with st.spinner("Đang gọi Backend AI để phân tích..."):
        try:
            # 1. Chuẩn bị file
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type or "application/octet-stream"
            files = {
                "file": (uploaded_file.name, file_bytes, mime_type)
            }

            # 2. Gọi API với timeout dài (xử lý video lâu)
            response = requests.post(
                BACKEND_URL,
                files=files,
                timeout=120  # 👈 2 phút cho video
            )

            # 3. LƯÔNG XỬ LÝ AN TOÀN
            content_type = response.headers.get("Content-Type", "")
            st.write(f"📡 HTTP Status: `{response.status_code}`")
            st.write(f"📡 Content-Type: `{content_type}`")

            # 4. KIỂM TRA BODY CÓ RỖNG KHÔNG
            if not response.text or response.text.strip() == "":
                st.error("❌ Backend trả response rỗng (empty body).")
                st.info("👉 Nguyên nhân: backend có thể chưa return JSON, hoặc crash khi xử lý.")
                st.stop()

            # 5. KIỂM TRA CONTENT-TYPE PHẢI JSON
            if "application/json" not in content_type:
                st.error(
                    f"❌ Backend không trả JSON. "
                    f"Content-Type nhận được: `{content_type}`."
                )
                st.text_area("Nội dung backend trả về:", response.text[:1000])
                st.stop()

            # 6. PARSE JSON AN TOÀN
            try:
                result_data = response.json()
            except Exception as json_err:
                st.error(f"❌ Không parse được JSON: `{str(json_err)}`")
                st.text_area("Body raw:", response.text[:1000])
                st.stop()

            # 7. XỬ LÝ KẾT QUẢ
            if response.status_code == 200:
                is_fake = result_data.get("is_deepfake", None)
                score = result_data.get("confidence_score", 0)

                if is_fake is None:
                    st.error("❌ Backend trả 200 nhưng thiếu field `is_deepfake`.")
                    st.json(result_data)
                    st.stop()

                if is_fake:
                    result_color = "#f85149"
                    result_text = "🚨 CẢNH BÁO: FAKE"
                    sub_text = f"Phát hiện dấu hiệu giả mạo. Độ tin cậy: {score*100:.2f}%."
                else:
                    result_color = "#3fb950"
                    result_text = "✅ AN TOÀN: REAL"
                    sub_text = f"Khuôn mặt thật. Độ tin cậy: {(1-score)*100:.2f}%."

                st.markdown(f"""
                <div class="result-box" style="border-color: {result_color};">
                    <div class="card-title">Verification Result</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {result_color};">
                        {result_text}
                    </div>
                    <div style="color: #9aa4b2; margin-top: 8px;">
                        {sub_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Nếu là video, hiển thị aggregation_stats
                file_type = result_data.get("file_type", "image")
                agg = result_data.get("aggregation_stats")
                if file_type == "video" and agg:
                    st.info(f"""
                    **📊 Thống kê xử lý Video:**
                    - Tổng số frame đã phân tích: `{agg.get('total_frames_processed', '?')}`
                    - Điểm rủi ro cao nhất: `{agg.get('max_score', 0):.4f}`
                    - Điểm rủi ro thấp nhất: `{agg.get('min_score', 0):.4f}`
                    - Số frame không phát hiện mặt: `{agg.get('frames_no_face_detected', '?')}`
                    """)

                # Hiển thị JSON đầy đủ để debug
                with st.expander("🔍 Xem JSON đầy đủ từ backend"):
                    st.json(result_data)

            else:
                # 8. BACKEND TRẢ LỖI CÓ JSON
                error_msg = result_data.get('detail', 'Lỗi không xác định từ Backend')
                if isinstance(error_msg, list):
                    # FastAPI 422 trả list lỗi validation
                    error_msg = "; ".join([e.get("msg", "?") for e in error_msg])
                st.error(f"❌ Backend trả lỗi (HTTP {response.status_code}): {error_msg}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Không kết nối được tới Backend. Tunnel có thể đã die.")
            st.info("👉 Yêu cầu Huy chạy lại backend + tạo tunnel mới.")
        except requests.exceptions.Timeout:
            st.error("❌ Backend phản hồi quá chậm (timeout 120s).")
            st.info("👉 Có thể do video quá dài, hoặc backend bị treo.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Lỗi HTTP Request: `{str(e)}`")
        except Exception as e:
            st.error(f"❌ Lỗi hệ thống Frontend: `{str(e)}`")
            import traceback
            st.code(traceback.format_exc())
