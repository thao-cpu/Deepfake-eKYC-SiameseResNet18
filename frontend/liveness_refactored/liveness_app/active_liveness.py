"""
Orchestrator cho toàn bộ khối Active Liveness.
"""

import time

import streamlit as st
from streamlit_webrtc import webrtc_streamer

from liveness_app.active_backend import call_active_backend
from liveness_app.liveness_processor import LivenessProcessor
from liveness_app.render_status import render_active_status
from liveness_app.virtual_camera import check_virtual_camera_and_get_permission


def render_active_liveness():
    st.divider()
    st.subheader("🎥 Active Liveness")
    st.caption(
        "Realtime face verification using MediaPipe "
        "and interactive challenges."
    )

    _render_ready_badges()
    _render_deepfake_toggle()

    camera_start_allowed = check_virtual_camera_and_get_permission()

    ctx, camera_placeholder = _setup_webrtc(camera_start_allowed)
    _sync_deepfake_mode_into_processor(ctx)
    _render_reset_button(ctx)
    _run_status_polling_and_handle_result(ctx, camera_placeholder)


# ============================================================
# ACTIVE LIVENESS STATUS (badges)
# ============================================================
def _render_ready_badges():
    active_col1, active_col2, active_col3 = st.columns(3)

    with active_col1:
        st.markdown(
            '<span class="ready">● Camera Ready</span>',
            unsafe_allow_html=True
        )

    with active_col2:
        st.markdown(
            '<span class="ready">● Face Detection Ready</span>',
            unsafe_allow_html=True
        )

    with active_col3:
        st.markdown(
            '<span class="ready">● Face Alignment Ready</span>',
            unsafe_allow_html=True
        )


# ============================================================
# NÚT CHUYỂN CHẾ ĐỘ DEEPFAKE FILTER
# ============================================================
def _render_deepfake_toggle():
    if "deepfake_mode_enabled" not in st.session_state:
        st.session_state.deepfake_mode_enabled = True

    st.markdown("")

    toggle_col, note_col = st.columns([1, 3])

    with toggle_col:
        mode_label = (
            "🛡️ Deepfake Filter: BẬT"
            if st.session_state.deepfake_mode_enabled
            else "⚪ Deepfake Filter: TẮT"
        )
        if st.button(mode_label, key="toggle_deepfake_mode", use_container_width=True):
            st.session_state.deepfake_mode_enabled = (
                not st.session_state.deepfake_mode_enabled
            )
            st.rerun()

    with note_col:
        if st.session_state.deepfake_mode_enabled:
            st.caption(
                "Đang BẬT: áp dụng filter artifact lên khuôn mặt để test model."
            )
        else:
            st.caption(
                "Đang TẮT: camera bình thường, không áp dụng filter."
            )

    # Dropdown chọn filter
    st.markdown("")
    filter_col1, filter_col2 = st.columns([1.5, 2.5])

    with filter_col1:
        from liveness_app.deepfake_filters import DEEPFAKE_FACE_FILTERS
        selected_filter = st.selectbox(
            "🎭 Chọn Artifact",
            options=list(DEEPFAKE_FACE_FILTERS.keys()),
            format_func=lambda x: DEEPFAKE_FACE_FILTERS[x][0],
            key="face_filter_select"
        )
        st.session_state.face_filter = selected_filter

    with filter_col2:
        st.caption("Chọn loại artifact deepfake muốn giả lập trên khuôn mặt.")


# ============================================================
# STREAMLIT WEBRTC
# ============================================================
def _create_liveness_processor():
    return LivenessProcessor(
        deepfake_enabled=st.session_state.get("deepfake_mode_enabled", False),
        face_filter=st.session_state.get("face_filter", "realistic")
    )


def _setup_webrtc(camera_start_allowed):
    camera_placeholder = st.empty()

    if not camera_start_allowed:
        with camera_placeholder:
            st.info(
                "📷 Camera đang bị khoá — vui lòng xác nhận webcam vật lý ở "
                "cảnh báo phía trên trước khi bắt đầu Active Liveness."
            )
        return None, camera_placeholder

    with camera_placeholder:
        ctx = webrtc_streamer(
            key="active-liveness",
            video_processor_factory=_create_liveness_processor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
        )
    return ctx, camera_placeholder


def _sync_deepfake_mode_into_processor(ctx):
    if not (ctx and ctx.video_processor):
        return

    _prev_mode = st.session_state.get("_prev_deepfake_mode_enabled")
    _now_mode = st.session_state.deepfake_mode_enabled

    if _prev_mode != _now_mode:
        with ctx.video_processor.deepfake_lock:
            ctx.video_processor.deepfake_scores = []
        ctx.video_processor.last_deepfake_input_jpeg = None
        st.session_state._prev_deepfake_mode_enabled = _now_mode

    ctx.video_processor.deepfake_enabled = _now_mode


# ============================================================
# NÚT RESET
# ============================================================
def _render_reset_button(ctx):
    if not (ctx and ctx.video_processor):
        return

    if st.button("🔄 Reset & Làm lại từ đầu", key="reset_active_liveness_full"):
        p = ctx.video_processor

        st.session_state.active_api_called = False
        st.session_state.pop("deepfake_verdict", None)

        p.status = "DETECTING"
        p.phase = "ALIGN"
        p.session_start = time.time()
        p.aligned_since = None
        p.current_challenge = None
        p.challenge_start = None
        p.blink_confirmed_count = 0
        p.eye_was_closed = False
        p.turn_confirm_count = 0
        p.blink_counter = 0
        p.look_up_confirm_count = 0
        p.look_down_confirm_count = 0
        p.tilt_left_confirm_count = 0
        p.tilt_right_confirm_count = 0
        with p.deepfake_lock:
            p.deepfake_scores = []
        p.last_deepfake_input_jpeg = None

        st.rerun()


# ============================================================
# UI STATUS & POLLING
# ============================================================
def _run_status_polling_and_handle_result(ctx, camera_placeholder):
    if "active_api_called" not in st.session_state:
        st.session_state.active_api_called = False

    if ctx and ctx.video_processor:
        _current_processor_id = id(ctx.video_processor)
        if st.session_state.get("_last_processor_id") != _current_processor_id:
            st.session_state._last_processor_id = _current_processor_id
            st.session_state.active_api_called = False
            st.session_state.pop("deepfake_verdict", None)

    status_box = st.empty()
    deepfake_input_preview = st.empty()

    if not (ctx and ctx.video_processor and not st.session_state.active_api_called):
        return

    processor = ctx.video_processor

    while ctx.state.playing and processor.status == "DETECTING":
        live_is_fake, live_avg, live_n = processor.get_deepfake_verdict()

        if processor.phase == "ALIGN":
            remaining = max(
                0.0,
                processor.align_timeout - (time.time() - processor.session_start),
            )
            hold_progress = None
            if processor.aligned_since is not None:
                hold_progress = (
                    time.time() - processor.aligned_since,
                    processor.oval_hold_required,
                )
            instruction = (
                "Giữ mặt trong vòng oval, đừng di chuyển"
                if processor.face_aligned
                else "Đưa khuôn mặt vào vòng oval hiển thị trên camera"
            )
        else:
            remaining = max(
                0.0,
                processor.challenge_timeout - (time.time() - processor.challenge_start),
            )
            hold_progress = None
            challenge_label_map = {
                "blink": "Chớp mắt (đưa mặt đã ổn định trong oval)",
                "turn_left": "Quay đầu sang TRÁI",
                "turn_right": "Quay đầu sang PHẢI",
            }
            instruction = challenge_label_map.get(
                processor.current_challenge, "Thực hiện challenge"
            )

        render_active_status(
            status_box,
            instruction,
            processor.status,
            remaining,
            "⏳ Chưa gọi",
            deepfake_info=(live_n, live_avg, live_is_fake)
            if processor.deepfake_enabled else None,
            hold_progress=hold_progress,
        )

        if processor.deepfake_enabled and processor.last_deepfake_input_jpeg:
            with deepfake_input_preview.container():
                st.caption("🛡️ Input model deepfake đang nhận:")
                st.image(processor.last_deepfake_input_jpeg, width=240)
        else:
            deepfake_input_preview.empty()

        time.sleep(0.3)

    final_status = processor.status

    if final_status == "PASS":
        _handle_pass(ctx, processor, camera_placeholder, status_box)
    elif final_status == "FAIL":
        _handle_fail(processor, status_box)
    else:
        _handle_camera_stopped(final_status, status_box)


# ============================================================
# TRƯỜNG HỢP 1: PASS
# ============================================================
def _handle_pass(ctx, processor, camera_placeholder, status_box):
    st.session_state.active_api_called = True

    render_active_status(
        status_box, "Đã căn mặt vào oval thành công", "PASS", 0.0, "🔄 Đang gọi..."
    )

    is_suspected_fake, avg_fake_score, n_samples = processor.get_deepfake_verdict()
    st.session_state.deepfake_verdict = {
        "is_suspected_fake": is_suspected_fake,
        "avg_score": avg_fake_score,
        "n_samples": n_samples,
    }

    camera_placeholder.empty()

    try:
        ctx.stop()
    except Exception:
        pass

    time.sleep(0.5)

    st.success("✅ Active Liveness PASS — Camera đã tự ngắt!")
    st.markdown("---")

    if n_samples > 0:
        df_color = "#f85149" if is_suspected_fake else "#3fb950"
        df_verdict_text = (
            "🚨 NGHI NGỜ DEEPFAKE" if is_suspected_fake else "✅ Không nghi ngờ"
        )
        st.markdown(f"""
        <div class="result-box" style="border-color: {df_color};">
            <div class="card-title">Deepfake Filter (Active Session) — TEST</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: {df_color};">
                {df_verdict_text}
            </div>
            <div style="color: #9aa4b2; margin-top: 8px;">
                Điểm fake trung bình: {avg_fake_score*100:.1f}%
                ({n_samples} lần check)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Deepfake Filter TẮT hoặc chưa check lần nào.")

    backend_is_real, api_result_status = call_active_backend(processor)

    render_active_status(
        status_box,
        "Đã căn mặt vào oval thành công",
        "PASS",
        0.0,
        api_result_status,
    )


# ============================================================
# TRƯỜNG HỢP 2: FAIL
# ============================================================
def _handle_fail(processor, status_box):
    render_active_status(
        status_box,
        "Không căn được mặt vào oval kịp thời", "FAIL", 0.0,
        "🚫 Không gọi (chưa xác thực được)"
    )
    st.error("❌ Active Liveness FAIL — Hết thời gian")

    if st.button("🔄 Thử lại"):
        st.session_state.active_api_called = False
        processor.status = "DETECTING"
        processor.phase = "ALIGN"
        processor.session_start = time.time()
        processor.aligned_since = None
        processor.current_challenge = None
        processor.challenge_start = None
        processor.blink_confirmed_count = 0
        processor.eye_was_closed = False
        processor.turn_confirm_count = 0
        processor.blink_counter = 0
        st.rerun()


# ============================================================
# TRƯỜNG HỢP 3: CAMERA DỪNG GIỮA CHỪNG
# ============================================================
def _handle_camera_stopped(final_status, status_box):
    render_active_status(
        status_box,
        "Camera đã dừng", final_status, 0.0,
        "⏹️ Camera đã dừng, chưa hoàn thành xác thực"
    )