import streamlit as st

st.set_page_config(
    page_title="Liveness Detection",
    page_icon="🔐",
    layout="wide",
)

# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .header {
        padding: 10px 0 25px 0;
    }

    .title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9aa4b2;
        font-size: 1rem;
    }

    .status-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 22px;
        min-height: 210px;
    }

    .card-title {
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 18px;
    }

    .status-row {
        padding: 10px 0;
        border-bottom: 1px solid #252b33;
    }

    .status-label {
        color: #9aa4b2;
    }

    .ready {
        color: #3fb950;
        font-weight: 600;
    }

    .waiting {
        color: #d29922;
        font-weight: 600;
    }

    .info-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px;
        margin-top: 20px;
    }

    .result-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">
    <div class="title">🔐 Liveness Detection</div>
    <div class="subtitle">
        AI-powered real/fake verification system
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SYSTEM STATUS
# ============================================================

col_status, col_space = st.columns([1, 3])

with col_status:
    st.markdown(
        '<span class="ready">● SYSTEM READY</span>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# PASSIVE LIVENESS
# ============================================================
left, right = st.columns([1.25, 1])


# ------------------------------------------------------------
# LEFT: INPUT
# ------------------------------------------------------------

with left:

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


# ------------------------------------------------------------
# RIGHT: VERIFICATION STATUS
# ------------------------------------------------------------

with right:

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
# RESULT PLACEHOLDER
# ============================================================

if uploaded_file is not None:

    st.markdown("""
    <div class="result-box">

        <div class="card-title">
            Verification Result
        </div>

        <div style="font-size: 2rem; font-weight: 700;">
            ⏳ WAITING FOR BACKEND
        </div>

        <div style="color: #9aa4b2; margin-top: 8px;">
            Passive Liveness API has not been connected yet.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ACTIVE LIVENESS
# ============================================================

st.divider()

st.subheader("🎥 Active Liveness")

st.caption(
    "Realtime face verification using MediaPipe "
    "and interactive challenges."
)


# ============================================================
# ACTIVE LIVENESS STATUS
# ============================================================

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
        '<span class="ready">● Challenge Ready</span>',
        unsafe_allow_html=True
    )
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import random
import time
import cv2

from liveness.face_mesh_utils import (
    get_face_mesh,
    detect_blink,
    detect_head_turn,
)


CHALLENGES = [
    "blink",
    "turn_left",
    "turn_right",
]


class LivenessProcessor(VideoProcessorBase):

    def __init__(self):
        self.face_mesh = get_face_mesh()

        self.current_challenge = random.choice(
            CHALLENGES
        )

        self.challenge_start = time.time()
        self.challenge_timeout = 7

        self.status = "DETECTING"

        self.blink_counter = 0

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        # ====================================================
        # IMPORTANT:
        # face_mesh_utils.py mới tự xử lý BGR -> RGB.
        # Không convert sang RGB ở đây.
        # ====================================================

        results = self.face_mesh.process(img)

        # ====================================================
        # MediaPipe 0.10.35:
        #
        # results.face_landmarks
        #
        # Không còn:
        # results.multi_face_landmarks
        # ====================================================

        if (
            results.face_landmarks
            and self.status == "DETECTING"
        ):

            # FaceLandmarker trả về:
            # face_landmarks[0] = landmarks của face đầu tiên

            landmarks = results.face_landmarks[0]

            # =================================================
            # BLINK
            # =================================================

            if self.current_challenge == "blink":

                is_blink, ear = detect_blink(
                    landmarks
                )

                if is_blink:

                    self.blink_counter += 1

                    if self.blink_counter >= 1:
                        self.status = "PASS"

            # =================================================
            # HEAD TURN
            # =================================================

            elif self.current_challenge in (
                "turn_left",
                "turn_right",
            ):

                direction, yaw = detect_head_turn(
                    landmarks
                )

                expected = (
                    "left"
                    if self.current_challenge == "turn_left"
                    else "right"
                )

                if direction == expected:
                    self.status = "PASS"

            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for lm in landmarks:

                x = int(
                    lm.x * img.shape[1]
                )

                y = int(
                    lm.y * img.shape[0]
                )

                # Chỉ vẽ landmark nằm trong frame
                if (
                    0 <= x < img.shape[1]
                    and 0 <= y < img.shape[0]
                ):

                    cv2.circle(
                        img,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1,
                    )

        # ====================================================
        # TIMEOUT
        # ====================================================

        if (
            time.time() - self.challenge_start
            > self.challenge_timeout
            and self.status == "DETECTING"
        ):

            self.status = "FAIL"

        # ====================================================
        # INSTRUCTION
        # ====================================================

        label_map = {

            "blink":
                "Vui long chop mat",

            "turn_left":
                "Vui long quay dau sang TRAI",

            "turn_right":
                "Vui long quay dau sang PHAI",
        }

        cv2.putText(
            img,
            label_map[self.current_challenge],
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # ====================================================
        # STATUS
        # ====================================================

        if self.status == "PASS":
            status_color = (0, 255, 0)

        elif self.status == "FAIL":
            status_color = (0, 0, 255)

        else:
            status_color = (0, 255, 255)

        cv2.putText(
            img,
            f"Status: {self.status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        # ====================================================
        # RETURN FRAME
        # ====================================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

    def close(self):
        """
        Clean up MediaPipe FaceLandmarker.
        """
        if self.face_mesh is not None:
            self.face_mesh.close()


# ============================================================
# STREAMLIT WEBRTC
# ============================================================

ctx = webrtc_streamer(
    key="active-liveness",

    video_processor_factory=LivenessProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)


# ============================================================
# UI STATUS
# ============================================================

if ctx.video_processor:

    if ctx.video_processor.status == "PASS":

        st.success(
            "✅ Active Liveness PASS — "
            "Đã xác minh cử chỉ thành công"
        )

    elif ctx.video_processor.status == "FAIL":

        st.error(
            "❌ Active Liveness FAIL — "
            "Hết thời gian, vui lòng thử lại"
        )

        if st.button("🔄 Thử lại"):

            ctx.video_processor.status = "DETECTING"

            ctx.video_processor.current_challenge = (
                random.choice(CHALLENGES)
            )

            ctx.video_processor.challenge_start = (
                time.time()
            )

            ctx.video_processor.blink_counter = 0