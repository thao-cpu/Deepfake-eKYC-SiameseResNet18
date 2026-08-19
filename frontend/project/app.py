import streamlit as st

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Deepfake Detection")
st.write("Passive Liveness Detection using Siamese ResNet18")

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")
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