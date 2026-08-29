"""
LivenessProcessor: video processor cho streamlit-webrtc, xử lý toàn bộ
luồng Active Liveness realtime (oval align -> challenge blink/turn/...),
đồng thời chạy nền một worker gọi API kiểm tra deepfake theo từng frame.
"""

import queue
import random
import threading
import time

import av
import cv2
import numpy as np
import requests
from streamlit_webrtc import VideoProcessorBase

from liveness.face_mesh_utils import (
    get_face_mesh,
    detect_blink,
    detect_head_turn,
)

from liveness_app.config import (
    CHALLENGES,
    DEEPFAKE_CHECK_URL,
    DEEPFAKE_CHECK_INTERVAL,
    DEEPFAKE_FAKE_THRESHOLD,
)
from liveness_app.deepfake_filters import apply_deepfake_face_filter


class LivenessProcessor(VideoProcessorBase):

    def __init__(self, deepfake_enabled=False, face_filter="realistic"):
        self.face_mesh = None

        # --- Giai đoạn 1: Oval face-framing ---
        self.phase = "ALIGN"
        self.oval_hold_required = 1.5
        self.align_timeout = 25.0
        self.session_start = time.time()
        self.aligned_since = None
        self.face_aligned = False

        # --- Giai đoạn 2: Challenge ---
        self.current_challenge = None
        self.challenge_start = None
        self.challenge_timeout = 10.0
        self.challenge_grace = 0.6
        self.phase_switch_time = None
        self.blink_confirmed_count = 0
        self.eye_was_closed = False
        self.turn_confirm_count = 0
        self.turn_confirm_required = 3
        self.look_up_confirm_count = 0
        self.look_up_confirm_required = 5
        self.look_down_confirm_count = 0
        self.look_down_confirm_required = 5
        self.tilt_left_confirm_count = 0
        self.tilt_left_confirm_required = 5
        self.tilt_right_confirm_count = 0
        self.tilt_right_confirm_required = 5

        self.status = "DETECTING"
        self.blink_counter = 0

        # --- Giảm lag face mesh ---
        self.frame_count = 0
        self.process_every_n = 2
        self.last_landmarks = None

        # --- Deepfake ---
        self.deepfake_enabled = deepfake_enabled
        self.face_filter = face_filter
        self.last_frame = None
        self.last_frame_clean = None
        self.last_frame_fake = None
        self.last_deepfake_input_jpeg = None
        self.deepfake_scores = []
        self.deepfake_lock = threading.Lock()
        self.deepfake_queue = queue.Queue(maxsize=1)
        self.last_deepfake_check = 0.0
        self.deepfake_stop = threading.Event()
        self.deepfake_thread = threading.Thread(
            target=self._deepfake_worker, daemon=True
        )
        self.deepfake_thread.start()

    def _deepfake_worker(self):
        while not self.deepfake_stop.is_set():
            try:
                frame_bytes = self.deepfake_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame_bytes is None:
                continue
            try:
                files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
                resp = requests.post(DEEPFAKE_CHECK_URL, files=files, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    is_fake = data.get("is_deepfake")
                    score = data.get("confidence_score")
                    if score is not None:
                        with self.deepfake_lock:
                            self.deepfake_scores.append((is_fake, score))
            except Exception as e:
                print(f"[deepfake_worker] lỗi: {e}")

    def get_deepfake_verdict(self):
        with self.deepfake_lock:
            scores = list(self.deepfake_scores)
        if not scores:
            return False, 0.0, 0
        avg_score = sum(s for _, s in scores) / len(scores)
        n_fake_votes = sum(1 for is_fake, _ in scores if is_fake)
        is_suspected_fake = (
            avg_score > DEEPFAKE_FAKE_THRESHOLD
            or n_fake_votes > len(scores) / 2
        )
        return is_suspected_fake, avg_score, len(scores)

    def get_last_frame_jpeg(self, quality=90):
        frame = self.last_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return buf.tobytes()

    def recv(self, frame):
        if self.face_mesh is None:
            self.face_mesh = get_face_mesh()

        img = frame.to_ndarray(format="bgr24")
        self.last_frame_clean = img

        if self.status != "DETECTING":
            return frame

        self.frame_count += 1

        # --- Gửi frame cho API deepfake (background thread) ---
        if self.deepfake_enabled:
            now = time.time()
            if now - self.last_deepfake_check >= DEEPFAKE_CHECK_INTERVAL:
                self.last_deepfake_check = now
                small_img = cv2.resize(img, (320, 240))
                ok, buf = cv2.imencode(".jpg", small_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    self.last_deepfake_input_jpeg = buf.tobytes()
                    try:
                        self.deepfake_queue.put_nowait(self.last_deepfake_input_jpeg)
                    except queue.Full:
                        pass

        # --- Face mesh detection ---
        try:
            if self.frame_count % self.process_every_n == 0 or self.last_landmarks is None:
                results = self.face_mesh.process(img)
                if results and results.face_landmarks:
                    self.last_landmarks = results.face_landmarks[0]
                else:
                    self.last_landmarks = None
        except Exception as e:
            print(f"Lỗi face mesh: {e}")
            cv2.putText(img, f"Loi: {str(e)[:40]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img_h, img_w = img.shape[:2]
        oval_center = (img_w // 2, img_h // 2)
        oval_axes = (int(img_w * 0.22), int(img_h * 0.32))
        is_aligned_now = False
        landmarks = self.last_landmarks

        # === GIAI ĐOẠN 1: ALIGN ===
        if self.phase == "ALIGN" and self.status == "DETECTING":
            if landmarks is not None:
                nose = landmarks[1]
                left_cheek = landmarks[234]
                right_cheek = landmarks[454]
                nose_x = int(nose.x * img_w)
                nose_y = int(nose.y * img_h)
                face_width_px = abs(right_cheek.x - left_cheek.x) * img_w
                dx = (nose_x - oval_center[0]) / oval_axes[0]
                dy = (nose_y - oval_center[1]) / oval_axes[1]
                nose_in_oval = (dx * dx + dy * dy) <= 1.0
                min_face_w = oval_axes[0] * 1.1
                max_face_w = oval_axes[0] * 2.2
                size_ok = min_face_w <= face_width_px <= max_face_w
                is_aligned_now = nose_in_oval and size_ok

                if is_aligned_now:
                    if self.aligned_since is None:
                        self.aligned_since = time.time()
                    held_seconds = time.time() - self.aligned_since
                    if held_seconds >= self.oval_hold_required:
                        self.phase = "CHALLENGE"
                        self.current_challenge = random.choice(CHALLENGES)
                        self.challenge_start = time.time()
                        self.phase_switch_time = time.time()
                        self.blink_confirmed_count = 0
                        self.eye_was_closed = False
                        self.turn_confirm_count = 0
                        self.look_up_confirm_count = 0
                        self.look_down_confirm_count = 0
                        self.tilt_left_confirm_count = 0
                        self.tilt_right_confirm_count = 0
                else:
                    self.aligned_since = None

                for lm in landmarks:
                    x = int(lm.x * img_w)
                    y = int(lm.y * img_h)
                    if 0 <= x < img_w and 0 <= y < img_h:
                        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            self.face_aligned = is_aligned_now
            oval_color = (0, 255, 0) if is_aligned_now else (0, 200, 255)
            cv2.ellipse(img, oval_center, oval_axes, 0, 0, 360, oval_color, 3)

            if self.aligned_since is not None:
                held = time.time() - self.aligned_since
                cv2.putText(img, f"Giu yen: {held:.1f}s / {self.oval_hold_required:.1f}s",
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if time.time() - self.session_start > self.align_timeout and self.status == "DETECTING":
                self.status = "FAIL"

            instruction = "Dua khuon mat vao vong oval va giu yen" if not is_aligned_now else "Dang giu yen... khong di chuyen"
            cv2.putText(img, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # === GIAI ĐOẠN 2: CHALLENGE ===
        elif self.phase == "CHALLENGE" and self.status == "DETECTING":
            cv2.ellipse(img, oval_center, oval_axes, 0, 0, 360, (120, 120, 120), 2)
            in_grace = (time.time() - self.phase_switch_time) < self.challenge_grace

            if landmarks is not None and not in_grace:
                if self.current_challenge == "blink":
                    is_eye_closed, _ear = detect_blink(landmarks)
                    if is_eye_closed:
                        self.eye_was_closed = True
                    else:
                        if self.eye_was_closed:
                            self.blink_confirmed_count += 1
                            self.eye_was_closed = False
                    if self.blink_confirmed_count >= 1:
                        self.status = "PASS"

                elif self.current_challenge in ("turn_left", "turn_right"):
                    direction, _yaw = detect_head_turn(landmarks)
                    expected = "left" if self.current_challenge == "turn_left" else "right"
                    if direction == expected:
                        self.turn_confirm_count += 1
                    else:
                        self.turn_confirm_count = max(0, self.turn_confirm_count - 1)
                    if self.turn_confirm_count >= self.turn_confirm_required:
                        self.status = "PASS"

                elif self.current_challenge in ("look_up", "look_down"):
                    nose = landmarks[1]
                    left_eye = landmarks[159]
                    right_eye = landmarks[386]
                    chin = landmarks[152]
                    forehead = landmarks[10]
                    eye_line_y = (left_eye.y + right_eye.y) / 2
                    face_h = abs(chin.y - forehead.y) * img_h
                    nose_to_eyes_dist = abs(nose.y - eye_line_y) * img_h
                    look_ratio = nose_to_eyes_dist / face_h if face_h > 0 else 0
                    if self.current_challenge == "look_up" and look_ratio < 0.22:
                        self.look_up_confirm_count += 1
                    elif self.current_challenge == "look_down" and look_ratio > 0.42:
                        self.look_down_confirm_count += 1
                    else:
                        if self.current_challenge == "look_up":
                            self.look_up_confirm_count = max(0, self.look_up_confirm_count - 1)
                        else:
                            self.look_down_confirm_count = max(0, self.look_down_confirm_count - 1)
                    if self.current_challenge == "look_up" and self.look_up_confirm_count >= self.look_up_confirm_required:
                        self.status = "PASS"
                    if self.current_challenge == "look_down" and self.look_down_confirm_count >= self.look_down_confirm_required:
                        self.status = "PASS"

                elif self.current_challenge in ("tilt_left", "tilt_right"):
                    left_eye = landmarks[159]
                    right_eye = landmarks[386]
                    left_cheek = landmarks[234]
                    right_cheek = landmarks[454]
                    face_w = abs(right_cheek.x - left_cheek.x) * img_w
                    eye_diff = (right_eye.y - left_eye.y) * img_h
                    tilt_ratio = eye_diff / face_w if face_w > 0 else 0
                    if self.current_challenge == "tilt_left" and tilt_ratio < -0.04:
                        self.tilt_left_confirm_count += 1
                    elif self.current_challenge == "tilt_right" and tilt_ratio > 0.04:
                        self.tilt_right_confirm_count += 1
                    else:
                        if self.current_challenge == "tilt_left":
                            self.tilt_left_confirm_count = max(0, self.tilt_left_confirm_count - 1)
                        else:
                            self.tilt_right_confirm_count = max(0, self.tilt_right_confirm_count - 1)
                    if self.current_challenge == "tilt_left" and self.tilt_left_confirm_count >= self.tilt_left_confirm_required:
                        self.status = "PASS"
                    if self.current_challenge == "tilt_right" and self.tilt_right_confirm_count >= self.tilt_right_confirm_required:
                        self.status = "PASS"

                for lm in landmarks:
                    x = int(lm.x * img_w)
                    y = int(lm.y * img_h)
                    if 0 <= x < img_w and 0 <= y < img_h:
                        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            if time.time() - self.challenge_start > self.challenge_timeout and self.status == "DETECTING":
                self.status = "FAIL"

            label_map = {
                "blink": "Vui long CHOP MAT",
                "turn_left": "Vui long quay dau sang TRAI",
                "turn_right": "Vui long quay dau sang PHAI",
                "look_up": "Vui long NHIN LEN TREN",
                "look_down": "Vui long CÚI ĐẦU XUỐNG",
                "tilt_left": "Vui long NGHIÊNG ĐẦU SANG TRAI",
                "tilt_right": "Vui long NGHIÊNG ĐẦU SANG PHAI",
            }
            instruction = label_map.get(self.current_challenge, "...")
            if in_grace:
                instruction = "Chuan bi... " + instruction
            cv2.putText(img, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # === STATUS ===
        if self.status == "PASS":
            status_color = (0, 255, 0)
        elif self.status == "FAIL":
            status_color = (0, 0, 255)
        else:
            status_color = (0, 255, 255)
        cv2.putText(img, f"Status: {self.status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # === APPLY FILTER & RETURN ===
        if self.deepfake_enabled and self.last_landmarks is not None:
            img = apply_deepfake_face_filter(img, self.last_landmarks, self.face_filter)
            self.last_frame_fake = img

        self.last_frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def close(self):
        self.deepfake_stop.set()
        try:
            self.deepfake_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.face_mesh is not None:
            self.face_mesh.close()