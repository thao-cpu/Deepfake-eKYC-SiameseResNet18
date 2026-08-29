import os
import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "face_landmarker.task"
)


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1

LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Mouth landmarks for MAR
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


# ============================================================
# FACE LANDMARKER
# ============================================================

class FaceMeshWrapper:

    def __init__(self):

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"\nFace Landmarker model not found:\n"
                f"{MODEL_PATH}\n\n"
                f"Please put face_landmarker.task inside:\n"
                f"project/models/"
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.detector = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

    def process(self, frame):

        if frame is None:
            return None

        # OpenCV BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        return result

    def close(self):
        self.detector.close()


def get_face_mesh():
    """
    Keep the old function name so other modules
    do not need to change.
    """
    return FaceMeshWrapper()


# ============================================================
# GEOMETRY
# ============================================================

def _dist(p1, p2):

    return np.linalg.norm(
        np.array([p1.x, p1.y]) -
        np.array([p2.x, p2.y])
    )


# ============================================================
# EAR
# ============================================================

def eye_aspect_ratio(landmarks, eye_idx):

    pts = [landmarks[i] for i in eye_idx]

    vertical1 = _dist(pts[1], pts[5])
    vertical2 = _dist(pts[2], pts[4])

    horizontal = _dist(pts[0], pts[3])

    if horizontal < 1e-8:
        return 0.0

    return (
        vertical1 + vertical2
    ) / (2.0 * horizontal)


# ============================================================
# MAR
# ============================================================

def mouth_aspect_ratio(landmarks):

    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]

    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = _dist(top, bottom)
    horizontal = _dist(left, right)

    if horizontal < 1e-8:
        return 0.0

    return vertical / horizontal


# ============================================================
# YAW
# ============================================================

def estimate_yaw(landmarks):

    nose = landmarks[NOSE_TIP]
    left = landmarks[LEFT_CHEEK]
    right = landmarks[RIGHT_CHEEK]

    dist_left = _dist(nose, left)
    dist_right = _dist(nose, right)

    denominator = dist_right + dist_left

    if denominator < 1e-8:
        return 0.0

    return (
        dist_right - dist_left
    ) / denominator


# ============================================================
# BLINK
# ============================================================

def detect_blink(
    landmarks,
    threshold=0.21
):

    left_ear = eye_aspect_ratio(
        landmarks,
        LEFT_EYE
    )

    right_ear = eye_aspect_ratio(
        landmarks,
        RIGHT_EYE
    )

    avg_ear = (
        left_ear + right_ear
    ) / 2.0

    return avg_ear < threshold, avg_ear


# ============================================================
# HEAD TURN
# ============================================================

def detect_head_turn(
    landmarks,
    threshold=0.15
):

    yaw = estimate_yaw(landmarks)

    if yaw > threshold:
        return "right", yaw

    elif yaw < -threshold:
        return "left", yaw

    return "center", yaw


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_face_features(landmarks):

    blink, ear = detect_blink(landmarks)

    head_direction, yaw = detect_head_turn(
        landmarks
    )

    mar = mouth_aspect_ratio(
        landmarks
    )

    return {
        "ear": float(ear),
        "blink": bool(blink),
        "yaw": float(yaw),
        "head_direction": head_direction,
        "mar": float(mar),
    }


# ============================================================
# CLOSE HELPER
# ============================================================

def close_face_mesh(face_mesh):

    if face_mesh is not None:
        try:
            face_mesh.close()
        except Exception:
            pass