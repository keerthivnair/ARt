import time

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe import Image, ImageFormat
from config import HAND_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, LANDMARK

MODEL_PATH = "models/hand_landmarker.task"


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]


class HandTracker:
    def __init__(self, max_hands=2):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._hands_count = 0
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def process(self, frame):
        """
        Synchronous per-frame detection. Requires a strictly increasing
        timestamp_ms per call, so we derive it from a monotonic clock
        rather than trusting the caller to track it.
        """
        image = Image(ImageFormat.SRGB, frame)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        return self.landmarker.detect_for_video(image, timestamp_ms)

    def get_landmarks(self, results):
        landmarks_list = []
        if results and results.hand_landmarks:
            self._hands_count = len(results.hand_landmarks)
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                lm = {}
                for name, idx in LANDMARK.items():
                    landmark = hand_landmarks[idx]
                    lm[name] = (landmark.x, landmark.y, landmark.z)
                
                raw_handedness = "Unknown"
                score = 1.0
                if results.handedness and i < len(results.handedness):
                    categories = results.handedness[i]
                    if categories:
                        raw_handedness = categories[0].category_name
                        score = categories[0].score

                # In mirrored camera view (cv2.flip(frame, 1)), MediaPipe's raw handedness is inverted.
                # A physical Left hand looks like a Right hand in 2D image coordinates, so raw "Right" = physical "Left".
                if raw_handedness == "Right":
                    physical_handedness = "Left"
                elif raw_handedness == "Left":
                    physical_handedness = "Right"
                else:
                    physical_handedness = "Unknown"

                lm["raw_handedness"] = raw_handedness
                lm["physical_handedness"] = physical_handedness
                lm["handedness"] = physical_handedness
                lm["score"] = score
                landmarks_list.append(lm)

            # Disambiguate if handedness is unknown or two hands have the same assigned handedness
            if len(landmarks_list) == 1:
                wrist_x = landmarks_list[0]["WRIST"][0]
                if landmarks_list[0]["handedness"] == "Unknown":
                    landmarks_list[0]["handedness"] = "Left" if wrist_x < 0.5 else "Right"
            elif len(landmarks_list) == 2:
                h0_wrist_x = landmarks_list[0]["WRIST"][0]
                h1_wrist_x = landmarks_list[1]["WRIST"][0]
                if landmarks_list[0]["handedness"] == landmarks_list[1]["handedness"]:
                    if h0_wrist_x < h1_wrist_x:
                        landmarks_list[0]["handedness"] = "Left"
                        landmarks_list[1]["handedness"] = "Right"
                    else:
                        landmarks_list[0]["handedness"] = "Right"
                        landmarks_list[1]["handedness"] = "Left"
        else:
            self._hands_count = 0
        return landmarks_list

    def draw_landmarks(self, frame, results):
        if not results or not results.hand_landmarks:
            return frame

        import cv2
        h, w, _ = frame.shape
        for hand_landmarks in results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        return frame

    def get_hands_count(self):
        return self._hands_count

    def release(self):
        self.landmarker.close()