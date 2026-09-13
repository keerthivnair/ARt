import time

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe import Image, ImageFormat
from config import HAND_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, LANDMARK

MODEL_PATH = "models/hand_landmarker.task"


class HandTracker:
    def __init__(self, max_hands=1):
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
            for hand_landmarks in results.hand_landmarks:
                lm = {}
                for name, idx in LANDMARK.items():
                    landmark = hand_landmarks[idx]
                    lm[name] = (landmark.x, landmark.y, landmark.z)
                landmarks_list.append(lm)
        else:
            self._hands_count = 0
        return landmarks_list

    def draw_landmarks(self, frame, results):
        if not results or not results.hand_landmarks:
            return frame

        from mediapipe.framework.formats import landmark_pb2
        from mediapipe.python.solutions import drawing_utils, drawing_styles, hands as mp_hands

        for hand_landmarks in results.hand_landmarks:
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ])
            drawing_utils.draw_landmarks(
                frame,
                proto,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )
        return frame

    def get_hands_count(self):
        return self._hands_count

    def release(self):
        self.landmarker.close()