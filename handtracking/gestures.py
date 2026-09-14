import numpy as np
from config import (
    PINCH_THRESHOLD,
)


class GestureRecognizer:
    def __init__(self):
        self.current_gesture = "none"
        self._pinching = False
        self._open_palm = False
        self.scaling_mode = False

    def analyze(self, landmarks):
        if not landmarks:
            self.current_gesture = "none"
            self._pinching = False
            self._open_palm = False
            self.scaling_mode = False
            return self.current_gesture

        fingers_up = self._count_fingers_up(landmarks)
        self._open_palm = fingers_up >= 4
        self._pinching = self._is_pinching(landmarks)
        dist = self.get_pinch_distance(landmarks)
        
        is_only_index = (fingers_up == 1 and self._is_index_pointing(landmarks))
        is_thumb_and_index = (fingers_up == 2 and self._is_thumb_and_index_up(landmarks))

        if self._pinching or (is_thumb_and_index and dist < 0.2):
            self.scaling_mode = True
            
        if fingers_up == 0 or fingers_up >= 4:
            self.scaling_mode = False
            
        if is_only_index and not self._pinching:
             self.scaling_mode = False

        if self.scaling_mode and dist < 0.2:
            self.current_gesture = "pinch"
        elif fingers_up == 0:
            self.current_gesture = "fist"
        elif is_only_index:
            self.current_gesture = "point"
        elif is_thumb_and_index and dist >= 0.2:
            self.current_gesture = "point"
        elif fingers_up >= 4:
            self.current_gesture = "open_palm"
        else:
            self.current_gesture = "open_palm" if fingers_up >= 2 else "fist"
            self.scaling_mode = False

        return self.current_gesture

    def _count_fingers_up(self, landmarks):
        count = 0
        if self._is_finger_up(landmarks, "THUMB_TIP", "THUMB_MCP", landmarks["WRIST"]):
            count += 1
        for tip_name, pip_name in zip(["INDEX_TIP", "MIDDLE_TIP", "RING_TIP", "PINKY_TIP"],
                                       ["INDEX_PIP", "MIDDLE_PIP", "RING_PIP", "PINKY_PIP"]):
            if self._is_finger_up(landmarks, tip_name, pip_name, landmarks["WRIST"]):
                count += 1
        return count

    def _is_finger_up(self, landmarks, tip_name, pip_name, wrist):
        tip = landmarks[tip_name]
        pip = landmarks[pip_name]
        return tip[1] < pip[1] and tip[1] < wrist[1]

    def _is_pinching(self, landmarks):
        thumb_tip = landmarks["THUMB_TIP"]
        index_tip = landmarks["INDEX_TIP"]
        dist = np.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
        return dist < PINCH_THRESHOLD

    def _is_index_pointing(self, landmarks):
        index_tip = landmarks["INDEX_TIP"]
        middle_tip = landmarks["MIDDLE_TIP"]
        return index_tip[1] < middle_tip[1]

    def _is_thumb_and_index_up(self, landmarks):
        wrist = landmarks["WRIST"]
        thumb_up = self._is_finger_up(landmarks, "THUMB_TIP", "THUMB_MCP", wrist)
        index_up = self._is_finger_up(landmarks, "INDEX_TIP", "INDEX_PIP", wrist)
        middle_up = self._is_finger_up(landmarks, "MIDDLE_TIP", "MIDDLE_PIP", wrist)
        ring_up = self._is_finger_up(landmarks, "RING_TIP", "RING_PIP", wrist)
        pinky_up = self._is_finger_up(landmarks, "PINKY_TIP", "PINKY_PIP", wrist)
        
        return thumb_up and index_up and not middle_up and not ring_up and not pinky_up

    def get_pinch_distance(self, landmarks):
        if len(landmarks) == 0:
            return 0.0
        thumb_tip = landmarks["THUMB_TIP"]
        index_tip = landmarks["INDEX_TIP"]
        return np.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)

    def is_pinching(self):
        return self._pinching

    def is_open_palm(self):
        return self._open_palm
