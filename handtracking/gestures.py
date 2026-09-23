import numpy as np
from config import (
    PINCH_THRESHOLD,
)


class GestureRecognizer:
    def __init__(self):
        self.current_gesture = "none"
        self.previous_gesture = 'none'
        self.return_gesture = 'none'
        self.gesture_switch_threshold = 5
        self.gesture_persist = 0
        self._pinch_active = False          # NEW: sticky state

    def analyze(self, landmarks):
        if not landmarks:
            self.current_gesture = "none"
            self._pinch_active = False       # NEW: reset on hand loss
            self._update_persistence()
            return self.return_gesture

        fingers_up = self._count_fingers_up(landmarks)
        self._open_palm = fingers_up > 4
        dist = self.get_pinch_distance(landmarks)
        print(fingers_up, dist)

        is_only_index = (fingers_up == 1 and self._is_index_pointing(landmarks))
        is_only_thumb_and_index = (fingers_up == 2 and self._is_thumb_and_index_up(landmarks))
        is_only_index_and_middle = (fingers_up == 2 and self._is_index_pointing(landmarks) and self._is_middle_pointing(landmarks))

        # NEW: sticky pinch state — only re-armed once fingers separate fully
        if is_only_thumb_and_index and dist < PINCH_THRESHOLD:
            self._pinch_active = True
        elif not is_only_thumb_and_index:
            self._pinch_active = False

        if is_only_index:
            self.current_gesture = 'point'
        elif is_only_index_and_middle:
            self.current_gesture = 'erase'
        elif is_only_thumb_and_index and self._pinch_active:   # CHANGED
            self.current_gesture = 'pinch'
        elif fingers_up > 4:
            self.current_gesture = 'open_palm'
        elif fingers_up < 1:
            self.current_gesture = 'fist'
        else:
            self.current_gesture = 'none'

        self._update_persistence()
        return self.return_gesture

    def _update_persistence(self):
        if self.previous_gesture == self.current_gesture:
            self.gesture_persist += 1
            if self.gesture_persist >= self.gesture_switch_threshold:
                self.return_gesture = self.current_gesture
        else:
            self.gesture_persist = 1
        self.previous_gesture = self.current_gesture

    def _count_fingers_up(self, landmarks):
        count = 0
        if self._is_thumb_up(landmarks):
            count += 1
        for tip_name, pip_name in zip(["INDEX_TIP", "MIDDLE_TIP", "RING_TIP", "PINKY_TIP"],
                                       ["INDEX_PIP", "MIDDLE_PIP", "RING_PIP", "PINKY_PIP"]):
            if self._is_finger_up(landmarks, tip_name, pip_name, landmarks["WRIST"]):
                count += 1
        return count

    def _is_finger_up(self, landmarks, tip_name, pip_name, wrist):
        # Valid for index/middle/ring/pinky: they extend roughly vertically,
        # so a smaller y at the tip than at the pip/wrist means "up".
        tip = landmarks[tip_name]
        pip = landmarks[pip_name]
        return tip[1] < pip[1] and tip[1] < wrist[1]

    def _is_thumb_up(self, landmarks):
        # FIX: the thumb extends sideways (x-axis), not vertically, so the
        # generic y-axis _is_finger_up check was wrong for it and was
        # corrupting fingers_up for fist / pinch / open_palm detection.
        # Instead, check whether the tip is farther from the wrist on the
        # x-axis than the MCP joint is — this works regardless of whether
        # it's a left or right hand in frame.
        thumb_tip = landmarks["THUMB_TIP"]
        thumb_mcp = landmarks["THUMB_MCP"]
        wrist = landmarks["WRIST"]
        return abs(thumb_tip[0] - wrist[0]) > abs(thumb_mcp[0] - wrist[0])

    def _is_pinching(self, landmarks):
        thumb_tip = landmarks["THUMB_TIP"]
        index_tip = landmarks["INDEX_TIP"]
        dist = np.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
        return dist < PINCH_THRESHOLD

    def _is_index_pointing(self, landmarks):
        index_tip = landmarks["INDEX_TIP"]
        index_dip = landmarks["INDEX_DIP"]
        return index_tip[1] < index_dip[1]

    def _is_middle_pointing(self, landmarks):
        middle_tip = landmarks['MIDDLE_TIP']
        middle_dip = landmarks['MIDDLE_DIP']
        return middle_tip[1] < middle_dip[1]

    def _is_thumb_and_index_up(self, landmarks):
        wrist = landmarks["WRIST"]
        thumb_up = self._is_thumb_up(landmarks)
        index_up = self._is_finger_up(landmarks, "INDEX_TIP", "INDEX_PIP", wrist)
        return thumb_up and index_up

    def get_pinch_distance(self, landmarks):
        if len(landmarks) == 0:
            return 0.0
        thumb_tip = landmarks["THUMB_TIP"]
        index_tip = landmarks["INDEX_TIP"]
        return np.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
