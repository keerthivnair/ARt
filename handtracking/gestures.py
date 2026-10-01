import math

import numpy as np

from config import PINCH_THRESHOLD

FINGER_KEYS = ["INDEX", "MIDDLE", "RING", "PINKY"]


class GestureRecognizer:
    """Classifies one hand into the gesture vocabulary used by the 3D app.

    Recognised gestures: ``point``, ``pinch``, ``two_fingers``,
    ``three_fingers``, ``rotate``, ``open_palm``, ``fist`` and ``none``.
    """

    def __init__(self, switch_threshold=2):
        self.current_gesture = "none"
        self.previous_gesture = "none"
        self.return_gesture = "none"
        self.gesture_switch_threshold = switch_threshold
        self.gesture_persist = 0
        self._pinch_active = False

    # -- public API -------------------------------------------------------
    def analyze(self, landmarks):
        if not landmarks:
            self.current_gesture = "none"
            self._pinch_active = False
            self._update_persistence()
            return self.return_gesture

        fingers = self._finger_states(landmarks)
        index = fingers["INDEX"]
        middle = fingers["MIDDLE"]
        ring = fingers["RING"]
        pinky = fingers["PINKY"]

        pinch_ready = self._is_thumb_up(landmarks) and index
        distance = self.get_pinch_distance(landmarks)

        # Sticky pinch: only re-armed once thumb and index fully separate, so a
        # pinch is not dropped mid-drag when the fingers drift together.
        if pinch_ready and distance < PINCH_THRESHOLD:
            self._pinch_active = True
        elif not pinch_ready:
            self._pinch_active = False

        if self._pinch_active:
            self.current_gesture = "pinch"
        elif index and self._is_index_pointing(landmarks) and not (middle or ring or pinky):
            self.current_gesture = "point"
        elif index and middle and not ring and not pinky:
            self.current_gesture = "two_fingers"
        elif index and middle and ring and not pinky:
            self.current_gesture = "three_fingers"
        elif index and pinky and not middle and not ring:
            self.current_gesture = "rotate"
        elif index and middle and ring and pinky:
            self.current_gesture = "open_palm"
        elif not (index or middle or ring or pinky):
            self.current_gesture = "fist"
        else:
            self.current_gesture = "none"

        self._update_persistence()
        return self.return_gesture

    def get_pinch_distance(self, landmarks):
        if not landmarks:
            return 0.0
        thumb_tip = landmarks["THUMB_TIP"]
        index_tip = landmarks["INDEX_TIP"]
        return math.sqrt(
            (thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2
        )

    def get_hand_roll_angle(self, landmarks):
        """Screen-space roll of the hand, measured across the knuckle line.

        The index MCP to pinky MCP axis stays defined for every gesture, so it
        can be sampled continuously while the rotate gesture is held.
        """
        if not landmarks:
            return 0.0
        start = landmarks["INDEX_MCP"]
        end = landmarks["PINKY_MCP"]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if math.hypot(dx, dy) < 1e-6:
            return 0.0
        return math.atan2(dy, dx)

    def get_finger_states(self, landmarks):
        return self._finger_states(landmarks)

    # -- internals --------------------------------------------------------
    def _update_persistence(self):
        if self.previous_gesture == self.current_gesture:
            self.gesture_persist += 1
            if self.gesture_persist >= self.gesture_switch_threshold:
                self.return_gesture = self.current_gesture
        else:
            self.gesture_persist = 1
        self.previous_gesture = self.current_gesture

    def _finger_states(self, landmarks):
        wrist = landmarks["WRIST"]
        states = {}
        for finger in FINGER_KEYS:
            states[finger] = self._is_finger_up(
                landmarks, f"{finger}_TIP", f"{finger}_PIP", wrist
            )
        return states

    def _is_finger_up(self, landmarks, tip_name, pip_name, wrist):
        # The four fingers extend roughly vertically in the image, so a tip
        # above both its PIP joint and the wrist reads as extended.
        tip = landmarks[tip_name]
        pip = landmarks[pip_name]
        return tip[1] < pip[1] and tip[1] < wrist[1]

    def _is_thumb_up(self, landmarks):
        # The thumb extends sideways, so it is compared on x rather than y.
        thumb_tip = landmarks["THUMB_TIP"]
        thumb_mcp = landmarks["THUMB_MCP"]
        wrist = landmarks["WRIST"]
        return abs(thumb_tip[0] - wrist[0]) > abs(thumb_mcp[0] - wrist[0])

    def _is_index_pointing(self, landmarks):
        return landmarks["INDEX_TIP"][1] < landmarks["INDEX_DIP"][1]
