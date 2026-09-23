import cv2
import numpy as np
import time
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    PROCESS_WIDTH,
    PROCESS_HEIGHT,
    DRAWING_COLOR,
    COLOR_PALETTE,
)
from handtracking.tracker import HandTracker
from handtracking.gestures import GestureRecognizer
from drawing.canvas import DrawingCanvas


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracker = HandTracker(max_hands=2)
    gesture_recognizers = {
        "Left": GestureRecognizer(switch_threshold=2),
        "Right": GestureRecognizer(switch_threshold=2),
    }
    canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)

    process_frame = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8)

    print("Hand Drawing System started. Press 'q' to quit.")
    print("Point index finger to draw.")
    print("Pinch with LEFT hand to MOVE (pan) drawing around.")
    print("Pinch with RIGHT hand to SCALE (zoom) drawing.")
    print("Point index & middle finger to erase.")
    print("Press 'c' to clear canvas. Press 'n' for next color.")

    fps_time = time.time()
    fps = 0
    frame_count = 0

    prev_left_pinch_pos = None
    prev_right_pinch_dist = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

        frame_count += 1
        process_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

        results = tracker.process(process_frame)
        landmarks = tracker.get_landmarks(results)

        # Draw hand skeleton overlay
        tracker.draw_landmarks(frame, results)

        active_gestures = {}
        left_pinching = False
        right_pinching = False
        drawing_active = False

        if landmarks:
            for lm in landmarks:
                hand_side = lm.get("handedness", "Right")
                if hand_side not in gesture_recognizers:
                    gesture_recognizers[hand_side] = GestureRecognizer(switch_threshold=2)

                gesture = gesture_recognizers[hand_side].analyze(lm)
                active_gestures[hand_side] = gesture

                index_tip = lm["INDEX_TIP"]
                thumb_tip = lm["THUMB_TIP"]
                cx = int(index_tip[0] * SCREEN_WIDTH)
                cy = int(index_tip[1] * SCREEN_HEIGHT)
                tx_thumb = int(thumb_tip[0] * SCREEN_WIDTH)
                ty_thumb = int(thumb_tip[1] * SCREEN_HEIGHT)

                # 1. Left Hand Pinch -> Move / Pan Drawing Canvas
                if hand_side == "Left" and gesture == "pinch":
                    left_pinching = True
                    canvas.finalize_active_stroke()
                    pinch_pos = ((cx + tx_thumb) // 2, (cy + ty_thumb) // 2)

                    if prev_left_pinch_pos is not None:
                        dx = pinch_pos[0] - prev_left_pinch_pos[0]
                        dy = pinch_pos[1] - prev_left_pinch_pos[1]
                        canvas.transform.tx += dx
                        canvas.transform.ty += dy
                    prev_left_pinch_pos = pinch_pos

                    # Draw move feedback
                    cv2.line(frame, (cx, cy), (tx_thumb, ty_thumb), (255, 0, 255), 3)
                    cv2.circle(frame, pinch_pos, 10, (255, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"Moving ({int(canvas.transform.tx)}, {int(canvas.transform.ty)})",
                        (pinch_pos[0] + 15, pinch_pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                    )

                # 2. Point gesture -> Draw
                elif gesture == "point":
                    drawing_active = True
                    canvas.add_point_to_active_stroke((cx, cy))
                    cv2.circle(frame, (cx, cy), 8, canvas.color, -1)

                # 3. Erase gesture -> Erase strokes
                elif gesture == "erase":
                    canvas.finalize_active_stroke()
                    canvas.remove_points_from_strokes((cx, cy), radius=15)
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

                # 4. Right Hand Pinch -> Scale Zoom
                elif hand_side == "Right" and gesture == "pinch":
                    right_pinching = True
                    canvas.finalize_active_stroke()
                    dist = gesture_recognizers["Right"].get_pinch_distance(lm)
                    if prev_right_pinch_dist is not None:
                        delta = dist - prev_right_pinch_dist
                        canvas.transform.scale += delta * 5.0
                        canvas.transform.scale = max(0.1, min(canvas.transform.scale, 10.0))
                    prev_right_pinch_dist = dist

                    # Draw scale feedback
                    pinch_pos = ((cx + tx_thumb) // 2, (cy + ty_thumb) // 2)
                    cv2.line(frame, (cx, cy), (tx_thumb, ty_thumb), (0, 255, 255), 3)
                    cv2.circle(frame, pinch_pos, 10, (0, 255, 255), -1)

        if not left_pinching:
            prev_left_pinch_pos = None

        if not right_pinching:
            prev_right_pinch_dist = None

        if not drawing_active:
            canvas.finalize_active_stroke()

        fps_now = time.time()
        elapsed = fps_now - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = fps_now

        # UI Header Info
        fps_text = f"FPS: {fps:.0f} | Color: {canvas.color_index + 1}/{len(COLOR_PALETTE)}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        status_str = " | ".join([f"{h}: {g}" for h, g in active_gestures.items()]) if active_gestures else "No hands"
        cv2.putText(
            frame,
            f"Hands: {status_str}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if active_gestures else (0, 0, 255),
            2,
        )

        transform_text = f"Scale: {canvas.transform.scale:.2f}x | Pan: ({int(canvas.transform.tx)}, {int(canvas.transform.ty)})"
        cv2.putText(
            frame,
            transform_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )

        foreground = canvas.re_render_frame()
        mask = canvas.get_foreground_mask()

        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        combined = cv2.add(frame_bg, foreground)

        color_msg = "Pinch Left Hand = Move Drawing | Pinch Right Hand = Scale | Point = Draw | 'n'=color | 'c'=clear | 'q'=quit"
        cv2.putText(
            combined,
            color_msg,
            (10, SCREEN_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        cv2.imshow("ARt - Hand Drawing", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas.clear()
            print("Canvas cleared.")
        elif key == ord("n"):
            color = canvas.next_color()
            print(f"Color changed to index {canvas.color_index}: {color}")

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
