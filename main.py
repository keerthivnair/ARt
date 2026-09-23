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

    print("ARt System started. Press 'q' to quit.")
    print("RIGHT HAND: Point to draw, Pinch to scale zoom, Erase with 2 fingers.")
    print("LEFT HAND: Point to draw a SELECTION CIRCLE around a drawing.")
    print("LEFT HAND: Pinch to MOVE selected drawing (or whole canvas).")
    print("Press 'd' or show Open Palm to deselect. Press 'c' to clear canvas.")

    fps_time = time.time()
    fps = 0
    frame_count = 0

    prev_left_pinch_pos = None
    prev_right_pinch_dist = None
    left_lasso_points = []

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

        # Draw hand skeleton overlay natively using OpenCV
        tracker.draw_landmarks(frame, results)

        active_gestures = {}
        left_pinching = False
        right_pinching = False
        drawing_active = False
        left_pointing = False

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

                # =========================================================
                # LEFT HAND CONTROLS (Selection & Selection Panning)
                # =========================================================
                if hand_side == "Left":
                    if gesture == "point":
                        left_pointing = True
                        canvas.finalize_active_stroke()
                        left_lasso_points.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 6, (255, 255, 0), -1)

                    elif gesture == "pinch":
                        left_pinching = True
                        canvas.finalize_active_stroke()
                        pinch_pos = ((cx + tx_thumb) // 2, (cy + ty_thumb) // 2)

                        if prev_left_pinch_pos is not None:
                            dx = pinch_pos[0] - prev_left_pinch_pos[0]
                            dy = pinch_pos[1] - prev_left_pinch_pos[1]

                            if canvas.has_selection():
                                canvas.move_selected_strokes(dx, dy)
                            else:
                                canvas.transform.tx += dx
                                canvas.transform.ty += dy

                        prev_left_pinch_pos = pinch_pos

                        # Draw visual indicator for Left Hand Pinch Move
                        color = (255, 255, 0) if canvas.has_selection() else (255, 0, 255)
                        label = "Move Selection" if canvas.has_selection() else "Move Canvas"
                        cv2.line(frame, (cx, cy), (tx_thumb, ty_thumb), color, 3)
                        cv2.circle(frame, pinch_pos, 10, color, -1)
                        cv2.putText(
                            frame,
                            label,
                            (pinch_pos[0] + 15, pinch_pos[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                    elif gesture == "fist":
                        canvas.deselect_all()

                # =========================================================
                # RIGHT HAND CONTROLS (Drawing, Erasing, Scaling)
                # =========================================================
                else:
                    if gesture == "point":
                        drawing_active = True
                        canvas.add_point_to_active_stroke((cx, cy))
                        cv2.circle(frame, (cx, cy), 8, canvas.color, -1)

                    elif gesture == "erase":
                        canvas.finalize_active_stroke()
                        canvas.remove_points_from_strokes((cx, cy), radius=15)
                        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

                    elif gesture == "pinch":
                        right_pinching = True
                        canvas.finalize_active_stroke()
                        dist = gesture_recognizers["Right"].get_pinch_distance(lm)
                        if prev_right_pinch_dist is not None:
                            delta = dist - prev_right_pinch_dist
                            if canvas.has_selection():
                                scale_factor = max(0.8, min(1.2, 1.0 + delta * 3.0))
                                canvas.scale_selected_strokes(scale_factor)
                            else:
                                canvas.transform.scale += delta * 5.0
                                canvas.transform.scale = max(0.1, min(canvas.transform.scale, 10.0))
                        prev_right_pinch_dist = dist

                        pinch_pos = ((cx + tx_thumb) // 2, (cy + ty_thumb) // 2)
                        cv2.line(frame, (cx, cy), (tx_thumb, ty_thumb), (0, 255, 255), 3)
                        cv2.circle(frame, pinch_pos, 10, (0, 255, 255), -1)

        # Handle Left Hand Lasso Finalization
        if not left_pointing and len(left_lasso_points) > 0:
            if len(left_lasso_points) >= 3:
                count = canvas.select_strokes_in_polygon(left_lasso_points)
                print(f"Selection complete: {count} stroke(s) selected.")
            left_lasso_points = []

        # Draw live selection lasso path on frame
        if len(left_lasso_points) >= 2:
            for i in range(1, len(left_lasso_points)):
                cv2.line(frame, left_lasso_points[i - 1], left_lasso_points[i], (255, 255, 0), 2)
            cv2.putText(
                frame,
                "Selecting...",
                left_lasso_points[-1],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        if not left_pinching:
            prev_left_pinch_pos = None

        if not right_pinching:
            prev_right_pinch_dist = None

        if not drawing_active:
            canvas.finalize_active_stroke()

        # Draw bounding box around selected strokes
        canvas.draw_selection_overlay(frame)

        fps_now = time.time()
        elapsed = fps_now - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = fps_now

        # UI Header & Status Info
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

        sel_count = sum(1 for s in canvas.strokes if s.selected)
        transform_text = f"Scale: {canvas.transform.scale:.2f}x | Pan: ({int(canvas.transform.tx)}, {int(canvas.transform.ty)}) | Selected: {sel_count}"
        cv2.putText(
            frame,
            transform_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0) if sel_count > 0 else (255, 200, 0),
            2,
        )

        foreground = canvas.re_render_frame()
        mask = canvas.get_foreground_mask()

        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        combined = cv2.add(frame_bg, foreground)

        color_msg = "Left Point = Circle Select | Left Pinch = Move | Left Fist = Deselect | Right Pinch = Scale | 'c' = Clear | 'q' = Quit"
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
        elif key == ord("d"):
            canvas.deselect_all()
            print("Selection cleared.")
        elif key == ord("n"):
            color = canvas.next_color()
            print(f"Color changed to index {canvas.color_index}: {color}")

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
