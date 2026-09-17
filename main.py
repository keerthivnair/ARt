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

    tracker = HandTracker(max_hands=1)
    gesture_recognizer = GestureRecognizer()
    canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)

    process_frame = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8)

    print("Hand Drawing System started. Press 'q' to quit.")
    print("Point index finger to draw.")
    print("Pinch thumb and index finger to scale.")
    print("Press 'c' to clear canvas. Press 'n' for next color.")

    fps_time = time.time()
    fps = 0
    frame_count = 0
    
    prev_pinch_dist = None

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

        if landmarks:
            lm = landmarks[0]
            gesture = gesture_recognizer.analyze(lm)
            
            index_tip = lm["INDEX_TIP"]
            cx = int(index_tip[0] * SCREEN_WIDTH)
            cy = int(index_tip[1] * SCREEN_HEIGHT)

            if gesture == "point":
                canvas.add_point_to_active_stroke((cx, cy))
                prev_pinch_dist = None
                cv2.circle(frame, (cx, cy), 8, DRAWING_COLOR, -1)
            elif gesture == 'erase':
                # FIX: without this, a stroke you just drew stays in
                # canvas.active_stroke (not canvas.strokes) until some other
                # gesture finalizes it, so remove_points_from_strokes()
                # (which only loops over canvas.strokes) can't touch it yet.
                canvas.finalize_active_stroke()
                canvas.remove_points_from_strokes((cx, cy), radius=15)
                prev_pinch_dist = None
                cv2.circle(frame, (cx, cy), 15, DRAWING_COLOR, -1)

            elif gesture == "pinch":
                canvas.finalize_active_stroke()
                dist = gesture_recognizer.get_pinch_distance(lm)
                if prev_pinch_dist is not None:
                    delta = dist - prev_pinch_dist
                    canvas.transform.scale += delta * 5.0
                    canvas.transform.scale = max(0.1, min(canvas.transform.scale, 10.0))
                prev_pinch_dist = dist
            else:
                canvas.finalize_active_stroke()
                prev_pinch_dist = None

        else:
            canvas.finalize_active_stroke()
            prev_pinch_dist = None

        fps_now = time.time()
        elapsed = fps_now - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = fps_now

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
        gesture_text = f"Gesture: {gesture}" if landmarks else "Gesture: none"
        cv2.putText(
            frame,
            gesture_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if landmarks else (0, 0, 255),
            2,
        )

        scale_text = f"Scale: {canvas.transform.scale:.2f}x"
        cv2.putText(
            frame,
            scale_text,
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

        color_msg = "Press 'n'=next color | 'c'=clear | 'q'=quit"
        cv2.putText(
            combined,
            color_msg,
            (10, SCREEN_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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