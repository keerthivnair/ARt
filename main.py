import os
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import cv2
import numpy as np

from config import (
    COLOR_PALETTE,
    DEPTH_SENSITIVITY,
    DOLLY_SENSITIVITY,
    ORBIT_SENSITIVITY,
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    ROTATE_SENSITIVITY,
    SAVE_DIR,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from drawing.canvas import DrawingCanvas
from drawing.geometry import angle_difference
from handtracking.gestures import GestureRecognizer
from handtracking.tracker import HandTracker


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

    print("ARt 3D System started. Press 'q' to quit.")
    print("RIGHT: Point=draw on draw plane, 2 fingers=erase, Pinch=scale/dolly,")
    print("       3 fingers=push in/out in Z, Index+Pinky=rotate, Open palm=toggle snap")
    print("LEFT : Point=lasso select, Pinch=move/pan, 2 fingers=snap to hand,")
    print("       3 fingers=orbit view, Index+Pinky=rotate, Fist=deselect, Open palm=reset view")

    os.makedirs(SAVE_DIR, exist_ok=True)
    notification_text = ""
    notification_time = 0.0

    fps_time = time.time()
    fps = 0
    frame_count = 0

    prev_left_pinch_pos = None
    prev_right_pinch_dist = None
    left_lasso_points = []
    prev_left_orbit_pos = None
    prev_right_depth_y = None
    prev_left_roll = None
    prev_right_roll = None

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
        tracker.draw_landmarks(frame, results)

        active_gestures = {}
        left_pointing = False
        left_pinching = False
        right_pinching = False
        drawing_active = False

        for lm in landmarks:
            hand_side = lm.get("handedness", "Right")
            if hand_side not in gesture_recognizers:
                gesture_recognizers[hand_side] = GestureRecognizer(switch_threshold=2)
            recognizer = gesture_recognizers[hand_side]

            gesture = recognizer.analyze(lm)
            active_gestures[hand_side] = gesture

            index_tip = lm["INDEX_TIP"]
            thumb_tip = lm["THUMB_TIP"]
            cx = int(index_tip[0] * SCREEN_WIDTH)
            cy = int(index_tip[1] * SCREEN_HEIGHT)
            tx_thumb = int(thumb_tip[0] * SCREEN_WIDTH)
            ty_thumb = int(thumb_tip[1] * SCREEN_HEIGHT)

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
                        if canvas.has_selection():
                            canvas.move_selected_strokes(prev_left_pinch_pos, pinch_pos)
                            label, color = "Move Selection (draw plane)", (255, 255, 0)
                        else:
                            canvas.pan_view(prev_left_pinch_pos, pinch_pos)
                            label, color = "Pan View", (255, 0, 255)
                    else:
                        label, color = ("Move Selection (draw plane)" if canvas.has_selection() else "Pan View"), (
                            (255, 255, 0) if canvas.has_selection() else (255, 0, 255)
                        )
                    prev_left_pinch_pos = pinch_pos

                    cv2.line(frame, (cx, cy), (tx_thumb, ty_thumb), color, 3)
                    cv2.circle(frame, pinch_pos, 10, color, -1)
                    cv2.putText(
                        frame,
                        label,
                        (pinch_pos[0] + 15, pinch_pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                elif gesture == "two_fingers":
                    canvas.finalize_active_stroke()
                    if canvas.has_selection():
                        canvas.snap_selected_strokes_to(cx, cy)
                    hand_center = ((cx + cx) // 2, (cy + cy) // 2)
                    cv2.circle(frame, hand_center, 16, (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        "SNAP selection to hand",
                        (hand_center[0] + 20, hand_center[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                elif gesture == "three_fingers":
                    canvas.finalize_active_stroke()
                    if prev_left_orbit_pos is not None:
                        dx = cx - prev_left_orbit_pos[0]
                        dy = cy - prev_left_orbit_pos[1]
                        canvas.orbit_view(-dx * ORBIT_SENSITIVITY, -dy * ORBIT_SENSITIVITY)
                    prev_left_orbit_pos = (cx, cy)
                    cv2.circle(frame, (cx, cy), 20, (0, 200, 255), 2)
                    cv2.putText(
                        frame,
                        "ORBIT view",
                        (cx + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 255),
                        2,
                    )

                elif gesture == "rotate":
                    canvas.finalize_active_stroke()
                    roll = recognizer.get_hand_roll_angle(lm)
                    if prev_left_roll is not None:
                        canvas.rotate_selection_or_view(
                            angle_difference(roll, prev_left_roll) * ROTATE_SENSITIVITY
                        )
                    prev_left_roll = roll
                    target = "selection" if canvas.has_selection() else "view plane"
                    color = (255, 255, 0) if canvas.has_selection() else (0, 255, 255)
                    cv2.putText(
                        frame,
                        f"ROTATE {target}",
                        (cx + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                elif gesture == "fist":
                    canvas.finalize_active_stroke()
                    canvas.deselect_all()
                    notification_text = "Selection cleared"
                    notification_time = time.time()

                elif gesture == "open_palm":
                    canvas.finalize_active_stroke()
                    canvas.reset_view()
                    notification_text = "View reset"
                    notification_time = time.time()

            else:
                if gesture == "point":
                    drawing_active = True
                    canvas.add_point_to_active_stroke((cx, cy))
                    canvas.draw_draw_plane_marker(frame, (cx, cy), canvas.color)
                    cv2.circle(frame, (cx, cy), 8, canvas.color, -1)

                elif gesture == "two_fingers":
                    canvas.finalize_active_stroke()
                    canvas.remove_points_from_strokes((cx, cy), radius=15)
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

                elif gesture == "pinch":
                    right_pinching = True
                    canvas.finalize_active_stroke()
                    distance = recognizer.get_pinch_distance(lm)
                    if prev_right_pinch_dist is not None:
                        delta = distance - prev_right_pinch_dist
                        if canvas.has_selection():
                            scale_factor = max(0.8, min(1.2, 1.0 + delta * 3.0))
                            canvas.scale_selected_strokes(scale_factor)
                        else:
                            canvas.dolly_view(delta * 500.0 * DOLLY_SENSITIVITY)
                    prev_right_pinch_dist = distance

                    pinch_pos = ((cx + tx_thumb) // 2, (cy + ty_thumb) // 2)
                    cv2.line(frame, (cx, cy), (tx_thumb, ty_thumb), (0, 255, 255), 3)
                    cv2.circle(frame, pinch_pos, 10, (0, 255, 255), -1)
                    label = "Scale Selection" if canvas.has_selection() else "Dolly View (Z)"
                    cv2.putText(
                        frame,
                        label,
                        (pinch_pos[0] + 15, pinch_pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                elif gesture == "three_fingers":
                    canvas.finalize_active_stroke()
                    if prev_right_depth_y is not None:
                        dy = prev_right_depth_y - cy
                        if canvas.has_selection():
                            canvas.translate_selected_depth(dy * DEPTH_SENSITIVITY)
                            label, color = "Translate Z (selection)", (0, 200, 255)
                        else:
                            canvas.adjust_draw_distance(dy * DEPTH_SENSITIVITY)
                            label, color = "Draw plane distance", (0, 200, 255)
                    else:
                        label, color = "Translate Z (selection)", (0, 200, 255)
                    prev_right_depth_y = cy
                    cv2.putText(
                        frame,
                        label,
                        (cx + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                elif gesture == "rotate":
                    canvas.finalize_active_stroke()
                    roll = recognizer.get_hand_roll_angle(lm)
                    if prev_right_roll is not None:
                        canvas.rotate_selection_or_view(
                            angle_difference(roll, prev_right_roll) * ROTATE_SENSITIVITY
                        )
                    prev_right_roll = roll
                    target = "selection" if canvas.has_selection() else "view plane"
                    color = (255, 255, 0) if canvas.has_selection() else (0, 255, 255)
                    cv2.putText(
                        frame,
                        f"ROTATE {target}",
                        (cx + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                elif gesture == "open_palm":
                    canvas.finalize_active_stroke()
                    state = canvas.toggle_snap()
                    notification_text = f"Vertex snapping {'ON' if state else 'OFF'}"
                    notification_time = time.time()

        if not left_pointing and len(left_lasso_points) > 0:
            if len(left_lasso_points) >= 3:
                count = canvas.select_strokes_in_polygon(left_lasso_points)
                print(f"Selection complete: {count} stroke(s) selected.")
            left_lasso_points = []

        if len(left_lasso_points) >= 2:
            for index in range(1, len(left_lasso_points)):
                cv2.line(frame, left_lasso_points[index - 1], left_lasso_points[index], (255, 255, 0), 2)
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
        if "three_fingers" not in active_gestures.values():
            prev_left_orbit_pos = None
            prev_right_depth_y = None
        if "rotate" not in active_gestures.values():
            prev_left_roll = None
            prev_right_roll = None
        if not drawing_active:
            canvas.finalize_active_stroke()

        canvas.draw_selection_overlay(frame)

        now = time.time()
        elapsed = now - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = now

        camera = canvas.camera
        cv2.putText(
            frame,
            f"FPS: {fps:.0f} | Color: {canvas.color_index + 1}/{len(COLOR_PALETTE)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        status = " | ".join(f"{h}: {g}" for h, g in active_gestures.items()) if active_gestures else "No hands"
        cv2.putText(
            frame,
            f"Hands: {status}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if active_gestures else (0, 0, 255),
            2,
        )

        sel_count = len(canvas.scene.selected_strokes())
        snap_label = f"{canvas.snap_radius:.0f}" if canvas.snap_enabled else "OFF"
        transform_text = (
            f"yaw {np.degrees(camera.yaw):.0f} pitch {np.degrees(camera.pitch):.0f} "
            f"roll {np.degrees(camera.roll):.0f} | draw plane {camera.draw_distance:.0f} "
            f"| sel {sel_count} | snap {snap_label}"
        )
        cv2.putText(
            frame,
            transform_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0) if sel_count > 0 else (255, 200, 0),
            2,
        )

        foreground = canvas.re_render_frame()
        mask = canvas.get_foreground_mask()

        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        combined = cv2.add(frame_bg, foreground)

        help_msg = (
            "L: point=lasso pinch=move 2f=snap 3f=orbit idx+pinky=rotate | "
            "R: point=draw 2f=erase pinch=scale 3f=Z idx+pinky=rotate | "
            "keys: n c d s l m v [ ] - = r g q"
        )
        cv2.putText(
            combined,
            help_msg,
            (10, SCREEN_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )

        if time.time() - notification_time < 1.5:
            cv2.putText(
                combined,
                notification_text,
                (SCREEN_WIDTH // 2 - 160, SCREEN_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

        cv2.imshow("ARt - 3D Hand Drawing", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas.clear()
            print("Scene cleared.")
        elif key == ord("d"):
            canvas.deselect_all()
            print("Selection cleared.")
        elif key == ord("n"):
            color = canvas.next_color()
            print(f"Color changed to index {canvas.color_index}: {color}")
        elif key == ord("v"):
            state = canvas.toggle_snap()
            print(f"Vertex snapping {'ON' if state else 'OFF'}")
        elif key in (ord("["), ord("-")):
            print(f"Snap radius: {canvas.adjust_snap_radius(-10.0):.0f}")
        elif key in (ord("]"), ord("=")):
            print(f"Snap radius: {canvas.adjust_snap_radius(10.0):.0f}")
        elif key == ord("r"):
            canvas.reset_view()
            print("View reset.")
        elif key == ord("g"):
            canvas.show_draw_plane = not canvas.show_draw_plane
            print(f"Draw plane guide {'shown' if canvas.show_draw_plane else 'hidden'}")
        elif key == ord("s"):
            root = tk.Tk()
            root.withdraw()
            default_name = f"ARt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = filedialog.asksaveasfilename(
                initialdir=os.path.abspath(SAVE_DIR),
                initialfile=default_name,
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
            )
            root.destroy()
            if path:
                canvas.save_to_file(path)
                notification_text = f"Saved to {os.path.basename(path)}"
                notification_time = time.time()
        elif key == ord("l") or key == ord("m"):
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                initialdir=os.path.abspath(SAVE_DIR),
                defaultextension=".json",
                filetypes=[("JSON File", "*.json")],
            )
            root.destroy()
            if path:
                success = canvas.load_from_file(path, merge=(key == ord("m")))
                if success:
                    action = "Merged" if key == ord("m") else "Loaded"
                    notification_text = f"{action} from {os.path.basename(path)}"
                    notification_time = time.time()

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
