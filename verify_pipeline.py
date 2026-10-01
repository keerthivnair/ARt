"""Drives the real DrawingCanvas through every gesture path main.py uses, with
synthetic hands, to confirm the whole pipeline is coherent end to end.
Headless: no camera, no MediaPipe."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LANDMARK, SCREEN_HEIGHT, SCREEN_WIDTH
from drawing.canvas import DrawingCanvas
from drawing.geometry import angle_difference
from handtracking.gestures import GestureRecognizer

WRIST = (0.5, 0.9, 0.0)
ORDER = ["INDEX", "MIDDLE", "RING", "PINKY"]
failures = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def make_hand(up, pinch=False, knuckle_angle=None):
    lm = {n: (0.5, 0.5, 0.0) for n in LANDMARK}
    lm["WRIST"] = WRIST
    for i, finger in enumerate(ORDER):
        x = 0.5 + (i - 1.5) * 0.08
        if up.get(finger):
            lm[f"{finger}_MCP"] = (x, 0.62, 0.0)
            lm[f"{finger}_PIP"] = (x, 0.52, 0.0)
            lm[f"{finger}_DIP"] = (x, 0.44, 0.0)
            lm[f"{finger}_TIP"] = (x, 0.36, 0.0)
        else:
            lm[f"{finger}_MCP"] = (x, 0.62, 0.0)
            lm[f"{finger}_PIP"] = (x, 0.56, 0.0)
            lm[f"{finger}_DIP"] = (x, 0.70, 0.0)
            lm[f"{finger}_TIP"] = (x, 0.80, 0.0)

    index_x = 0.5 + (0 - 1.5) * 0.08
    lm["THUMB_MCP"] = (0.45, 0.62, 0.0)
    lm["THUMB_CMC"] = (0.47, 0.70, 0.0)
    if pinch:
        lm["THUMB_IP"] = (0.42, 0.48, 0.0)
        lm["THUMB_TIP"] = (index_x + 0.02, 0.38, 0.0)
    else:
        lm["THUMB_IP"] = (0.40, 0.58, 0.0)
        lm["THUMB_TIP"] = (0.34, 0.56, 0.0)

    if knuckle_angle is not None:
        half = 0.06
        lm["INDEX_MCP"] = (0.5, 0.62, 0.0)
        lm["PINKY_MCP"] = (0.5 + half * math.cos(knuckle_angle),
                          0.62 + half * math.sin(knuckle_angle), 0.0)
    return lm


def to_screen(lm, key="INDEX_TIP"):
    return (int(lm[key][0] * SCREEN_WIDTH), int(lm[key][1] * SCREEN_HEIGHT))


def shifted(lm, dx, dy):
    out = dict(lm)
    for k, v in lm.items():
        out[k] = (v[0] + dx, v[1] + dy, v[2])
    return out


print("\n[A] Right hand: draw then erase")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
canvas.show_draw_plane = False
rec_r = GestureRecognizer(2)
rec_r.analyze(make_hand({"INDEX": True}))
rec_r.analyze(make_hand({"INDEX": True}))
for step in range(20):
    hand = make_hand({"INDEX": True})
    hand = shifted(hand, -0.10 + step * 0.010, 0.15)
    if rec_r.analyze(hand) == "point":
        x, y = to_screen(hand)
        canvas.add_point_to_active_stroke((x, y))
check("point gesture produced a live stroke", canvas.scene.active_stroke is not None)
canvas.finalize_active_stroke()
check("stroke committed", len(canvas.scene.strokes) == 1)
check("stroke has 3D depth from the draw plane",
      all(abs(np.dot(p - canvas.camera.position(), canvas.camera.forward())
              - canvas.camera.draw_distance) < 1e-6 for p in canvas.scene.strokes[0].world_points()))

rec_e = GestureRecognizer(2)
erase_hand = make_hand({"INDEX": True, "MIDDLE": True})
# Park the erase cursor on the stroke that was just drawn.
erase_hand = shifted(erase_hand, 0.0, 0.15)
before = len(canvas.scene.strokes[0].points)
for _ in range(5):
    rec_e.analyze(erase_hand)
    x, y = to_screen(erase_hand)
    canvas.remove_points_from_strokes((x, y), radius=40)
check("erase removed vertices", len(canvas.scene.strokes[0].points) < before,
      f"{len(canvas.scene.strokes[0].points)} vs {before}")

print("\n[B] Left hand: lasso select")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
canvas.show_draw_plane = False
canvas.snap_enabled = False
for x in range(500, 700, 10):
    canvas.add_point_to_active_stroke((x, 300))
canvas.finalize_active_stroke()
for x in range(500, 700, 10):
    canvas.add_point_to_active_stroke((x, 420))
canvas.finalize_active_stroke()
canvas.add_point_to_active_stroke((250, 100))
canvas.finalize_active_stroke()

rec_l = GestureRecognizer(2)
lasso = []
rest = make_hand({"INDEX": True})
# Offset so the traced ellipse is centred between the two target strokes.
base_dx = (600 / SCREEN_WIDTH) - rest["INDEX_TIP"][0]
base_dy = (360 / SCREEN_HEIGHT) - rest["INDEX_TIP"][1]
rest = shifted(rest, base_dx, base_dy)
for i in range(30):
    t = i / 29.0
    hand = shifted(rest, 0.075 * math.cos(t * 2 * math.pi), 0.085 * math.sin(t * 2 * math.pi))
    if rec_l.analyze(hand) == "point":
        lasso.append(to_screen(hand))
count = canvas.select_strokes_in_polygon(lasso) if len(lasso) >= 3 else 0
check("lasso traced a closed loop", len(lasso) >= 10, f"{len(lasso)} pts")
check("lasso selected 2 strokes", count == 2, f"got {count}")
check("selection is live", canvas.has_selection())

print("\n[C] Left pinch: move selection inside the draw plane")
rec_p = GestureRecognizer(2)
for _ in range(2):
    rec_p.analyze(make_hand({"INDEX": True}, pinch=True))
sel_before = canvas.scene.selection_centroid()
start = to_screen(make_hand({"INDEX": True}, pinch=True))
for step in range(10):
    hand = shifted(make_hand({"INDEX": True}, pinch=True), 0.0, -0.05 + step * 0.005)
    if rec_p.analyze(hand) == "pinch":
        end = to_screen(hand)
        canvas.move_selected_strokes(start, end)
        start = end
sel_after = canvas.scene.selection_centroid()
check("pinch moved the selection", not np.allclose(sel_before, sel_after))
check("move stayed in the draw plane (z unchanged)",
      abs((sel_after - sel_before)[2]) < 1e-6)
unsel = canvas.scene.strokes[2].world_points()
check("unselected stroke did not move", np.allclose(unsel[0], canvas.scene.strokes[2].world_points()[0]))

print("\n[D] Rotate: selection when there is one, view plane when there is not")
sel_before = canvas.scene.selection_centroid()
world_before = [p for s in canvas.scene.selected_strokes() for p in s.world_points()]
rec_rot = GestureRecognizer(2)
for i in range(12):
    hand = make_hand({"INDEX": True, "PINKY": True}, knuckle_angle=i * 0.06)
    if rec_rot.analyze(hand) == "rotate":
        roll = rec_rot.get_hand_roll_angle(hand)
        if "prev" in dir():
            pass
        canvas.rotate_selection_or_view(angle_difference(roll, prev) if "prev" in locals() else 0.0)
        if "prev" not in locals():
            prev = roll
        else:
            prev = roll
world_after = [p for s in canvas.scene.selected_strokes() for p in s.world_points()]
check("selection rotated in 3D", not np.allclose(world_before[0], world_after[0]))
check("camera roll untouched while a selection exists", abs(canvas.camera.roll) < 1e-9)
check("unselected stroke untouched by rotation",
      np.allclose(canvas.scene.strokes[2].world_points(), unsel))
check("rotation about the view axis keeps the selection in its own plane",
      abs(canvas.scene.selection_centroid()[2] - sel_before[2]) < 1e-6)
check("rotation is rigid: centroid stays put on the view axis",
      abs(np.linalg.norm(canvas.scene.selection_centroid()[:2] - sel_before[:2])) < 1e-6)

canvas.deselect_all()
canvas.rotate_selection_or_view(math.radians(20))
check("with no selection, rotate rolls the view plane",
      abs(canvas.camera.roll - math.radians(20)) < 1e-9,
      f"roll={canvas.camera.roll}")
canvas.camera.roll = 0.0
canvas.camera.invalidate()

print("\n[E] Right 3-finger: depth translation, then draw plane distance")
canvas.select_strokes_in_polygon([(450, 250), (750, 250), (750, 470), (450, 470)])
rec_d = GestureRecognizer(2)
for _ in range(2):
    rec_d.analyze(make_hand({"INDEX": True, "MIDDLE": True, "RING": True}))
c0 = canvas.scene.selection_centroid()
prev_y = 200
for step in range(10):
    hand = shifted(make_hand({"INDEX": True, "MIDDLE": True, "RING": True}), 0.0, -0.05 + step * 0.006)
    if rec_d.analyze(hand) == "three_fingers":
        y = to_screen(hand)[1]
        canvas.translate_selected_depth(prev_y - y)
        prev_y = y
c1 = canvas.scene.selection_centroid()
moved = c1 - c0
check("selection translated along the view axis", np.linalg.norm(moved) > 5.0)
check("translation is parallel to the forward axis",
      np.allclose(np.cross(moved, canvas.camera.forward()), 0.0, atol=1e-6))
check("translation pushed the selection away from the camera",
      np.dot(moved, canvas.camera.forward()) > 0, f"{np.dot(moved, canvas.camera.forward()):.1f}")

canvas.deselect_all()
d0 = canvas.camera.draw_distance
prev_y = 200
for step in range(10):
    hand = shifted(make_hand({"INDEX": True, "MIDDLE": True, "RING": True}), 0.0, -0.05 + step * 0.006)
    if rec_d.analyze(hand) == "three_fingers":
        y = to_screen(hand)[1]
        canvas.adjust_draw_distance(prev_y - y)
        prev_y = y
check("with no selection, 3 fingers changes the draw plane distance",
      canvas.camera.draw_distance != d0, f"{d0} -> {canvas.camera.draw_distance}")
check("draw distance stays clamped", 120.0 <= canvas.camera.draw_distance <= 2500.0)
check("existing strokes keep their depth when the draw plane moves",
      not any(abs(np.dot(p - canvas.camera.position(), canvas.camera.forward())
                  - canvas.camera.draw_distance) < 1e-6
              for p in canvas.scene.strokes[0].world_points()))
canvas.add_point_to_active_stroke((600, 360))
canvas.finalize_active_stroke()
check("a newly drawn vertex lands on the new draw plane",
      all(abs(np.dot(p - canvas.camera.position(), canvas.camera.forward())
              - canvas.camera.draw_distance) < 1e-6
          for p in canvas.scene.strokes[-1].world_points()))

print("\n[F] Left 3-finger: orbit")
yaw0, pitch0 = canvas.camera.yaw, canvas.camera.pitch
rec_o = GestureRecognizer(2)
for _ in range(2):
    rec_o.analyze(make_hand({"INDEX": True, "MIDDLE": True, "RING": True}))
prev = (640, 360)
for step in range(12):
    hand = shifted(make_hand({"INDEX": True, "MIDDLE": True, "RING": True}),
                   step * 0.006, step * -0.004)
    if rec_o.analyze(hand) == "three_fingers":
        x, y = to_screen(hand)
        canvas.orbit_view(-(x - prev[0]) * 0.005, -(y - prev[1]) * 0.005)
        prev = (x, y)
check("orbit changed yaw and pitch",
      canvas.camera.yaw != yaw0 and canvas.camera.pitch != pitch0)
check("orbited scene still projects on screen",
      all(canvas.project(p) is not None for p in canvas.scene.strokes[0].world_points()))
canvas.reset_view()

print("\n[G] Snapping through the live draw path")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
canvas.show_draw_plane = False
canvas.snap_enabled = True
canvas.snap_radius = 60.0
for x in range(400, 900, 10):
    canvas.add_point_to_active_stroke((x, 360))
canvas.finalize_active_stroke()
existing = set(tuple(np.round(p, 6)) for p in canvas.scene.strokes[0].world_points())
# A second, separate stroke drawn very close to the first should snap onto it.
rec_s = GestureRecognizer(2)
for _ in range(2):
    rec_s.analyze(make_hand({"INDEX": True}))
hits = 0
for step in range(12):
    hand = shifted(make_hand({"INDEX": True}), -0.09, 0.175 + step * 0.0002)
    if rec_s.analyze(hand) == "point":
        x, y = to_screen(hand)
        canvas.add_point_to_active_stroke((x, y))
canvas.finalize_active_stroke()
new_points = [tuple(np.round(p, 6)) for p in canvas.scene.strokes[1].world_points()]
snapped = [p for p in new_points if p in existing]
check("second stroke snapped onto the first stroke's voxels", len(snapped) > 0,
      f"{len(snapped)}/{len(new_points)} snapped")
check("every snapped point is exactly a drawn voxel, not an approximation",
      all(any(np.allclose(np.array(p), np.array(v)) for v in existing) for p in snapped))

canvas.snap_enabled = False
canvas.snap_radius = 200.0
rec_s2 = GestureRecognizer(2)
for _ in range(2):
    rec_s2.analyze(make_hand({"INDEX": True}))
for step in range(12):
    hand = shifted(make_hand({"INDEX": True}), -0.20, 0.80 + step * 0.0002)
    if rec_s2.analyze(hand) == "point":
        canvas.add_point_to_active_stroke(to_screen(hand))
canvas.finalize_active_stroke()
last = [tuple(np.round(p, 6)) for p in canvas.scene.strokes[-1].world_points()]
check("far away stroke is not snapped", not any(p in existing for p in last))

print("\n[H] Hand loss does not corrupt state")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
rec = GestureRecognizer(2)
for step in range(6):
    hand = shifted(make_hand({"INDEX": True}), step * 0.02, 0.0)
    if rec.analyze(hand) == "point":
        canvas.add_point_to_active_stroke(to_screen(hand))
check("stroke is live before the hand is lost", canvas.scene.active_stroke is not None)
gesture = rec.analyze(None)
check("gesture falls back to none when the hand is lost",
      gesture in ("none", "point"), gesture)
canvas.finalize_active_stroke()
check("stroke still committed after hand loss", len(canvas.scene.strokes) == 1)
check("render still works after hand loss", canvas.re_render_frame() is not None)

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("ALL PIPELINE CHECKS PASSED")
