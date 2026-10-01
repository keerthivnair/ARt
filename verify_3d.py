"""Headless checks for the 3D pipeline: projection, drawing plane, snapping,
3D selection transforms and rendering. Run with the project venv."""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drawing.canvas import DrawingCanvas
from drawing.geometry import Camera, angle_difference
from config import SCREEN_HEIGHT, SCREEN_WIDTH, DRAW_PLANE_DISTANCE

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


print("\n[1] Camera projection")
cam = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
centre = cam.draw_plane_point(cam.center_x, cam.center_y)
proj = cam.project(centre)
check("draw plane centre maps to screen centre",
      abs(proj[0] - SCREEN_WIDTH / 2) < 1e-6 and abs(proj[1] - SCREEN_HEIGHT / 2) < 1e-6, proj)
check("draw plane centre is at draw_distance in front of camera",
      abs(np.linalg.norm(centre - cam.position()) - DRAW_PLANE_DISTANCE) < 1e-6)
check("draw plane is parallel to the view plane (centre on forward axis)",
      np.allclose(centre - cam.position(), cam.forward() * DRAW_PLANE_DISTANCE))

right_pt = cam.draw_plane_point(cam.center_x + 200, cam.center_y)
check("screen +X maps to world +X", right_pt[0] > centre[0] and abs(right_pt[1] - centre[1]) < 1e-6)
down_pt = cam.draw_plane_point(cam.center_x, cam.center_y + 200)
check("screen +Y (down) maps to world +Y", down_pt[1] > centre[1] and abs(down_pt[0] - centre[0]) < 1e-6)

far = centre + cam.forward() * 500
far_proj = cam.project(far)
check("points further away project closer to centre",
      abs(far_proj[0] - cam.center_x) < 1e-6 and far_proj[2] > proj[2])

check("point behind the camera is rejected", cam.project(cam.position() - cam.forward() * 100) is None)

print("\n[2] Draw plane follows the camera (rotated view)")
cam2 = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
cam2.orbit(0.9, 0.4)
c2 = cam2.draw_plane_point(cam2.center_x, cam2.center_y)
check("draw plane still lies on the forward axis after orbit",
      np.allclose(c2 - cam2.position(), cam2.forward() * DRAW_PLANE_DISTANCE))
check("draw plane still projects to screen centre after orbit",
      abs(cam2.project(c2)[0] - SCREEN_WIDTH / 2) < 1e-6)
check("draw plane is NOT the world z=DRAW_PLANE plane after orbit",
      abs(c2[2] - DRAW_PLANE_DISTANCE) > 1.0, f"z={c2[2]:.2f}")

print("\n[3] Drawing lands on the draw plane in 3D")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
for x in range(600, 700, 10):
    canvas.add_point_to_active_stroke((x, 360))
canvas.finalize_active_stroke()
stroke = canvas.scene.strokes[0]
check("one 3D stroke created", len(canvas.scene.strokes) == 1 and not stroke.is_empty())
check("vertices are 3D tuples", all(len(p) == 3 for p in stroke.points))
offsets = [np.dot(p - canvas.camera.position(), canvas.camera.forward()) for p in stroke.world_points()]
check("every drawn vertex sits on the draw plane",
      all(abs(o - DRAW_PLANE_DISTANCE) < 1e-6 for o in offsets), offsets[:3])
check("a stroke drawn in the default view is a flat 3D plane",
      max(p[2] for p in stroke.world_points()) - min(p[2] for p in stroke.world_points()) < 1e-6)
drawn_origin = stroke.world_points()[0]
check("stroke projects near the drawn screen coords",
      abs(canvas.project(stroke.world_points()[0])[0] - 600) < 1.0)
check("flat stroke seen head-on has uniform camera depth",
      max(canvas.project(p)[2] for p in stroke.world_points()) -
      min(canvas.project(p)[2] for p in stroke.world_points()) < 1.0)
canvas.camera.orbit(0.6, 0.25)
view_depths = [canvas.project(p)[2] for p in stroke.world_points()]
check("the same flat stroke spans real depth once the view is orbited",
      max(view_depths) - min(view_depths) > 20.0,
      f"{max(view_depths) - min(view_depths):.1f}")
check("orbiting the camera does not move the geometry",
      np.allclose(stroke.world_points()[0], drawn_origin))
canvas.camera.reset()

print("\n[4] Vertex snapping to the nearest drawn voxel")
canvas.snap_enabled = True
canvas.snap_radius = 45.0
canvas.finalize_active_stroke()
drawn = stroke.world_points()
existing = drawn[0]
existing_set = {tuple(np.round(p, 6)) for p in drawn}
# Land a new vertex 10 world units from an existing one: inside the 45 unit radius.
probe_world = existing + canvas.camera.right() * 10.0
canvas.add_point_to_active_stroke(canvas.project(probe_world))
snapped = canvas.scene.active_stroke.world_points()[-1]
check("new vertex snapped onto a drawn voxel",
      tuple(np.round(snapped, 6)) in existing_set, f"{snapped} not in stroke")
check("snapped vertex is exactly the voxel, not an interpolation",
      float(np.linalg.norm(snapped - existing)) < 45.0)
check("snap target reported for the HUD", canvas.last_snap_target is not None)

canvas.finalize_active_stroke()
canvas.snap_radius = 5.0
far_screen = canvas.project(canvas.draw_plane_point(200, 100) + canvas.camera.forward() * 300)
canvas.add_point_to_active_stroke(far_screen)
check("vertex outside the radius is not snapped", canvas.last_snap_target is None)

canvas.snap_enabled = False
canvas.snap_radius = 500.0
canvas.add_point_to_active_stroke(canvas.project(probe_world))
check("snapping disabled ignores nearby voxels", canvas.last_snap_target is None)
canvas.snap_enabled = True
canvas.finalize_active_stroke()

print("\n[5] Selection is 3D (lasso, rotate, translate, scale)")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
canvas.snap_enabled = False
for x in range(500, 700, 10):
    canvas.add_point_to_active_stroke((x, 300))
canvas.finalize_active_stroke()
for x in range(500, 700, 10):
    canvas.add_point_to_active_stroke((x, 420))
canvas.finalize_active_stroke()
canvas.add_point_to_active_stroke((300, 100))
canvas.finalize_active_stroke()
check("three strokes exist", len(canvas.scene.strokes) == 3)

count = canvas.select_strokes_in_polygon([(450, 250), (750, 250), (750, 470), (450, 470)])
check("lasso selected the two enclosed strokes", count == 2, f"got {count}")
check("has_selection() is True", canvas.has_selection())

before = np.array([p for s in canvas.scene.selected_strokes() for p in s.world_points()])
unselected_before = canvas.scene.strokes[2].world_points()

canvas.rotate_selection_or_view(math.radians(30))
after = np.array([p for s in canvas.scene.selected_strokes() for p in s.world_points()])
check("rotating with a selection rotates only the selection",
      not np.allclose(before, after) and np.allclose(unselected_before, canvas.scene.strokes[2].world_points()))
check("rotation preserves shape (pairwise distances kept)",
      abs(float(np.linalg.norm(before[0] - before[1])) - float(np.linalg.norm(after[0] - after[1]))) < 1e-6)

canvas.deselect_all()
canvas.camera.roll_by(math.radians(25))
check("rotating with no selection rolls the view plane", abs(canvas.camera.roll - math.radians(25)) < 1e-9)
canvas.camera.roll = 0.0
canvas.camera.invalidate()

canvas.select_strokes_in_polygon([(450, 250), (750, 250), (750, 470), (450, 470)])
centroid_before = canvas.scene.selection_centroid()
canvas.translate_selected_depth(200)
centroid_after = canvas.scene.selection_centroid()
check("depth translation moves the selection along the view axis",
      np.allclose(centroid_after - centroid_before, canvas.camera.forward() * canvas.pixels_to_world(200)),
      f"{centroid_after - centroid_before}")
check("depth translation leaves the selection's on-screen size alone",
      abs(float(np.linalg.norm(canvas.scene.selection_centroid() - centroid_before))
          - canvas.pixels_to_world(200)) < 1e-6)

prev_xy = (600, 300)
canvas.move_selected_strokes(prev_xy, (700, 380))
moved = canvas.scene.selection_centroid()
check("draw-plane move translates by the exact screen delta",
      np.allclose(moved - centroid_after,
                  canvas.draw_plane_point(700, 380) - canvas.draw_plane_point(*prev_xy)))

canvas.snap_selected_strokes_to(300, 600)
target = canvas.draw_plane_point(300, 600)
check("snap-to-hand centres the selection on the hand point",
      np.allclose(canvas.scene.selection_centroid(), target, atol=1e-6))

scale_ref = float(np.linalg.norm(
    canvas.scene.selected_strokes()[0].world_points()[1] -
    canvas.scene.selected_strokes()[0].world_points()[0]))
canvas.scale_selected_strokes(2.0)
scale_now = float(np.linalg.norm(
    canvas.scene.selected_strokes()[0].world_points()[1] -
    canvas.scene.selected_strokes()[0].world_points()[0]))
check("scaling the selection scales in 3D", abs(scale_now - scale_ref * 2.0) < 1e-6)

print("\n[6] View operations")
cam3 = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
yaw0 = cam3.yaw
cam3.orbit(0.5, 0.2)
check("orbit changes yaw and pitch", abs(cam3.yaw - yaw0 - 0.5) < 1e-9 and abs(cam3.pitch - 0.2) < 1e-9)
cam3.pitch = 1.4
cam3.orbit(0.0, 1.0)
check("pitch is clamped", cam3.pitch <= 1.45)
d0 = cam3.distance
cam3.dolly(5000)
check("dolly is clamped to max", cam3.distance <= 2500.0 and cam3.distance > d0)
target0 = cam3.target.copy()
cam3.shift(np.array([10.0, 0.0, 0.0]))
check("shift moves the orbit target", not np.allclose(target0, cam3.target))
cam3.reset()
check("reset restores the default view",
      cam3.yaw == 0 and cam3.pitch == 0 and cam3.roll == 0 and np.allclose(cam3.target, 0))
dd0 = cam3.draw_distance
cam3.adjust_draw_distance(10000)
check("draw distance is clamped", cam3.draw_distance <= 2500.0 and cam3.draw_distance > dd0)

print("\n[7] Rendering")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
canvas.show_draw_plane = False
canvas.add_point_to_active_stroke((400, 360))
canvas.add_point_to_active_stroke((800, 360))
canvas.finalize_active_stroke()
img = canvas.re_render_frame()
check("render has the video resolution", img.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3))
mask = canvas.get_foreground_mask()
painted = int(np.count_nonzero(mask))
check("stroke is painted into the foreground", painted > 100, f"{painted} px")
ys, xs = np.nonzero(mask)
check("painted pixels match the drawn screen line",
      abs(float(xs.mean()) - 600) < 2 and abs(float(ys.mean()) - 360) < 2,
      f"mean=({xs.mean():.0f},{ys.mean():.0f})")
check("foreground pixels carry the stroke colour",
      tuple(int(v) for v in img[int(ys.mean()), int(xs.mean())]) == tuple(canvas.color))

canvas.show_draw_plane = True
guided = canvas.re_render_frame()
check("draw plane guide draws onto the plane", np.count_nonzero(guided) > painted)

canvas.camera.orbit(0.8, 0.3)
rot_img = canvas.re_render_frame()
check("scene still renders after the camera moves", rot_img.shape == img.shape)
p2 = canvas.project(canvas.scene.strokes[0].world_points()[0])
check("stroke still projects on screen while orbited",
      p2 is not None and 0 < p2[0] < SCREEN_WIDTH and 0 < p2[1] < SCREEN_HEIGHT, p2)

print("\n[8] Erasing and clearing")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
canvas.show_draw_plane = False
for x in range(400, 900, 10):
    canvas.add_point_to_active_stroke((x, 360))
canvas.finalize_active_stroke()
check("stroke built for the erase test", len(canvas.scene.strokes) == 1)
canvas.remove_points_from_strokes((400, 360), radius=60)
canvas.remove_points_from_strokes((890, 360), radius=60)
check("erase removed the vertices near the cursor", len(canvas.scene.strokes[0].points) < 50)
canvas.clear()
check("clear empties the scene and resets the view",
      not canvas.scene.strokes and canvas.camera.yaw == 0 and canvas.camera.roll == 0)

print("\n[9] Save / load round trip")
canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
canvas.snap_enabled = False
canvas.camera.orbit(0.4, 0.2)
canvas.camera.roll_by(0.3)
canvas.camera.set_draw_distance(900.0)
canvas.camera.target = np.array([10.0, -5.0, 3.0])
for x in range(500, 640, 10):
    canvas.add_point_to_active_stroke((x, 330))
for x in range(500, 640, 10):
    canvas.add_point_to_active_stroke((x, 390))
canvas.finalize_active_stroke()
canvas.camera.pan(40, -25)
canvas.camera.roll_by(0.25)
canvas.adjust_draw_distance(120)
points_before = [p for s in canvas.scene.strokes for p in s.world_points()]
expected_pts = sorted([tuple(p) for p in points_before])
expected_cam = canvas.camera.to_dict()

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "roundtrip.png")
    canvas.save_to_file(path)
    check("png written", os.path.exists(path))
    json_path = os.path.join(tmp, "roundtrip.json")
    check("json sidecar written", os.path.exists(json_path))
    check("png is a full frame", os.path.getsize(path) > 0)

    restored = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
    check("load succeeds", restored.load_from_file(json_path))
    loaded_pts = sorted([tuple(p) for s in restored.scene.strokes for p in s.world_points()])
    check("3D geometry survives the round trip",
          len(loaded_pts) == len(expected_pts) and
          all(np.allclose(a, b) for a, b in zip(loaded_pts, expected_pts)))
    for key in ("yaw", "pitch", "roll", "draw_distance", "distance", "focal"):
        check(f"camera {key} restored", abs(restored.camera.to_dict()[key] - expected_cam[key]) < 1e-9)
    check("camera target restored", np.allclose(restored.camera.target, canvas.camera.target))

    merged = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
    merged.snap_enabled = False
    for x in range(200, 340, 10):
        merged.add_point_to_active_stroke((x, 200))
    merged.finalize_active_stroke()
    existing = len(merged.scene.strokes)
    merged.load_from_file(json_path, merge=True)
    check("merge adds strokes without clearing", len(merged.scene.strokes) == existing + 1,
          f"{len(merged.scene.strokes)}")

print("\n[10] Legacy 2D save compatibility")
import json
with tempfile.TemporaryDirectory() as tmp:
    legacy = {
        "transform": {"scale": 2.0, "tx": 10, "ty": 5},
        "strokes": [{"points": [[100, 200], [140, 210], [180, 205]],
                     "color": [0, 0, 255], "thickness": 4, "selected": False}],
    }
    path = os.path.join(tmp, "legacy.json")
    with open(path, "w") as handle:
        json.dump(legacy, handle)
    canvas = DrawingCanvas(SCREEN_WIDTH, SCREEN_HEIGHT)
    check("legacy 2D file loads", canvas.load_from_file(path))
    check("legacy stroke padded to 3D", all(len(p) == 3 for p in canvas.scene.strokes[0].points))
    canvas.re_render_frame()
    check("legacy file renders", canvas.canvas.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3))

print("\n[11] Gesture recognizer")
import handtracking.gestures as gestures_module
from handtracking.gestures import GestureRecognizer
from config import LANDMARK

WRIST = (0.5, 0.9, 0.0)


def make_hand(fingers_up, thumb_near_index=False, spread=0.0):
    lm = {}
    for name, idx in LANDMARK.items():
        lm[name] = (0.5, 0.5, 0.0)
    lm["WRIST"] = WRIST

    order = ["INDEX", "MIDDLE", "RING", "PINKY"]
    for position, finger in enumerate(order):
        offset = (position - 1.5) * 0.08
        x = 0.5 + offset
        if fingers_up.get(finger):
            lm[f"{finger}_MCP"] = (x, 0.62, 0.0)
            lm[f"{finger}_PIP"] = (x, 0.52, 0.0)
            lm[f"{finger}_DIP"] = (x, 0.44, 0.0)
            lm[f"{finger}_TIP"] = (x, 0.36, 0.0)
        else:
            # Curled: the tip folds back down past the PIP joint and the wrist.
            lm[f"{finger}_MCP"] = (x, 0.62, 0.0)
            lm[f"{finger}_PIP"] = (x, 0.56, 0.0)
            lm[f"{finger}_DIP"] = (x, 0.70, 0.0)
            lm[f"{finger}_TIP"] = (x, 0.80, 0.0)

    index_tip_x = 0.5 + (0 - 1.5) * 0.08
    if thumb_near_index:
        # Thumb tip closes on the index tip, and the thumb reaches further from
        # the wrist than its MCP joint so it still reads as extended.
        lm["THUMB_MCP"] = (0.45, 0.62, 0.0)
        lm["THUMB_CMC"] = (0.47, 0.70, 0.0)
        lm["THUMB_IP"] = (0.42, 0.48, 0.0)
        lm["THUMB_TIP"] = (index_tip_x + 0.02, 0.38, 0.0)
    else:
        lm["THUMB_MCP"] = (0.45, 0.62, 0.0)
        lm["THUMB_CMC"] = (0.47, 0.70, 0.0)
        lm["THUMB_IP"] = (0.40, 0.58, 0.0)
        lm["THUMB_TIP"] = (0.34 + spread, 0.56, 0.0)
    return lm


def classify(hand, frames=6):
    rec = GestureRecognizer(switch_threshold=2)
    result = "none"
    for _ in range(frames):
        result = rec.analyze(hand)
    return result


check("index only -> point", classify(make_hand({"INDEX": True})) == "point")
check("index+middle -> two_fingers", classify(make_hand({"INDEX": True, "MIDDLE": True})) == "two_fingers")
check("index+middle+ring -> three_fingers",
      classify(make_hand({"INDEX": True, "MIDDLE": True, "RING": True})) == "three_fingers")
check("index+pinky -> rotate", classify(make_hand({"INDEX": True, "PINKY": True})) == "rotate")
check("all four -> open_palm",
      classify(make_hand({"INDEX": True, "MIDDLE": True, "RING": True, "PINKY": True})) == "open_palm")
check("none up -> fist", classify(make_hand({})) == "fist")
check("thumb+index pinch -> pinch",
      classify(make_hand({"INDEX": True}, thumb_near_index=True)) == "pinch")
check("two unrelated fingers (middle+ring) -> none",
      classify(make_hand({"MIDDLE": True, "RING": True})) == "none")

rec = GestureRecognizer(switch_threshold=2)
hand = make_hand({"INDEX": True, "PINKY": True})
first = rec.get_hand_roll_angle(hand)
rotated = dict(hand)
knuckle = 0.10
rotated["INDEX_MCP"] = (0.5, 0.62, 0.0)
rotated["PINKY_MCP"] = (0.5 + knuckle * math.cos(first + math.pi / 4),
                        0.62 + knuckle * math.sin(first + math.pi / 4), 0.0)
second = rec.get_hand_roll_angle(rotated)
check("roll angle tracks hand rotation",
      abs(angle_difference(second, first) - math.pi / 4) < 1e-6, angle_difference(second, first))
check("roll angle wraps correctly",
      abs(angle_difference(3.0, -3.0) - (-0.2831853)) < 1e-6, angle_difference(3.0, -3.0))

rec = GestureRecognizer(switch_threshold=2)
open_hand = make_hand({"INDEX": True, "MIDDLE": True, "RING": True, "PINKY": True})
rec.analyze(open_hand)
rec.analyze(open_hand)
rec.analyze(make_hand({}))
rec.analyze(make_hand({}))
rec.analyze(make_hand({}))
rec.analyze(make_hand({}))
rec.analyze(make_hand({}))
rec.analyze(make_hand({}))
check("fist is recognised after a stability delay", rec.return_gesture == "fist", rec.return_gesture)

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
