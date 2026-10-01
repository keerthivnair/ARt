import json
import os

import cv2
import numpy as np

from config import (
    COLOR_PALETTE,
    DRAWING_THICKNESS,
    DRAW_PLANE_GUIDE,
    MAX_THICKNESS_SCALE,
    MIN_PROJECTION_DEPTH,
    MIN_SHADE,
    MIN_THICKNESS_SCALE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SNAP_ENABLED,
    SNAP_RADIUS,
)
from drawing.geometry import Camera, clamp
from drawing.scene import Scene


class DrawingCanvas:
    """Renders 3D strokes as a perspective projection onto the 2D view plane."""

    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = int(width)
        self.height = int(height)

        self.camera = Camera(self.width, self.height)
        self.scene = Scene()

        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.color_index = 0
        self.color = COLOR_PALETTE[self.color_index]
        self.thickness = DRAWING_THICKNESS

        self.snap_enabled = SNAP_ENABLED
        self.snap_radius = SNAP_RADIUS
        self.snap_grid = None
        self.last_snap_target = None

        self.show_draw_plane = DRAW_PLANE_GUIDE

    # -- projection helpers ----------------------------------------------
    def project(self, point):
        return self.camera.project(point, MIN_PROJECTION_DEPTH)

    def draw_plane_point(self, screen_x, screen_y):
        return self.camera.draw_plane_point(screen_x, screen_y)

    def pixels_to_world(self, pixels, distance=None):
        return self.camera.pixels_to_world(pixels, distance)

    # -- drawing ----------------------------------------------------------
    def add_point_to_active_stroke(self, screen_point):
        world_point = self.draw_plane_point(screen_point[0], screen_point[1])
        radius = self.snap_radius if self.snap_enabled else 0.0
        snapped, hit = self.scene.snap_point(world_point, radius)
        self.last_snap_target = hit
        self.scene.add_point_to_active_stroke(snapped, self.color, self.thickness)

    def finalize_active_stroke(self):
        self.scene.finalize_active_stroke()

    def remove_points_from_strokes(self, screen_point, radius=15):
        world_point = self.draw_plane_point(screen_point[0], screen_point[1])
        world_radius = self.pixels_to_world(radius)
        self.scene.erase_near(world_point, world_radius)
        self.re_render_frame()

    # -- selection --------------------------------------------------------
    def has_selection(self):
        return self.scene.has_selection()

    def deselect_all(self):
        self.scene.deselect_all()

    def select_strokes_in_polygon(self, polygon_screen_points):
        return self.scene.select_in_polygon(polygon_screen_points, self.camera)

    def snap_selected_strokes_to(self, screen_x, screen_y):
        self.scene.move_selected_to(self.draw_plane_point(screen_x, screen_y))

    # -- transforms -------------------------------------------------------
    def move_selected_strokes(self, previous_xy, current_xy):
        """Translate the selection inside the draw plane by a screen-space drag."""
        if not self.has_selection():
            return
        start = self.draw_plane_point(previous_xy[0], previous_xy[1])
        end = self.draw_plane_point(current_xy[0], current_xy[1])
        self.scene.translate_selected(end - start)

    def translate_selected_depth(self, pixel_delta):
        """Translate the selection along the view axis.

        A positive ``pixel_delta`` means the hand moved up, which pushes the
        selection away from the camera, along the forward axis.
        """
        if not self.has_selection():
            return
        amount = self.pixels_to_world(pixel_delta)
        self.scene.translate_selected_depth(amount, self.camera.forward())

    def rotate_selection_or_view(self, angle):
        """Rotate the selection if there is one, otherwise the whole view plane."""
        if self.has_selection():
            self.scene.rotate_selected(self.camera.forward(), angle)
        else:
            self.camera.roll_by(angle)

    def scale_selected_strokes(self, scale_factor):
        self.scene.scale_selected(scale_factor)

    def pan_view(self, previous_xy, current_xy):
        start = self.draw_plane_point(previous_xy[0], previous_xy[1])
        end = self.draw_plane_point(current_xy[0], current_xy[1])
        self.camera.shift(-(end - start))

    def orbit_view(self, delta_yaw, delta_pitch):
        self.camera.orbit(delta_yaw, delta_pitch)

    def dolly_view(self, delta):
        self.camera.dolly(delta)

    def adjust_draw_distance(self, pixel_delta):
        self.camera.adjust_draw_distance(self.pixels_to_world(pixel_delta))

    def reset_view(self):
        self.camera.reset()

    # -- appearance -------------------------------------------------------
    def set_color(self, color):
        self.color = tuple(color)

    def next_color(self):
        self.color_index = (self.color_index + 1) % len(COLOR_PALETTE)
        self.color = COLOR_PALETTE[self.color_index]
        return self.color

    def set_thickness(self, thickness):
        self.thickness = max(1, min(int(thickness), 20))

    def toggle_snap(self):
        self.snap_enabled = not self.snap_enabled
        self.last_snap_target = None
        return self.snap_enabled

    def adjust_snap_radius(self, delta):
        self.snap_radius = clamp(self.snap_radius + delta, 5.0, 400.0)
        return self.snap_radius

    # -- rendering --------------------------------------------------------
    def _depth_appearance(self, depth):
        ratio = self.camera.draw_distance / max(depth, 1.0)
        thickness_scale = clamp(ratio, MIN_THICKNESS_SCALE, MAX_THICKNESS_SCALE)
        shade = clamp(MIN_SHADE + (1.0 - MIN_SHADE) * min(ratio, 1.0), MIN_SHADE, 1.0)
        return thickness_scale, shade

    def _draw_polyline(self, stroke, points, depths):
        for index in range(1, len(points)):
            p0, p1 = points[index - 1], points[index]
            scale, shade = self._depth_appearance((depths[index - 1] + depths[index]) * 0.5)
            thickness = max(1, int(round(stroke.thickness * scale)))

            if stroke.selected:
                glow = thickness + 4
                cv2.line(self.canvas, p0, p1, (255, 255, 0), glow, cv2.LINE_AA)

            color = tuple(int(channel * shade) for channel in stroke.color)
            cv2.line(self.canvas, p0, p1, color, thickness, cv2.LINE_AA)

    def _draw_stroke(self, stroke):
        if stroke.is_empty():
            return

        points = []
        depths = []
        for world_point in stroke.world_points():
            projected = self.project(world_point)
            if projected is None:
                if points:
                    self._flush(points, depths, stroke)
                points = []
                depths = []
                continue
            points.append((int(round(projected[0])), int(round(projected[1]))))
            depths.append(projected[2])
        self._flush(points, depths, stroke)

    def _flush(self, points, depths, stroke):
        if not points:
            return
        if len(points) == 1:
            scale, shade = self._depth_appearance(depths[0])
            color = tuple(int(channel * shade) for channel in stroke.color)
            if stroke.selected:
                color = (255, 255, 0)
            cv2.circle(
                self.canvas,
                points[0],
                max(1, int(round(stroke.thickness * scale / 2.0))),
                color,
                -1,
                cv2.LINE_AA,
            )
            return
        self._draw_polyline(stroke, points, depths)

    def _draw_plane_guide(self):
        camera = self.camera
        center = camera.draw_plane_point(camera.center_x, camera.center_y)
        half = camera.draw_distance * 0.45
        right = camera.right()
        up = camera.up()

        corners = [
            center - right * half - up * half,
            center + right * half - up * half,
            center + right * half + up * half,
            center - right * half + up * half,
        ]
        projected = [camera.project(corner) for corner in corners]
        if all(p is not None for p in projected):
            ring = [(int(round(p[0])), int(round(p[1]))) for p in projected]
            for index in range(4):
                cv2.line(self.canvas, ring[index], ring[(index + 1) % 4], (28, 28, 28), 1, cv2.LINE_AA)

        axis_x = [
            center - right * half * 1.15,
            center + right * half * 1.15,
        ]
        axis_y = [
            center - up * half * 1.15,
            center + up * half * 1.15,
        ]
        for axis, color in ((axis_x, (0, 0, 190)), (axis_y, (200, 90, 0))):
            a = camera.project(axis[0])
            b = camera.project(axis[1])
            if a is not None and b is not None:
                cv2.line(
                    self.canvas,
                    (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))),
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # A short stub along +Z makes the draw plane's offset from the view
        # plane visible once the camera is orbited.
        depth_axis = [center, center + camera.forward() * (camera.draw_distance * 0.22)]
        a = camera.project(depth_axis[0])
        b = camera.project(depth_axis[1])
        if a is not None and b is not None:
            cv2.line(
                self.canvas,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                (0, 170, 170),
                1,
                cv2.LINE_AA,
            )

        origin = camera.project(center)
        if origin is not None:
            cv2.circle(self.canvas, (int(round(origin[0])), int(round(origin[1]))), 3, (0, 170, 170), 1, cv2.LINE_AA)

    def re_render_frame(self):
        self.canvas[:] = 0

        if self.show_draw_plane:
            self._draw_plane_guide()

        for stroke in self.scene.strokes:
            self._draw_stroke(stroke)

        if self.scene.active_stroke is not None:
            self._draw_stroke(self.scene.active_stroke)

        return self.canvas

    def get_foreground_mask(self):
        return (np.any(self.canvas > 0, axis=2).astype(np.uint8)) * 255

    # -- overlays drawn straight onto the video frame ---------------------
    def draw_selection_overlay(self, frame):
        selected = self.scene.selected_strokes()
        if not selected:
            return

        screen_points = []
        depths = []
        for stroke in selected:
            for world_point in stroke.world_points():
                projected = self.project(world_point)
                if projected is None:
                    continue
                screen_points.append(projected)
                depths.append(projected[2])
        if not screen_points:
            return

        min_x = int(min(p[0] for p in screen_points)) - 10
        max_x = int(max(p[0] for p in screen_points)) + 10
        min_y = int(min(p[1] for p in screen_points)) - 10
        max_y = int(max(p[1] for p in screen_points)) + 10

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)
        cv2.putText(
            frame,
            "SELECTED",
            (min_x, max(min_y - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        centroid = self.scene.selection_centroid()
        centroid_projection = self.project(centroid)
        if centroid_projection is not None:
            cv2.drawMarker(
                frame,
                (int(centroid_projection[0]), int(centroid_projection[1])),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                16,
                2,
            )

        near = min(depths)
        far = max(depths)
        cv2.putText(
            frame,
            f"depth {near:.0f}..{far:.0f}",
            (min_x, min(max_y + 16, self.height - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )

    def draw_draw_plane_marker(self, frame, screen_point, color):
        """Marker for the 3D point a drawn vertex will land on."""
        projected = self.project(self.draw_plane_point(screen_point[0], screen_point[1]))
        if projected is None:
            return
        point = (int(round(projected[0])), int(round(projected[1])))
        cv2.drawMarker(frame, point, color, cv2.MARKER_CROSS, 14, 1)
        if self.last_snap_target is not None:
            cv2.circle(frame, point, 9, (0, 255, 255), 1, cv2.LINE_AA)

    # -- persistence ------------------------------------------------------
    def clear(self):
        self.scene.clear()
        self.reset_view()
        self.canvas[:] = 0
        self.last_snap_target = None

    def save_to_file(self, filepath):
        if not filepath.endswith(".png"):
            filepath += ".png"

        cv2.imwrite(filepath, self.re_render_frame())

        json_filepath = filepath.replace(".png", ".json")
        data = {
            "version": 2,
            "camera": self.camera.to_dict(),
            "color_index": self.color_index,
            "snap_enabled": self.snap_enabled,
            "snap_radius": self.snap_radius,
            "scene": self.scene.to_dict(),
        }
        with open(json_filepath, "w") as handle:
            json.dump(data, handle, indent=4)
        print(f"Artwork saved to {filepath} and {json_filepath}")

    def load_from_file(self, filepath, merge=False):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False

        with open(filepath, "r") as handle:
            data = json.load(handle)

        if not merge:
            self.scene.clear()
            self.reset_view()

        if "camera" in data and not merge:
            self.camera.load_dict(data["camera"])
        if "color_index" in data and not merge:
            self.color_index = int(data["color_index"]) % len(COLOR_PALETTE)
            self.color = COLOR_PALETTE[self.color_index]
        if "snap_enabled" in data and not merge:
            self.snap_enabled = bool(data["snap_enabled"])
        if "snap_radius" in data and not merge:
            self.snap_radius = float(data["snap_radius"])

        if "scene" in data:
            self.scene.load_dict(data["scene"], merge=merge)
        else:
            self.scene.load_dict(data, merge=merge)

        self.re_render_frame()
        print(f"Artwork loaded from {filepath} (merge={merge})")
        return True
