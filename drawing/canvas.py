import cv2
import numpy as np
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    DRAWING_COLOR,
    DRAWING_THICKNESS,
    COLOR_PALETTE,
)
from drawing.models import Stroke, Transform


class DrawingCanvas:
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.strokes = []
        self.active_stroke = None
        self.transform = Transform()
        
        self.color_index = 0
        self.color = COLOR_PALETTE[self.color_index]
        self.thickness = DRAWING_THICKNESS
        
        self.center = (width // 2, height // 2)

    def add_point_to_active_stroke(self, point):
        if self.active_stroke is None:
            self.active_stroke = Stroke(self.color, self.thickness)

        inv_scale = 1.0 / self.transform.scale
        cx, cy = self.center
        x, y = point
        
        orig_x = int((x - cx - self.transform.tx) * inv_scale + cx)
        orig_y = int((y - cy - self.transform.ty) * inv_scale + cy)
        
        self.active_stroke.add_point((orig_x, orig_y))

    def remove_points_from_strokes(self, point, radius=15):
        # FIX: erase by proximity instead of exact-pixel match. `radius` is
        # in screen-space pixels (matches the circle drawn in main.py), so
        # it also needs to be converted into original/unscaled space, same
        # as the point itself, otherwise erasing feels wrong when zoomed
        # in/out via the pinch gesture.
        inv_scale = 1.0 / self.transform.scale
        cx, cy = self.center
        x, y = point

        orig_x = (x - cx - self.transform.tx) * inv_scale + cx
        orig_y = (y - cy - self.transform.ty) * inv_scale + cy
        orig_radius = radius * inv_scale

        for stroke in self.strokes:
            stroke.remove_points_near(orig_x, orig_y, orig_radius)

        # Drop strokes that have been fully erased so they don't pile up.
        self.strokes = [s for s in self.strokes if not s.is_empty()]

        self.re_render_frame()

    def finalize_active_stroke(self):
        if self.active_stroke is not None and not self.active_stroke.is_empty():
            self.strokes.append(self.active_stroke)
        self.active_stroke = None

    def set_color(self, color):
        self.color = color

    def next_color(self):
        self.color_index = (self.color_index + 1) % len(COLOR_PALETTE)
        self.color = COLOR_PALETTE[self.color_index]
        return self.color

    def set_thickness(self, thickness):
        self.thickness = max(1, min(thickness, 20))

    def has_selection(self):
        return any(stroke.selected for stroke in self.strokes)

    def deselect_all(self):
        for stroke in self.strokes:
            stroke.selected = False

    def select_strokes_in_polygon(self, polygon_screen_pts):
        self.deselect_all()
        if len(polygon_screen_pts) < 3:
            return 0

        inv_scale = 1.0 / self.transform.scale
        cx, cy = self.center
        poly_canvas = []
        for (sx, sy) in polygon_screen_pts:
            ox = (sx - cx - self.transform.tx) * inv_scale + cx
            oy = (sy - cy - self.transform.ty) * inv_scale + cy
            poly_canvas.append((ox, oy))

        poly_arr = np.array(poly_canvas, dtype=np.float32)

        count = 0
        for stroke in self.strokes:
            is_inside = False
            for (px, py) in stroke.points:
                dist = cv2.pointPolygonTest(poly_arr, (float(px), float(py)), measureDist=False)
                if dist >= 0:
                    is_inside = True
                    break
            if is_inside:
                stroke.selected = True
                count += 1
        return count

    def move_selected_strokes(self, dx_screen, dy_screen):
        if not self.has_selection():
            return
        inv_scale = 1.0 / self.transform.scale
        cdx = dx_screen * inv_scale
        cdy = dy_screen * inv_scale
        for stroke in self.strokes:
            if stroke.selected:
                stroke.points = [(p[0] + cdx, p[1] + cdy) for p in stroke.points]

    def scale_selected_strokes(self, scale_factor):
        if not self.has_selection() or scale_factor == 1.0:
            return
        selected_points = [p for s in self.strokes if s.selected for p in s.points]
        if not selected_points:
            return
        centroid_x = sum(p[0] for p in selected_points) / len(selected_points)
        centroid_y = sum(p[1] for p in selected_points) / len(selected_points)

        for stroke in self.strokes:
            if stroke.selected:
                stroke.points = [
                    (
                        centroid_x + (p[0] - centroid_x) * scale_factor,
                        centroid_y + (p[1] - centroid_y) * scale_factor,
                    )
                    for p in stroke.points
                ]

    def draw_selection_overlay(self, frame):
        selected_screen_pts = [
            self.transform.apply(p, self.center)
            for s in self.strokes
            if s.selected
            for p in s.points
        ]
        if not selected_screen_pts:
            return

        min_x = min(pt[0] for pt in selected_screen_pts) - 10
        max_x = max(pt[0] for pt in selected_screen_pts) + 10
        min_y = min(pt[1] for pt in selected_screen_pts) - 10
        max_y = max(pt[1] for pt in selected_screen_pts) + 10

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

    def re_render_frame(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for stroke in self.strokes:
            self._draw_stroke(stroke)

        if self.active_stroke is not None:
            self._draw_stroke(self.active_stroke)

        return self.canvas

    def _draw_stroke(self, stroke):
        if len(stroke.points) < 2:
            if len(stroke.points) == 1:
                pt = self.transform.apply(stroke.points[0], self.center)
                color = (255, 255, 0) if stroke.selected else stroke.color
                cv2.circle(self.canvas, pt, stroke.thickness // 2, color, -1)
            return

        # If selected, draw a cyan glow behind the stroke
        if stroke.selected:
            for i in range(1, len(stroke.points)):
                pt1 = self.transform.apply(stroke.points[i - 1], self.center)
                pt2 = self.transform.apply(stroke.points[i], self.center)
                cv2.line(self.canvas, pt1, pt2, (255, 255, 0), stroke.thickness + 4)

        for i in range(1, len(stroke.points)):
            pt1 = self.transform.apply(stroke.points[i - 1], self.center)
            pt2 = self.transform.apply(stroke.points[i], self.center)
            cv2.line(self.canvas, pt1, pt2, stroke.color, stroke.thickness)

    def get_foreground_mask(self):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask

    def clear(self):
        self.strokes = []
        self.active_stroke = None
        self.transform = Transform()
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
