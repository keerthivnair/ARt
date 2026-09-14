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
        
        orig_x = int((x - cx) * inv_scale + cx)
        orig_y = int((y - cy) * inv_scale + cy)
        
        self.active_stroke.add_point((orig_x, orig_y))

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
                cv2.circle(self.canvas, pt, stroke.thickness // 2, stroke.color, -1)
            return

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
