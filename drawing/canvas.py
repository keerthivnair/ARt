import cv2
import numpy as np
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    DRAWING_COLOR,
    DRAWING_THICKNESS,
    BG_COLOR,
    COLOR_PALETTE,
)


class DrawingCanvas:
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.drawing = False
        self.color = DRAWING_COLOR
        self.thickness = DRAWING_THICKNESS
        self.prev_point = None
        self.color_index = 0

    def start_drawing(self, point):
        self.drawing = True
        self.prev_point = point

    def stop_drawing(self):
        self.drawing = False
        self.prev_point = None

    def draw_line(self, point):
        if not self.drawing:
            self.drawing = True
            self.prev_point = point
            return
        if self.prev_point is None:
            self.prev_point = point
            return
        cv2.line(
            self.canvas,
            self.prev_point,
            point,
            self.color,
            self.thickness,
        )
        self.prev_point = point

    def set_color(self, color):
        self.color = color

    def next_color(self):
        self.color_index = (self.color_index + 1) % len(COLOR_PALETTE)
        self.color = COLOR_PALETTE[self.color_index]
        return self.color

    def set_thickness(self, thickness):
        self.thickness = max(1, min(thickness, 20))

    def get_composite(self):
        return cv2.addWeighted(self.canvas, 1.0, np.zeros_like(self.canvas), 0.0, 0)

    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_undo_canvas(self):
        return self.canvas.copy()
