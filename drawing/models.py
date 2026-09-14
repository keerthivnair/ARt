class Transform:
    def __init__(self):
        self.scale = 1.0
        self.tx = 0.0
        self.ty = 0.0

    def apply(self, point, center=(0, 0)):
        x, y = point
        cx, cy = center

        scaled_x = int((x - cx) * self.scale + cx)
        scaled_y = int((y - cy) * self.scale + cy)
        
        return (scaled_x, scaled_y)


class Stroke:
    def __init__(self, color, thickness):
        self.points = []
        self.color = color
        self.thickness = thickness

    def add_point(self, point):
        self.points.append(point)

    def is_empty(self):
        return len(self.points) == 0
