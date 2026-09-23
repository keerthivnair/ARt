class Transform:
    def __init__(self):
        self.scale = 1.0
        self.tx = 0.0
        self.ty = 0.0

    def apply(self, point, center=(0, 0)):
        x, y = point
        cx, cy = center

        scaled_x = int((x - cx) * self.scale + cx + self.tx)
        scaled_y = int((y - cy) * self.scale + cy + self.ty)
        
        return (scaled_x, scaled_y)


class Stroke:
    def __init__(self, color, thickness):
        self.points = []
        self.color = color
        self.thickness = thickness

    def add_point(self, point):
        self.points.append(point)

    def remove_point(self, point):
        self.points.remove(point)

    def remove_points_near(self, x, y, radius):
        radius_sq = radius * radius
        self.points = [
            p for p in self.points
            if (p[0] - x) ** 2 + (p[1] - y) ** 2 > radius_sq
        ]

    def is_empty(self):
        return len(self.points) == 0