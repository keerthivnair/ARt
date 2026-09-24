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

    def to_dict(self):
        return {
            "scale": self.scale,
            "tx": self.tx,
            "ty": self.ty
        }

    @classmethod
    def from_dict(cls, data):
        t = cls()
        t.scale = data.get("scale", 1.0)
        t.tx = data.get("tx", 0.0)
        t.ty = data.get("ty", 0.0)
        return t


class Stroke:
    def __init__(self, color, thickness):
        self.points = []
        self.color = color
        self.thickness = thickness
        self.selected = False

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

    def to_dict(self):
        return {
            "points": self.points,
            "color": self.color,
            "thickness": self.thickness,
            "selected": self.selected
        }

    @classmethod
    def from_dict(cls, data):
        color = tuple(data.get("color", (0, 255, 0)))
        thickness = data.get("thickness", 5)
        stroke = cls(color, thickness)
        stroke.points = [tuple(p) for p in data.get("points", [])]
        stroke.selected = data.get("selected", False)
        return stroke