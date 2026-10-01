import numpy as np

from drawing.geometry import mat4_identity, mat4_invert, transform_point


class Stroke3D:
    """A polyline of 3D vertices living on the draw plane.

    Vertices are kept in the stroke's own local space and placed in the world by
    ``transform`` (a 4x4 matrix). Selection edits such as rotate, translate and
    scale only rewrite that matrix, so the original drawn geometry is never
    destroyed and a transform can be undone by composition.
    """

    def __init__(self, color, thickness, points=None, transform=None):
        self.points = list(points) if points else []
        self.color = tuple(color)
        self.thickness = int(thickness)
        self.selected = False
        self.transform = mat4_identity() if transform is None else np.array(transform, dtype=float)

    def add_point(self, point):
        self.points.append((float(point[0]), float(point[1]), float(point[2])))

    def is_empty(self):
        return len(self.points) == 0

    def world_point(self, index):
        return transform_point(self.transform, self.points[index])

    def world_points(self):
        return [transform_point(self.transform, p) for p in self.points]

    def local_points_from_world(self, world_points):
        inverse = mat4_invert(self.transform)
        return [transform_point(inverse, p) for p in world_points]

    def centroid(self):
        if not self.points:
            return np.array([0.0, 0.0, 0.0])
        stacked = np.array(self.world_points(), dtype=float)
        return stacked.mean(axis=0)

    def remove_world_points_near(self, target, radius):
        radius_squared = radius * radius
        keep = []
        for point in self.points:
            delta = transform_point(self.transform, point) - np.asarray(target, dtype=float)
            if float(np.dot(delta, delta)) > radius_squared:
                keep.append(point)
        self.points = keep

    def to_dict(self):
        return {
            "points": [list(p) for p in self.points],
            "color": list(self.color),
            "thickness": self.thickness,
            "selected": self.selected,
            "transform": self.transform.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        thickness = int(data.get("thickness", 5))
        stroke = cls(tuple(data.get("color", (0, 255, 0))), thickness)
        for point in data.get("points", []):
            # Older 2D saves only had (x, y); they are loaded onto z = 0.
            if len(point) >= 3:
                stroke.add_point((point[0], point[1], point[2]))
            elif len(point) == 2:
                stroke.add_point((point[0], point[1], 0.0))
        stroke.selected = bool(data.get("selected", False))

        transform = data.get("transform")
        if transform is not None and np.array(transform).shape == (4, 4):
            stroke.transform = np.array(transform, dtype=float)
        return stroke
