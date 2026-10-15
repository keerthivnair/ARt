import numpy as np
import math
from drawing.geometry import mat4_identity, mat4_invert, transform_point

def _point_segment_dist_sq_2d(px, py, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = 0.0 if len_sq == 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    ex, ey = ax + dx * t - px, ay + dy * t - py
    return ex * ex + ey * ey


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


    def erase_in_screen_circle(self, project, center, radius):
        """Cut this stroke with a circular eraser defined in screen pixels.

        Returns None if the eraser did not touch the stroke. Otherwise returns a
        list of local-space point lists (each with >= 2 points) that survive.
        An empty list means the whole stroke was erased.
        """
        if not self.points:
            return None

        cx, cy = float(center[0]), float(center[1])
        r2 = radius * radius
        step = max(radius / 3.0, 2.0)  # max pixel spacing of vertices near the eraser
        reach_sq = (radius + step) ** 2

        local = [np.asarray(p, dtype=float) for p in self.points]
        projected = [project(transform_point(self.transform, p)) for p in local]

        # Densify only the segments that pass near the eraser, so the cut is
        # accurate even when the original samples are far apart.
        dense = []
        for i, (lp, sp) in enumerate(zip(local, projected)):
            dense.append((lp, sp))
            if i + 1 == len(local):
                break
            nxt, nsp = local[i + 1], projected[i + 1]
            if sp is None or nsp is None:
                continue
            if _point_segment_dist_sq_2d(cx, cy, sp, nsp) > reach_sq:
                continue
            length = math.hypot(nsp[0] - sp[0], nsp[1] - sp[1])
            pieces = int(math.ceil(length / step))
            for k in range(1, pieces):
                mid = lp + (nxt - lp) * (k / pieces)
                dense.append((mid, project(transform_point(self.transform, mid))))

        runs, current, touched = [], [], False
        for lp, sp in dense:
            inside = sp is not None and (sp[0] - cx) ** 2 + (sp[1] - cy) ** 2 <= r2
            if inside:
                touched = True
                if len(current) >= 2:
                    runs.append(current)
                current = []
            else:
                current.append((float(lp[0]), float(lp[1]), float(lp[2])))
        if len(current) >= 2:
            runs.append(current)

        return runs if touched else None



    def all_points_within(self, a, ab, ab_len_sq, radius):
        """Return True if every world point of this stroke is within ``radius``
        of the segment ``a->b`` (used to decide whether to delete a stroke
        entirely rather than just trim it)."""
        wp = self.world_points()
        if not wp:
            return True
        radius_squared = radius * radius
        for p in wp:
            pa = np.array(p, dtype=float) - a
            t = float(np.dot(pa, ab) / ab_len_sq)
            if t < 0.0:
                closest = a
            elif t > 1.0:
                closest = a + ab
            else:
                closest = a + ab * t
            if float(np.dot(np.array(p, dtype=float) - closest, np.array(p, dtype=float) - closest)) > radius_squared:
                return False
        return True

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