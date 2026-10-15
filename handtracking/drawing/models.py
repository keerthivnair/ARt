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
        self.remove_world_points_near_segment(
            np.asarray(target, dtype=float),
            np.asarray(target, dtype=float),
            1.0,
            radius,
        )

    def remove_world_points_near_segment(self, a, ab, ab_len_sq, radius):
        radius_squared = radius * radius
        old_len = len(self.points)
        keep = []
        for point in self.points:
            px, py, pz = transform_point(self.transform, point)
            ap = np.array([px, py, pz], dtype=float) - a
            t = float(np.dot(ap, ab) / ab_len_sq)
            if t < 0.0:
                closest = a
            elif t > 1.0:
                closest = a + ab
            else:
                closest = a + ab * t
            delta = np.array([px, py, pz], dtype=float) - closest
            if float(np.dot(delta, delta)) > radius_squared:
                keep.append(point)

        if len(keep) < 2 and old_len >= 2:
            # Don't leave a 0/1-point remnant that renders as a dot or snaps
            # across the erased area. Treat the stroke as removed instead.
            self.points = []
            return

        self.points = keep
        if len(self.points) >= 2:
            self.points = self._split_at_erased_gaps()

    def any_segment_within(self, a, ab, ab_len_sq, radius):
        """Return True if any part of this stroke is within ``radius`` of the
        segment ``a->b``."""
        radius_squared = radius * radius
        wp = self.world_points()
        if len(wp) < 2:
            if wp:
                d = float(np.linalg.norm(np.array(wp[0], dtype=float) - _closest_point_on_segment(a, ab, ab_len_sq, np.array(wp[0], dtype=float))))
                return d <= radius
            return False
        for i in range(len(wp) - 1):
            p0 = np.array(wp[i], dtype=float)
            p1 = np.array(wp[i + 1], dtype=float)
            if _segment_to_segment_distance_squared(a, ab, ab_len_sq, p0, p1) <= radius_squared:
                return True
        # Also check endpoints against the brush segment in case the stroke is
        # a single point or the brush barely touches an endpoint.
        for p in wp:
            pa = np.array(p, dtype=float) - a
            t = float(np.dot(pa, ab) / ab_len_sq)
            if t < 0.0:
                closest = a
            elif t > 1.0:
                closest = a + ab
            else:
                closest = a + ab * t
            if float(np.linalg.norm(np.array(p, dtype=float) - closest)) <= radius:
                return True
        return False

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

    def _split_at_erased_gaps(self):
        """If erase created a large gap between consecutive vertices, break the
        polyline into independent segments so we do not draw a line across the
        missing region.

        This returns the first segment only; additional segments are re-added
        as new strokes by the scene on erase.
        """
        if len(self.points) < 2:
            return list(self.points)

        gap_threshold = 60.0
        segments = []
        current = [self.points[0]]
        prev_w = transform_point(self.transform, self.points[0])
        for point in self.points[1:]:
            cur_w = transform_point(self.transform, point)
            if float(np.linalg.norm(cur_w - prev_w)) > gap_threshold:
                if len(current) >= 2:
                    segments.append(current)
                current = [point]
            else:
                current.append(point)
            prev_w = cur_w
        if len(current) >= 2:
            segments.append(current)
        if not segments:
            return [self.points[0]]
        self.points = segments[0]
        return self.points

    def split_after_erasing(self, scene):
        """If erase split this stroke into multiple chunks, turn the extra chunks
        into new strokes so the canvas can render each separately."""
        if len(self.points) < 2:
            return
        gaps = []
        prev_w = transform_point(self.transform, self.points[0])
        for idx in range(1, len(self.points)):
            cur_w = transform_point(self.transform, self.points[idx])
            if float(np.linalg.norm(cur_w - prev_w)) > 60.0:
                gaps.append(idx)
            prev_w = cur_w
        if not gaps:
            return
        pieces = []
        start = 0
        for g in gaps + [len(self.points)]:
            piece = self.points[start:g]
            if len(piece) >= 2:
                pieces.append(piece)
            start = g
        if len(pieces) <= 1:
            return
        self.points = pieces[0]
        for piece in pieces[1:]:
            new_stroke = Stroke3D(self.color, self.thickness, piece, self.transform)
            new_stroke.selected = self.selected
            scene.strokes.append(new_stroke)

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


def _closest_point_on_segment(a, ab, ab_len_sq, p):
    pa = p - a
    t = float(np.dot(pa, ab) / ab_len_sq)
    if t < 0.0:
        return a
    if t > 1.0:
        return a + ab
    return a + ab * t


def _segment_to_segment_distance_squared(a, ab, ab_len_sq, p0, p1):
    ba = p1 - p0
    ba_len_sq = float(np.dot(ba, ba)) or 1.0
    best = float("inf")
    for t in (0.0, 1.0):
        q = a + ab * t
        c = _closest_point_on_segment(p0, ba, ba_len_sq, q)
        best = min(best, float(np.dot(q - c, q - c)))
    for t in (0.0, 1.0):
        q = p0 + ba * t
        c = _closest_point_on_segment(a, ab, ab_len_sq, q)
        best = min(best, float(np.dot(q - c, q - c)))
    m = np.array([
        [ab[0], -ba[0]],
        [ab[1], -ba[1]],
        [ab[2], -ba[2]],
    ])
    rhs = p0 - a
    try:
        s, residuals, rank, sv = np.linalg.lstsq(m, rhs, rcond=None)
    except np.linalg.LinAlgError:
        s = np.array([0.0, 0.0])
    s0 = float(np.clip(s[0], 0.0, 1.0))
    s1 = float(np.clip(s[1], 0.0, 1.0))
    q0 = a + ab * s0
    q1 = p0 + ba * s1
    best = min(best, float(np.dot(q0 - q1, q0 - q1)))
    return best
