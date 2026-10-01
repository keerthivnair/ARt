import math

import numpy as np

from config import SNAP_GRID
from drawing.geometry import (
    mat4_multiply,
    mat4_rotate_about,
    mat4_scale_about,
    mat4_translation,
    transform_point,
)
from drawing.models import Stroke3D


class VoxelIndex:
    """Spatial hash over drawn vertices used for vertex snapping.

    Every vertex of every committed stroke is binned into a cubic voxel of
    ``cell`` world units, so a snap query only has to look at the 27 voxels
    around the query point instead of every vertex in the scene.
    """

    def __init__(self, cell=SNAP_GRID):
        self.cell = float(cell)
        self.buckets = {}

    def clear(self):
        self.buckets = {}

    def _key(self, point):
        return (
            int(math.floor(point[0] / self.cell)),
            int(math.floor(point[1] / self.cell)),
            int(math.floor(point[2] / self.cell)),
        )

    def insert(self, point, owner):
        self.buckets.setdefault(self._key(point), []).append(
            (float(point[0]), float(point[1]), float(point[2]), owner)
        )

    def nearest(self, point, radius, exclude_owner=None):
        """Closest indexed vertex within ``radius``, or None."""
        base = self._key(point)
        span = max(1, int(math.ceil(radius / self.cell)))
        target = np.asarray(point, dtype=float)
        best = None
        best_distance = radius
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for dz in range(-span, span + 1):
                    bucket = self.buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if not bucket:
                        continue
                    for vx, vy, vz, owner in bucket:
                        if exclude_owner is not None and owner is exclude_owner:
                            continue
                        distance = math.sqrt(
                            (vx - target[0]) ** 2 + (vy - target[1]) ** 2 + (vz - target[2]) ** 2
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best = (vx, vy, vz)
        return best


class Scene:
    """Owns the 3D strokes and every 3D edit performed on them."""

    def __init__(self):
        self.strokes = []
        self.active_stroke = None
        self.voxels = VoxelIndex()
        self._index_dirty = True

    # -- drawing ----------------------------------------------------------
    def add_point_to_active_stroke(self, world_point, color, thickness):
        if self.active_stroke is None:
            self.active_stroke = Stroke3D(color, thickness)
        self.active_stroke.add_point(world_point)
        self.voxels.insert(world_point, self.active_stroke)

    def finalize_active_stroke(self):
        if self.active_stroke is not None and not self.active_stroke.is_empty():
            self.strokes.append(self.active_stroke)
        self.active_stroke = None

    def snap_point(self, world_point, radius):
        """Snap a freshly drawn vertex to the nearest already drawn vertex.

        The active stroke is excluded, so a stroke never snaps to itself.
        Returns ``(snapped_point, hit_or_None)``.
        """
        if radius is None or radius <= 0.0:
            return world_point, None

        self._refresh_index()
        hit = self.voxels.nearest(world_point, radius, exclude_owner=self.active_stroke)
        if hit is None:
            return world_point, None
        return np.array(hit, dtype=float), hit

    def _refresh_index(self):
        if not self._index_dirty:
            return
        self.voxels.clear()
        for stroke in self.strokes:
            for point in stroke.world_points():
                self.voxels.insert(point, stroke)
        self._index_dirty = False

    def _invalidate_index(self):
        self._index_dirty = True

    # -- selection --------------------------------------------------------
    def has_selection(self):
        return any(stroke.selected for stroke in self.strokes)

    def deselect_all(self):
        for stroke in self.strokes:
            stroke.selected = False

    def select_in_polygon(self, polygon_screen_points, camera):
        import cv2

        self.deselect_all()
        if len(polygon_screen_points) < 3:
            return 0

        polygon = np.array(polygon_screen_points, dtype=np.float32)
        count = 0
        for stroke in self.strokes:
            for point in stroke.world_points():
                projected = camera.project(point)
                if projected is None:
                    continue
                if cv2.pointPolygonTest(polygon, (float(projected[0]), float(projected[1])), False) >= 0:
                    stroke.selected = True
                    count += 1
                    break
        return count

    def selected_strokes(self):
        return [stroke for stroke in self.strokes if stroke.selected]

    def selection_centroid(self):
        points = [p for stroke in self.selected_strokes() for p in stroke.world_points()]
        if not points:
            return np.array([0.0, 0.0, 0.0])
        return np.array(points, dtype=float).mean(axis=0)

    # -- 3D edits ---------------------------------------------------------
    def translate_selected(self, delta):
        delta = np.asarray(delta, dtype=float)
        if not self.has_selection():
            return
        for stroke in self.selected_strokes():
            stroke.transform = mat4_multiply(mat4_translation(delta), stroke.transform)
        self._invalidate_index()

    def translate_selected_depth(self, amount, axis):
        self.translate_selected(np.asarray(axis, dtype=float) * float(amount))

    def rotate_selected(self, axis, angle):
        if not self.has_selection() or abs(angle) < 1e-9:
            return
        pivot = self.selection_centroid()
        for stroke in self.selected_strokes():
            stroke.transform = mat4_rotate_about(stroke.transform, pivot, axis, angle)
        self._invalidate_index()

    def scale_selected(self, factor):
        if not self.has_selection() or abs(factor - 1.0) < 1e-9:
            return
        pivot = self.selection_centroid()
        for stroke in self.selected_strokes():
            stroke.transform = mat4_scale_about(stroke.transform, pivot, factor)
        self._invalidate_index()

    def move_selected_to(self, world_point):
        if not self.has_selection():
            return
        delta = np.asarray(world_point, dtype=float) - self.selection_centroid()
        self.translate_selected(delta)

    def rotate_all(self, axis, angle, pivot):
        for stroke in self.strokes:
            stroke.transform = mat4_rotate_about(stroke.transform, pivot, axis, angle)
        self._invalidate_index()

    def translate_all(self, delta):
        delta = np.asarray(delta, dtype=float)
        for stroke in self.strokes:
            stroke.transform = mat4_multiply(mat4_translation(delta), stroke.transform)
        self._invalidate_index()

    # -- erasing ----------------------------------------------------------
    def erase_near(self, world_point, radius):
        target = np.asarray(world_point, dtype=float)
        for stroke in self.strokes:
            stroke.remove_world_points_near(target, radius)
        self.strokes = [stroke for stroke in self.strokes if not stroke.is_empty()]
        self._invalidate_index()

    def clear(self):
        self.strokes = []
        self.active_stroke = None
        self.voxels.clear()
        self._index_dirty = True

    # -- persistence ------------------------------------------------------
    def to_dict(self):
        return {
            "strokes": [stroke.to_dict() for stroke in self.strokes],
        }

    def load_dict(self, data, merge=False):
        if not merge:
            self.clear()
        for stroke_data in data.get("strokes", []):
            self.strokes.append(Stroke3D.from_dict(stroke_data))
        self._invalidate_index()
