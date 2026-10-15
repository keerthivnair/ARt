import math

import numpy as np

from config import (
    CAMERA_FOV,
    DRAW_PLANE_DISTANCE,
    MAX_CAMERA_DISTANCE,
    MAX_DRAW_DISTANCE,
    MIN_CAMERA_DISTANCE,
    MIN_DRAW_DISTANCE,
    MIN_PROJECTION_DEPTH,
)

EPS = 1e-9

WORLD_UP = (0.0, -1.0, 0.0)
WORLD_FALLBACK_UP = (0.0, 0.0, -1.0)


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def vec3(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z], dtype=float)


def length(vector):
    return float(np.linalg.norm(vector))


def normalize(vector):
    magnitude = length(vector)
    if magnitude <= EPS:
        return vec3(0.0, 0.0, 1.0)
    return np.asarray(vector, dtype=float) / magnitude


def dot(a, b):
    return float(np.dot(a, b))


def cross(a, b):
    return np.cross(a, b)


def angle_difference(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def mat4_identity():
    return np.eye(4, dtype=float)


def mat4_translation(offset):
    matrix = np.eye(4, dtype=float)
    matrix[0, 3] = float(offset[0])
    matrix[1, 3] = float(offset[1])
    matrix[2, 3] = float(offset[2])
    return matrix


def mat4_scale(factor):
    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = float(factor)
    return matrix


def mat4_rotation(axis, angle):
    unit = normalize(axis)
    x, y, z = unit
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    one_minus = 1.0 - cos_a
    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = one_minus * x * x + cos_a
    matrix[0, 1] = one_minus * x * y - sin_a * z
    matrix[0, 2] = one_minus * x * z + sin_a * y
    matrix[1, 0] = one_minus * x * y + sin_a * z
    matrix[1, 1] = one_minus * y * y + cos_a
    matrix[1, 2] = one_minus * y * z - sin_a * x
    matrix[2, 0] = one_minus * x * z - sin_a * y
    matrix[2, 1] = one_minus * y * z + sin_a * x
    matrix[2, 2] = one_minus * z * z + cos_a
    return matrix


def mat4_multiply(a, b):
    return a @ b


def transform_point(matrix, point):
    vector = matrix @ np.array([point[0], point[1], point[2], 1.0], dtype=float)
    return vector[:3]


def transform_direction(matrix, vector):
    result = matrix @ np.array([vector[0], vector[1], vector[2], 0.0], dtype=float)
    return result[:3]


def mat4_invert(matrix):
    return np.linalg.inv(matrix)


def mat4_rotate_about(matrix, pivot, axis, angle):
    point = np.asarray(pivot, dtype=float)
    rotation = mat4_multiply(
        mat4_translation(point),
        mat4_multiply(
            mat4_rotation(axis, angle),
            mat4_multiply(mat4_translation(-point), matrix),
        ),
    )
    return rotation


def mat4_scale_about(matrix, pivot, factor):
    point = np.asarray(pivot, dtype=float)
    return mat4_multiply(
        mat4_translation(point),
        mat4_multiply(mat4_scale(factor), mat4_multiply(mat4_translation(-point), matrix)),
    )


class Camera:
    """Orbit camera with a perspective projection and a draw plane in front of it.

    ``target`` is the orbit centre in world space, ``distance`` the orbit
    radius. The draw plane is the plane whose points are ``draw_distance``
    units in front of the camera along its own forward axis, so the plane
    stays parallel to the view plane no matter how the camera is oriented.
    """

    def __init__(self, width, height, fov_deg=CAMERA_FOV, draw_distance=DRAW_PLANE_DISTANCE):
        self.width = float(width)
        self.height = float(height)
        self.center_x = width / 2.0
        self.center_y = height / 2.0
        self.focal = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

        self.target = vec3(0.0, 0.0, 0.0)
        self.distance = float(draw_distance)
        self.draw_distance = float(draw_distance)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self._basis = None

    # -- basis ------------------------------------------------------------
    def _get_basis(self):
        if self._basis is None:
            cos_pitch = math.cos(self.pitch)
            offset_dir = vec3(
                cos_pitch * math.sin(self.yaw),
                -math.sin(self.pitch),
                cos_pitch * math.cos(self.yaw),
            )
            forward = normalize(-offset_dir)

            reference = np.array(WORLD_UP, dtype=float)
            if abs(dot(reference, forward)) > 0.999:
                reference = np.array(WORLD_FALLBACK_UP, dtype=float)

            right = normalize(cross(reference, forward))
            up = normalize(cross(forward, right))

            if abs(self.roll) > EPS:
                roll_matrix = mat4_rotation(forward, self.roll)
                right = transform_direction(roll_matrix, right)
                up = transform_direction(roll_matrix, up)

            self._basis = (right, up, forward)
        return self._basis

    def invalidate(self):
        self._basis = None

    def forward(self):
        return self._get_basis()[2]

    def right(self):
        return self._get_basis()[0]

    def up(self):
        return self._get_basis()[1]

    def position(self):
        _, _, forward = self._get_basis()
        return self.target - forward * self.distance

    # -- projection -------------------------------------------------------
    def to_camera_space(self, point):
        _, _, forward = self._get_basis()
        return point - self.position()

    def project(self, point, min_depth=MIN_PROJECTION_DEPTH):
        """Project a world point to screen pixels, or None if behind the camera."""
        right, up, forward = self._get_basis()
        relative = np.asarray(point, dtype=float) - self.position()
        depth = dot(relative, forward)
        if depth < min_depth:
            return None
        focal = self.focal
        screen_x = self.center_x + focal * dot(relative, right) / depth
        screen_y = self.center_y - focal * dot(relative, up) / depth
        return (screen_x, screen_y, depth)

    def point_on_plane(self, screen_x, screen_y, distance=None):
        """Unproject a screen pixel onto a plane in front of the camera."""
        plane_distance = self.draw_distance if distance is None else float(distance)
        right, up, forward = self._get_basis()
        cam_x = (screen_x - self.center_x) * plane_distance / self.focal
        cam_y = -(screen_y - self.center_y) * plane_distance / self.focal
        return self.position() + right * cam_x + up * cam_y + forward * plane_distance

    def draw_plane_point(self, screen_x, screen_y):
        return self.point_on_plane(screen_x, screen_y, self.draw_distance)

    def pixels_to_world(self, pixels, distance=None):
        plane_distance = self.draw_distance if distance is None else float(distance)
        return abs(pixels) * plane_distance / self.focal

    # -- camera manipulation ---------------------------------------------
    def orbit(self, delta_yaw, delta_pitch):
        self.yaw += delta_yaw
        self.pitch = clamp(self.pitch + delta_pitch, -1.45, 1.45)
        self.invalidate()

    def roll_by(self, angle):
        self.roll += angle
        self.invalidate()

    def pan(self, delta_right, delta_up):
        right, up, _ = self._get_basis()
        offset = right * delta_right + up * delta_up
        self.target = self.target + offset
        self.invalidate()

    def shift(self, delta):
        self.target = self.target + np.asarray(delta, dtype=float)
        self.invalidate()

    def dolly(self, delta):
        self.distance = clamp(self.distance + delta, MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE)
        self.invalidate()

    def set_draw_distance(self, value):
        self.draw_distance = clamp(value, MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
        self.invalidate()

    def adjust_draw_distance(self, delta):
        self.set_draw_distance(self.draw_distance + delta)

    def reset(self):
        self.target = vec3(0.0, 0.0, 0.0)
        self.distance = self.draw_distance
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.invalidate()

    # -- persistence ------------------------------------------------------
    def to_dict(self):
        return {
            "focal": self.focal,
            "target": [float(v) for v in self.target],
            "distance": self.distance,
            "draw_distance": self.draw_distance,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
        }

    def load_dict(self, data):
        self.focal = float(data.get("focal", self.focal))
        target = data.get("target")
        if target is not None and len(target) == 3:
            self.target = vec3(*[float(v) for v in target])
        self.distance = clamp(
            float(data.get("distance", self.distance)),
            MIN_CAMERA_DISTANCE,
            MAX_CAMERA_DISTANCE,
        )
        self.draw_distance = clamp(
            float(data.get("draw_distance", self.draw_distance)),
            MIN_DRAW_DISTANCE,
            MAX_DRAW_DISTANCE,
        )
        self.yaw = float(data.get("yaw", 0.0))
        self.pitch = clamp(float(data.get("pitch", 0.0)), -1.45, 1.45)
        self.roll = float(data.get("roll", 0.0))
        self.invalidate()
