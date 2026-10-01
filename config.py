SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

DRAWING_COLOR = (0, 255, 0)
DRAWING_THICKNESS = 5
BG_COLOR = (0, 0, 0)

# --- 3D scene -------------------------------------------------------------
# World axes: +X right, +Y down, +Z into the scene (away from the camera).
# The view plane sits at the camera, the draw plane is a plane parallel to it
# placed DRAW_PLANE_DISTANCE units in front of the camera, so every vertex the
# user draws lands in 3D, not on the screen.
DRAW_PLANE_DISTANCE = 600.0
MIN_DRAW_DISTANCE = 120.0
MAX_DRAW_DISTANCE = 2500.0

CAMERA_FOV = 60.0
MIN_CAMERA_DISTANCE = 150.0
MAX_CAMERA_DISTANCE = 2500.0

# Points closer to the camera than this are treated as behind it and skipped.
MIN_PROJECTION_DEPTH = 1.0

# Perceived thickness is rescaled by draw_distance / depth and clamped, which
# gives far away strokes a thinner, darker look.
MIN_THICKNESS_SCALE = 0.4
MAX_THICKNESS_SCALE = 2.5
MIN_SHADE = 0.4

# --- snapping -------------------------------------------------------------
# A newly drawn vertex snaps to the nearest vertex of an already drawn stroke
# when it lands within SNAP_RADIUS world units of it.
SNAP_ENABLED = True
SNAP_RADIUS = 45.0
SNAP_GRID = 45.0

# --- gesture tuning -------------------------------------------------------
ORBIT_SENSITIVITY = 0.005
ROTATE_SENSITIVITY = 1.0
DEPTH_SENSITIVITY = 1.0
DOLLY_SENSITIVITY = 1.0

# Draw the draw plane guide grid onto the render.
DRAW_PLANE_GUIDE = True

HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

PINCH_THRESHOLD = 0.15

LANDMARK = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_MCP": 5,
    "INDEX_PIP": 6,
    "INDEX_DIP": 7,
    "INDEX_TIP": 8,
    "MIDDLE_MCP": 9,
    "MIDDLE_PIP": 10,
    "MIDDLE_DIP": 11,
    "MIDDLE_TIP": 12,
    "RING_MCP": 13,
    "RING_PIP": 14,
    "RING_DIP": 15,
    "RING_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

COLOR_PALETTE = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 128, 0),
]

SAVE_DIR = "saved_artworks"
