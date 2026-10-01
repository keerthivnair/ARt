# ARt - Gesture Controlled 3D Drawing System

A real-time hand-tracking drawing system built with OpenCV and MediaPipe. Drawing happens in **3D**: every stroke is a chain of 3D vertices, the user draws onto a **draw plane** placed a set distance in front of the view plane, and the screen shows a **perspective projection** of that 3D scene onto the 2D view plane. Orbiting, rotating, translating in depth, and vertex snapping are all available by gesture.

## Features

- **True 3D geometry**: strokes store `(x, y, z)` vertices placed in world space; the camera projects them with a perspective divide (`x_screen = cx + f·X/Z`).
- **Draw plane at a distance from the view plane**: a plane parallel to the view plane, `DRAW_PLANE_DISTANCE` units in front of the camera. Drawn vertices land on this plane, not on the screen. Its distance is adjustable live.
- **Draw plane guide**: an on-screen grid with X/Y axes and a `+Z` stub showing the plane floating in front of the view plane.
- **Depth cueing**: stroke thickness and brightness scale with distance from the camera, so depth reads naturally.
- **Rotate selection *or* the view plane**: one gesture rotates the current selection about the view axis; with no selection it rotates the whole view plane.
- **3D translation**: move a selection within the draw plane, and push/pull it along the view (Z) axis; orbit the camera with another gesture.
- **Vertex snapping**: a newly drawn vertex snaps to the nearest already-drawn vertex (voxel) within a configurable radius, using a spatial hash.
- **Multi-Hand Real-Time Tracking**: tracks up to 2 hands simultaneously with mirror-aware handedness detection.
- **Lasso selection**: draw a loop with your left index finger to select the enclosed strokes.
- **Color cycling** across an 8-color palette.

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Webcam

## Installation

```bash
pip install -r requirements.txt
```

The hand landmarker model (`models/hand_landmarker.task`) is required and included in the `models/` directory.

## Usage

```bash
python main.py
```

## Coordinate System

- World axes: **+X** right, **+Y** down, **+Z** into the scene (away from the camera).
- The **view plane** is at the camera; the **draw plane** is parallel to it, `DRAW_PLANE_DISTANCE` in front.
- The camera orbits a `target` point at `distance`, controlled by yaw / pitch / roll.

## Gesture Controls

### Right Hand (drawing & depth)

| Gesture | Action |
|---------|--------|
| **Point** (index) | **Draw** a 3D stroke on the draw plane, snapping to nearby voxels. |
| **Index + Middle** | **Erase** 3D vertices near the finger. |
| **Pinch** (thumb + index) | **Scale** the selection in 3D; with no selection, **dolly** the camera in/out. |
| **Index + Middle + Ring** | **Translate in Z**: push/pull the selection along the view axis; with no selection, **move the draw plane** closer/farther. |
| **Index + Pinky** | **Rotate** the selection (or the view plane if nothing is selected). |
| **Open Palm** | Toggle vertex snapping. |

### Left Hand (selection & view)

| Gesture | Action |
|---------|--------|
| **Point** (index) | Trace a **lasso** around strokes to select them. |
| **Pinch** (thumb + index) | **Move** the selection within the draw plane; with no selection, **pan** the view. |
| **Index + Middle** | **Snap** the selection's centroid onto the hand's draw-plane point. |
| **Index + Middle + Ring** | **Orbit** the view (yaw/pitch) around the scene. |
| **Index + Pinky** | **Rotate** the selection (or roll the view plane). |
| **Fist** | **Deselect** everything. |
| **Open Palm** | Reset the view to the default. |

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| `n` | Cycle to the next color |
| `c` | Clear the scene and reset the view |
| `d` | Deselect current selection |
| `v` | Toggle vertex snapping |
| `[` / `]` | Decrease / increase snap radius |
| `-` / `=` | Move the draw plane closer / farther |
| `r` | Reset the view |
| `g` | Toggle the draw plane guide |
| `s` | **Save** the current artwork |
| `l` | **Load** an artwork (replaces the scene) |
| `m` | **Merge** an artwork (adds to the scene) |
| `q` | Quit |

## Saving and Loading

`s` writes two files into `saved_artworks/`:
1. A `.png` of the rendered projection (shareable image).
2. A `.json` storing all 3D stroke vertices, per-stroke transforms, colors, thicknesses, and the full camera state (target, distance, draw distance, yaw, pitch, roll, focal).

`l` restores the scene and camera exactly. `m` adds the saved strokes to the current scene. Legacy 2D saves are still loadable (their `(x, y)` points are padded to 3D at `z = 0`).

## Architecture

```
ARt/
├── config.py                # Resolution, 3D/camera/draw-plane, snapping, gesture tuning
├── main.py                  # Entry point: camera → tracker → gestures → 3D canvas → HUD
├── handtracking/
│   ├── tracker.py           # Multi-hand MediaPipe detection + native OpenCV skeleton
│   └── gestures.py          # Gesture classification + hand roll angle
├── drawing/
│   ├── geometry.py          # 4x4 matrix math + perspective orbit Camera
│   ├── models.py            # Stroke3D: 3D vertices + per-stroke 4x4 transform
│   ├── scene.py             # Strokes, voxel spatial index, snapping, 3D selection ops
│   └── canvas.py            # Projection renderer, draw plane guide, overlays, save/load
├── models/
│   └── hand_landmarker.task # MediaPipe hand landmarker
├── verify_3d.py             # Headless 3D math & scene test suite
├── verify_pipeline.py       # Headless end-to-end gesture pipeline test suite
├── README.md
└── requirements.txt
```

## How 3D Editing Works

Each `Stroke3D` keeps its vertices in a local space and carries a 4×4 `transform` matrix. Selection edits never rewrite the drawn geometry; they only compose new matrices:

- **Translate** → `T(delta) · M`
- **Rotate** → `T(pivot) · R(axis, angle) · T(-pivot) · M`
- **Scale** → `T(pivot) · S(s) · T(-pivot) · M`

Rendering projects every world vertex through the camera and draws the polyline; segments behind the camera are clipped. Because selection edits are pure transforms, they compose cleanly and round-trip exactly through save/load.

## Verification

Two headless suites exercise the system without a camera or MediaPipe:

```bash
python verify_3d.py        # 78 checks: projection, draw plane, snapping, 3D selection math, save/load, gestures
python verify_pipeline.py  # 33 checks: end-to-end gesture → canvas pipeline
```
