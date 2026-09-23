# ARt - Gesture Controlled AR Drawing System

A real-time hand-tracking and 2D landmark detection system built with OpenCV and MediaPipe for virtual drawing, shape selection, and partial canvas transformations using intuitive hand gestures.

## Features

- **Multi-Hand Real-Time Tracking**: Tracks up to 2 hands simultaneously with mirror-aware handedness detection.
- **Left-Hand Circle Selection**: Draw a loop/circle with your left index finger around any part of the canvas to select specific strokes.
- **Partial Transformations**: Move or scale only selected strokes independently from the rest of the canvas.
- **Left-Hand Pinch-to-Move**: Pinch with your left hand to translate (pan) selected drawings or the entire canvas.
- **Right-Hand Pinch-to-Scale**: Pinch with your right hand to zoom/scale selected drawings or the entire canvas.
- **Point-to-Draw & Erase**: Draw using your right index finger; erase using index + middle finger.
- **Fist-to-Deselect**: Form a fist with your left hand to instantly deselect current selection.
- **Color Cycling**: Cycle through an 8-color palette.

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

## Gesture Controls & Functions

### Left Hand Gestures (Selection & Movement)

| Gesture | Hand | Action / Function |
|---------|------|-------------------|
| **Point** (Index Finger) | Left | Traces a **Selection Circle/Lasso** around strokes on the canvas. Releasing the point closes the loop and selects enclosed drawings. |
| **Pinch** (Thumb + Index) | Left | **Moves / Pans** the selected drawing around. If no selection is active, pans the entire canvas. |
| **Fist** (Closed Hand) | Left | **Deselects** the current selection, returning all strokes to unselected state. |

### Right Hand Gestures (Drawing, Erasing & Scaling)

| Gesture | Hand | Action / Function |
|---------|------|-------------------|
| **Point** (Index Finger) | Right | **Draws** on the canvas using the currently active color. |
| **Erase** (Index + Middle) | Right | **Erases** strokes near the finger tip. |
| **Pinch** (Thumb + Index) | Right | **Scales / Zooms** the selected drawing. If no selection is active, zooms the entire canvas. |

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| `n` | Cycle to the next color in the palette |
| `c` | Clear the entire canvas and reset view transform |
| `d` | Deselect current selection |
| `q` | Quit the application |

## Architecture

```
ARt/
├── config.py                # Configuration constants (resolution, colors, thresholds, landmarks)
├── main.py                  # Entry point tying together camera, tracker, gestures, canvas & UI
├── handtracking/
│   ├── __init__.py
│   ├── tracker.py           # Multi-hand detection via MediaPipe, handedness & native OpenCV skeleton rendering
│   └── gestures.py          # Per-hand gesture recognition (pinch, fist, point, erase, open palm)
├── drawing/
│   ├── __init__.py
│   ├── canvas.py            # Canvas state, polygon selection testing, stroke rendering & transformation
│   └── models.py            # Transform model (scale, tx, ty) & Stroke data model
├── models/
│   └── hand_landmarker.task # Pre-trained MediaPipe hand landmarker model
├── README.md
└── requirements.txt
```

## Landmark Reference

The system tracks 21 2D/3D hand landmarks per hand (MediaPipe Hand Landmarks):

- Landmark `0`: Wrist
- Landmarks `1-4`: Thumb (CMC, MCP, IP, Tip)
- Landmarks `5-8`: Index finger (MCP, PIP, DIP, Tip)
- Landmarks `9-12`: Middle finger (MCP, PIP, DIP, Tip)
- Landmarks `13-16`: Ring finger (MCP, PIP, DIP, Tip)
- Landmarks `17-20`: Pinky finger (MCP, PIP, DIP, Tip)
