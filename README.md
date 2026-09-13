# ARt - Gesture Controlled AR Drawing System

Hand tracking and landmark detection system for drawing and interacting with virtual objects using hand gestures.

## Features

- Real-time hand detection and 2D landmark tracking (21 landmarks per hand)
- Gesture recognition (pinch, fist, point, open palm, victory)
- Hand gesture drawing on a virtual canvas
- Color palette cycling
- Pinch-to-draw interaction model

## Requirements

- Python 3.10+
- Webcam
- See `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

The hand landmarker model (`models/hand_landmarker.task`) is required and should be placed in the `models/` directory.

## Usage

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| Pinch (thumb + index close) | Draw on canvas |
| `n` | Next color |
| `c` | Clear canvas |
| `q` | Quit |

## Performance

- **MediaPipe processes at 320x240** instead of full display resolution (1280x720)
- **Landmark drawing skipped** - only landmark data is used (visualization removed from hot path)
- **Minimal overlay text** - reduced font rendering overhead
- **Low buffer size** - camera buffer set to 1 for lowest latency

Expected FPS: 20-30+ on most hardware (varies by camera and CPU).

## Architecture

```
ARt/
├── config.py              # Constants (dimensions, colors, landmark indices, thresholds)
├── main.py                # Entry point; ties camera, tracker, gestures, and canvas together
├── handtracking/
│   ├── __init__.py
│   ├── tracker.py         # Hand detection via MediaPipe, landmark extraction
│   └── gestures.py        # Gesture classification (pinch, fist, point, open palm, victory)
├── drawing/
│   ├── __init__.py
│   └── canvas.py          # Canvas state management (drawing, colors, clearing)
├── models/
│   └── hand_landmarker.task # Pre-trained hand landmark model
├── README.md
└── requirements.txt
```

## Landmark Reference

The system tracks 21 hand landmarks per hand (MediaPipe Hand Landmarks):

- 1 wrist
- 4 thumb landmarks
- 4 index finger landmarks
- 4 middle finger landmarks
- 4 ring finger landmarks
- 4 pinky finger landmarks

All coordinates are normalized (0.0 - 1.0) relative to the camera frame.

## Gesture Definitions

- **Pinch**: Thumb tip and index tip are close (distance < 0.04). Used for drawing.
- **Fist**: No fingers extended.
- **Point**: Only index finger extended.
- **Open Palm**: 4+ fingers extended.
- **Victory**: 2 fingers extended (index + middle).
