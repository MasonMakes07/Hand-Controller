# Hand Controller

Control PC games using hand gestures detected through your webcam. Uses Google's MediaPipe to track hand landmarks in real time and maps gestures to keyboard input with one hand, and to mouse movement/clicks/scroll with the other, no controller required.

---
## How It Works

Your webcam feed is processed by MediaPipe's hand landmark model, which tracks 21 points per hand, on up to two hands at once. One hand (configurable) is your **keyboard hand**: its landmarks are normalized and fed into a small trained classifier that recognizes 10 static gestures and presses/releases the mapped key. The other hand is your **mouse hand**: its movement is tracked like a trackpad, and pinches/finger poses drive clicks, drags, and scrolling.

If no trained classifier model exists yet, the keyboard hand falls back to the original hardcoded threshold-based detection, so the app works out of the box on a fresh clone, see [Training Your Own Gestures](#training-your-own-gestures) to replace it with a model trained on your own hand.

---

## Gesture Map

### Keyboard hand

| Gesture | Key |
|---|---|
| ☝️ Pointing (index up, thumb in) | `W` |
| ✌️ Peace Sign | `A` |
| 🤟 Three Fingers | `D` |
| 🖐 Four Fingers | `S` |
| 🤚 Open Hand | `Space` |
| ✊ Fist | `Z` |
| 👍 Thumbs Up | `E` |
| 🤙 Finger Gun | `R` |
| 👌 OK Sign | `Q` |
| 🖕 Middle Finger | `F` |

> Keybinds can be changed in `KeyBinds.py`. The gesture label list itself lives in `config.py` (`KEYBOARD_GESTURE_LABELS`).

### Mouse hand

| Gesture | Action |
|---|---|
| ✊ Closed fist | Pause tracking (like lifting your finger off a trackpad) |
| 🤚 Open hand, moving | Move cursor (relative movement, like a trackpad) |
| 🤏 Thumb + index pinch (tap / hold) | Left click / drag |
| 🤏 Thumb + middle pinch | Right click |
| ✌️ Index + middle extended, moving up/down | Scroll |

---

## Requirements

- Python 3.9+
- Webcam
- Windows (tested on Windows 11 — mouse control uses `ctypes.windll` for screen bounds)

---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/your-username/hand-controller.git
cd hand-controller
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the MediaPipe model**

The model downloads automatically on first run. If you want to download it manually:
```bash
curl -o handcontroller/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

---

## Usage

```bash
python handcontroller/hand_controller.py
```

- A window will open showing your webcam feed with landmark overlay for both tracked hands
- The keyboard hand's active gesture and the mouse hand's current state are shown in the top left
- Press `P` to quit
- Pass `--legacy` to force the original threshold-based keyboard detection even if a trained model exists
- Pass `--debug-timing` to show live FPS and MediaPipe inference latency on-screen and in the console

> Make sure your game window is focused before gesturing so keypresses/mouse events are received by the game.

---

## Handedness Calibration

MediaPipe reports each detected hand as `"Left"` or `"Right"` based on real-world handedness. Because the webcam frame is mirrored before detection (so the preview feels like a mirror), which physical hand ends up labeled which way can vary by setup, so it's a one-time calibration step rather than a hardcoded assumption:

1. Run `hand_controller.py` and raise each hand one at a time.
2. Note which label (`Left`/`Right`) the overlay prints next to each physical hand.
3. Set `KEYBOARD_HAND_LABEL` / `MOUSE_HAND_LABEL` in `config.py` to match which hand you want doing which job.

---

## Training Your Own Gestures

The keyboard hand's gestures are recognized by a small classifier trained on **your** hand, camera, and lighting — not shipped with the repo (see `.gitignore`).

**1. Capture training data**
```bash
python handcontroller/capture_gestures.py --role keyboard
```
A window opens listing each gesture label and the key that selects it. Press the label's key, then `SPACE` to start/stop recording samples while you hold that gesture (move your hand around a bit — different angles/distances help). Include the `none` label for "hand present, no deliberate gesture." Samples are saved to `handcontroller/data/gestures_keyboard.csv`. Press `q` when done.

**2. Train the classifier**
```bash
python handcontroller/train_gesture_model.py --data handcontroller/data/gestures_keyboard.csv
```
Prints a classification report on a held-out test split so you can sanity-check per-gesture accuracy. Saves the trained model to `handcontroller/models/gesture_classifier_keyboard.joblib`.

**3. Run normally**

`hand_controller.py` automatically uses the trained model if it's present at that path, falling back to legacy threshold detection otherwise.

> The same tools support `--role mouse` if you want to experiment with training a learned model for the mouse hand's poses — the mouse hand currently uses simple geometric checks (`gesture_features.py`), not a trained classifier, by default.

---

## Calibrate Mouse Range

If the cursor can't reach every edge of the screen (common cause: your camera frame's aspect ratio doesn't give you equal comfortable movement room in both directions), calibrate your own usable range once:

```bash
python handcontroller/calibrate_mouse.py
```

Follow the 4 on-screen prompts — move your mouse hand to your comfortable leftmost, rightmost, topmost, and bottommost positions, pressing `SPACE` at each (press `r` to redo the previous stage, `q` to cancel). This saves `handcontroller/data/mouse_calibration.json`, and `hand_controller.py` picks it up automatically on the next run — moving through that same range now maps to the full screen in each direction. Delete that file to fall back to the default `MOUSE_SENSITIVITY`-based behavior.

---

## Tuning

Tunable values live in `handcontroller/config.py`:

```python
# Legacy threshold-based gesture detection
THRESHOLD        = 0.02  # how far fingers must move to register as up/down
THUMB_THRESHOLD  = 0.06  # how far thumb must be from index base to count as "out"

# Keyboard-gesture debounce (asymmetric on purpose: fast press, guarded release)
REQUIRED_FRAMES_ON   = 1  # frames a gesture must be seen before its key presses
REQUIRED_FRAMES_OFF  = 2  # frames a gesture must be absent before its key releases

# Frame resolution MediaPipe runs detection on (the preview window stays full-res)
DETECTION_SIZE = (480, 360)  # set to None to disable downscaling

# Learned classifier
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7  # minimum predicted-class probability to trigger a key

# Mouse control
MOUSE_SENSITIVITY     = 4.0   # fallback sensitivity when no calibration exists (see calibrate_mouse.py)
MOUSE_CALIBRATED_GAIN_X = 1.5 # extra gain on top of your calibrated range, X axis
MOUSE_CALIBRATED_GAIN_Y = 2.0 # extra gain on top of your calibrated range, Y axis (raise if top/bottom falls short)
MOUSE_ACCEL_GAIN    = 0.0   # 0 = linear; >0 adds speed-based acceleration
PINCH_CLOSE_RATIO   = 0.35  # pinch distance (normalized) below which a pinch registers
PINCH_OPEN_RATIO    = 0.5   # must widen past this to count as released (hysteresis, avoids flicker)
SCROLL_SENSITIVITY  = 15.0
```

Run with `--debug-timing` to see live FPS and MediaPipe inference latency on-screen and printed once per second in the console, so it's useful when tuning `DETECTION_SIZE` or judging whether the pipeline is actually the bottleneck versus, say, your webcam's own buffering.

**Tips for best detection:**
- Keep your hand 30–60cm from the camera
- Light your hand from the front — avoid backlighting
- Keep your whole hand in frame

---

## File Structure

```
hand-controller/
├── requirements.txt              # Python dependencies
└── handcontroller/
    ├── hand_controller.py        # Main script — two-hand tracking, keyboard + mouse pipelines
    ├── config.py                 # Tunables, gesture label lists, handedness role mapping
    ├── gesture_features.py       # Landmark normalization + feature extraction (shared by all scripts)
    ├── tracker.py                 # Shared MediaPipe HandLandmarker setup
    ├── debounce.py                # Frame-counter hysteresis helper
    ├── legacy_gestures.py        # Original hardcoded threshold gesture detection (fallback)
    ├── mouse_control.py          # Relative mouse movement + pinch click/drag/scroll
    ├── mouse_calibration.py      # Load/save the calibrated mouse range
    ├── calibrate_mouse.py        # Interactive mouse range calibration tool
    ├── camera.py                  # Threaded webcam grabber (avoids stale buffered frames)
    ├── KeyBinds.py                # Maps gesture names to keyboard keys
    ├── capture_gestures.py       # Data capture tool for training your own gesture classifier
    ├── train_gesture_model.py    # Trains a classifier from captured data
    ├── hand_landmarker.task       # MediaPipe model (auto-downloaded on first run)
    ├── data/                      # Captured gesture datasets + mouse calibration (gitignored)
    └── models/                    # Trained classifier models (gitignored)
```

---

## Dependencies

```
mediapipe
opencv-python
pynput
scikit-learn
joblib
numpy
```
