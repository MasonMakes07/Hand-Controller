# Hand Controller

Control PC games using hand gestures detected through your webcam. Uses Google's MediaPipe to track hand landmarks in real time and maps gestures to keyboard inputs — no controller required.

---
## How It Works

Your webcam feed is processed by MediaPipe's hand landmark model, which tracks 21 points on your hand 60+ times per second. The positions of fingertips relative to knuckles are compared to determine which gesture you're making, and the corresponding key is pressed or released instantly.

---

## Gesture Map

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

> Keybinds can be changed in `KeyBinds.py`

---

## Requirements

- Python 3.9+
- Webcam
- Windows (tested on Windows 11)

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
curl -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

---

## Usage

```bash
python hand_controller.py
```

- A window will open showing your webcam feed with landmark overlay
- The current detected gesture is shown in the top left
- Press `P` to quit

> Make sure your game window is focused before gesturing so keypresses are received by the game.

---

## Tuning

If gestures aren't detecting well, adjust these values at the top of `hand_controller.py`:

```python
THRESHOLD       = 0.02  # how far fingers must move to register as up/down
                        # raise if false positives, lower if gestures won't trigger

THUMB_THRESHOLD = 0.06  # how far thumb must be from index base to count as "out"
                        # watch the "thumb dist" readout in the camera window to tune this

REQUIRED_FRAMES = 2     # frames a gesture must be held before triggering
                        # raise to reduce flickering, lower for faster response
```

**Tips for best detection:**
- Keep your hand 30–60cm from the camera
- Light your hand from the front — avoid backlighting
- Keep your whole hand in frame

---

## File Structure

```
hand-controller/
├── hand_controller.py   # Main script — gesture detection + key sending
├── KeyBinds.py          # Maps gestures to keyboard keys
├── requirements.txt     # Python dependencies
└── hand_landmarker.task # MediaPipe model (auto-downloaded on first run)
```

---

## Dependencies

```
mediapipe
opencv-python
pynput
```