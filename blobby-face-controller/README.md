# Blobby Face Controller - YuNet Fast Version

## What it is

One-person low-latency face controller for Blobby Online.
It uses OpenCV YuNet through `cv2.FaceDetectorYN`, not MediaPipe.

## Controls

- move head or face left-right = `A` / `D` hold
- open mouth = `W` hold jump
- move face or head down = `Space` tap bonus
- `c` = recalibrate
- `q` = quit

`A` and `D` are never held together. `W` is independent from movement. `Space` taps without releasing `A`, `D`, or `W`. If no face is detected, the controller releases all keys.

## Setup

```powershell
py -3.11 -m venv "MICM projekt"
"MICM projekt\Scripts\activate"
cd blobby-face-controller
pip install -r requirements.txt
```

## Run

```powershell
python controller.py
```

## Fast run

```powershell
python controller.py --width 320 --height 240 --no-overlay
```

## If camera does not open

```powershell
python controller.py --camera-index 1
python controller.py --camera-backend msmf
python controller.py --camera-backend default
```

## Benchmark

```powershell
python benchmark.py
```

## Troubleshooting

- press `c` to recalibrate
- if movement drifts, recalibrate with a neutral pose
- if jump is too hard or too easy, tune `MOUTH_OPEN_ENTER_THRESHOLD` and `MOUTH_OPEN_EXIT_THRESHOLD` in `controller.py`
- if the game is not reacting, click the browser game window

## Notes

- default jump mode is `mouth_open`
- optional fallback remains available through `--jump-mode vertical_head`
- the YuNet ONNX model is expected at `models/face_detection_yunet_2023mar.onnx`
