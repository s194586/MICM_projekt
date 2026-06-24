# Blobby Face Controller - YuNet Fast Version

## What it is

One-person low-latency face controller for Blobby Online.
It uses OpenCV YuNet through `cv2.FaceDetectorYN`, not MediaPipe.

## Controls

- move head or face left-right = `A` / `D` hold
- calibrated smile = `W` hold jump
- move face or head down = `Space` tap bonus
- `c` = neutral calibration
- `m` = smile calibration
- `[` = lower jump sensitivity threshold
- `]` = raise jump sensitivity threshold
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

At startup keep a normal neutral face for about 1 second.
Then press `m` and smile as wide as possible for about 1 second to calibrate jump.

## Fast run

```powershell
python controller.py --width 320 --height 240 --no-overlay
```

## Alternate jump modes

```powershell
python controller.py --jump-mode calibrated_smile
python controller.py --jump-mode smile_or_mouth_open
python controller.py --jump-mode mouth_open
python controller.py --jump-enter 0.45 --jump-exit 0.30
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

- press `c` to recalibrate neutral pose
- press `m` to recalibrate the smile baseline
- if jump is too hard to trigger, press `[`
- if jump triggers by itself, press `]`
- if movement drifts, recalibrate with a neutral pose
- if you want a fallback, try `python controller.py --jump-mode smile_or_mouth_open`
- if you want the older ROI fallback only, try `python controller.py --jump-mode mouth_open`
- if the game is not reacting, click the browser game window

## Notes

- default jump mode is `calibrated_smile`
- smile score uses YuNet mouth corners and eye landmarks for normalization
- `mouth_open` remains available as a fallback mode
- the YuNet ONNX model is expected at `models/face_detection_yunet_2023mar.onnx`
