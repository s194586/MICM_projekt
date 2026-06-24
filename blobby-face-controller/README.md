# Blobby Face Controller - YuNet Fast Version

## What it is

One-person low-latency face controller for Blobby Online.
It uses OpenCV YuNet for face detection plus a lightweight landmark model on the face crop.

YuNet gives only 5 landmarks, so mouth opening is detected using an additional lightweight landmark model on the face crop.

## Controls

- face left or right = `A` / `D` hold
- mouth open = `W` hold
- face or head down = `Space` tap bonus
- `q` = quit
- `c` = recalibrate closed mouth
- `[` = lower jump threshold
- `]` = raise jump threshold
- `o` = overlay toggle

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

At startup keep a neutral face with closed mouth for about 1 second.

## Debug landmarks

```powershell
python controller.py --debug-landmarks
```

## Alternate jump modes

```powershell
python controller.py --jump-mode mouth_landmarks
python controller.py --jump-mode calibrated_smile
python controller.py --jump-mode mouth_open
python controller.py --jump-mode smile_or_mouth_open
python controller.py --jump-mode vertical_head_up
python controller.py --jump-enter 0.30 --jump-exit 0.18
```

`vertical_head_up` is only an experimental fallback and is not recommended for gameplay.

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

- press `c` with a closed mouth to recalibrate the neutral baseline
- if jump is too hard to trigger, press `[`
- if jump triggers by itself, press `]`
- if movement drifts, recalibrate with a neutral pose
- if the game is not reacting, click the browser game window
- if you want a fallback, try `python controller.py --jump-mode calibrated_smile`
- if you want the older ROI fallback, try `python controller.py --jump-mode mouth_open`

## Notes

- default jump mode is `mouth_landmarks`
- face detection stays on OpenCV YuNet ONNX
- mouth opening is measured from real mouth landmarks, not head-up
- the landmark model is expected at `models/pfld_68_face_landmarks.onnx`
- if the landmark model is missing, the controller prints a clear path to place it
