# Blobby Face Controller - YuNet Fast Version

## What it is

One-person low-latency face controller for Blobby Online.
It uses OpenCV YuNet for face detection plus a lightweight landmark model on the face crop.

YuNet gives only 5 landmarks, so mouth opening is detected using an additional lightweight landmark model on the face crop.

## Controls

- face left or right = hold left/right movement
- mouth open = hold jump
- face or head down = `Space` tap bonus
- `q` = quit
- `c` = recalibrate closed mouth
- `[` = lower jump threshold
- `]` = raise jump threshold
- `o` = overlay toggle

`A` and `D` are never held together. Jump hold is independent from movement. `Space` taps without releasing movement or jump. If no face is detected, the controller releases all keys.

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

Then:

1. Wait for the overlay.
2. Click the Blobby window during the startup countdown.
3. Do not click the overlay again.
4. Play.

Default keyboard output is Win32 `SendInput` with scan codes.
Default mapping is `A` / `D` / `W` / `Space`.

## Keyboard fallback

If Blobby does not react to the default backend:

```powershell
python controller.py --keyboard pynput
```

## Arrow key mapping

If Blobby uses arrows instead of `A` / `D` / `W`:

```powershell
python controller.py --left-key left --right-key right --jump-key up --bonus-key space
```

## Useful options

```powershell
python controller.py --focus-delay 0
python controller.py --jump-enter 0.30 --jump-exit 0.18
python controller.py --debug-landmarks
```

## Alternate jump modes

```powershell
python controller.py --jump-mode mouth_landmarks
python controller.py --jump-mode calibrated_smile
python controller.py --jump-mode mouth_open
python controller.py --jump-mode smile_or_mouth_open
python controller.py --jump-mode vertical_head_up
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
- if the game is not reacting, restart and click the browser game window during the countdown
- if the default backend still does not work, try `python controller.py --keyboard pynput`

## Notes

- default jump mode is `mouth_landmarks`
- default keyboard backend is `win32`
- face detection stays on OpenCV YuNet ONNX
- mouth opening is measured from real mouth landmarks, not head-up
- the landmark model is expected at `models/pfld_68_face_landmarks.onnx`
- if the landmark model is missing, the controller prints a clear path to place it
