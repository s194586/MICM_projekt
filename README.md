# Blobby Face Controller

Low-latency computer vision controller for Blobby Online. The project uses OpenCV YuNet for face detection and a lightweight PFLD 68-point landmark model for mouth-opening jump control.

## Features

- Real-time face-based game control
- One-player control mode
- OpenCV YuNet face detection
- PFLD 68-point facial landmarks
- Mouth-opening jump detection
- Low-latency keyboard output using Win32 SendInput
- Pynput fallback
- Simple overlay with FPS, face status, movement, jump and last key event

## Controls

- Move face/head left: hold `A`
- Move face/head right: hold `D`
- Open mouth: hold `W`
- Close mouth: release `W`
- `q`: quit
- `c`: recalibrate neutral face / closed mouth

## How It Works

1. Camera frame is captured in low-latency mode.
2. YuNet detects the face.
3. The face crop is passed to the PFLD landmark model.
4. Mouth aspect ratio is calculated from mouth landmarks.
5. Keyboard state is updated only when control state changes.

`MAR = (d(61,67) + d(62,66) + d(63,65)) / (2 * d(60,64))`

## Installation

Windows / Python 3.11:

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

## Recommended Usage

1. Start the controller.
2. Keep a neutral face and closed mouth during initial calibration.
3. Click the Blobby browser window during the countdown.
4. Play.

## Options

```powershell
python controller.py --keyboard pynput
python controller.py --left-key left --right-key right --jump-key up
python controller.py --focus-delay 0
```

## Troubleshooting

- If Blobby does not react, click the game window again.
- If Win32 input fails, run with `--keyboard pynput`.
- If jump triggers too easily or too rarely, press `c` and recalibrate with a closed mouth.
- If detection is unstable, improve lighting and keep your face centered.

## Performance

The current pipeline is optimized for low latency and processes only one face. You can run the lightweight benchmark with:

```powershell
cd blobby-face-controller
python benchmark.py
```

## Project Status

Final simplified gameplay version.

## Tech Stack

- Python
- OpenCV
- NumPy
- ONNX Runtime
- YuNet
- PFLD
- pynput fallback
- Win32 SendInput

## Notes

- This is an experimental HCI / game-control project.
- It does not use MediaPipe in the final version.
- It is designed for Windows.
- The ONNX models are included in the repo so the project runs after clone.
