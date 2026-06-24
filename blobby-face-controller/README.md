# Blobby Face Controller

Final one-player controller for Blobby Online.
It uses OpenCV YuNet for face detection and PFLD mouth landmarks for jump.

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

After start:

1. Keep your mouth closed for about 1 second.
2. Click the Blobby window during the startup countdown.
3. Play.

## Controls

- face left/right = `A` / `D` hold
- open mouth = `W` hold
- `c` = recalibrate closed mouth
- `q` = quit
- `o` = overlay toggle

`A` and `D` are never held together. Open mouth holds jump. If no face is detected, the controller releases all keys.

## Fallbacks

If the default input backend does not work:

```powershell
python controller.py --keyboard pynput
```

If the game uses arrow keys:

```powershell
python controller.py --left-key left --right-key right --jump-key up
```

## Notes

- default keyboard backend is Win32 `SendInput` scan codes
- default resolution is `424x240 @ 60 fps`
- the controller uses one fixed jump path: mouth landmarks to `W` hold
- required models:
  - `models/face_detection_yunet_2023mar.onnx`
  - `models/pfld_68_face_landmarks.onnx`
