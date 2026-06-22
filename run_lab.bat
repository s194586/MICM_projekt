@echo off
cd /d "%~dp0"
if exist "MICM projekt\Scripts\activate.bat" (
    call "MICM projekt\Scripts\activate.bat"
) else (
    echo [ERROR] Virtual environment "MICM projekt" not found.
    echo Create it first with:
    echo py -3.11 -m venv "MICM projekt"
    echo call "MICM projekt\Scripts\activate.bat"
    echo cd blobby-face-controller
    echo pip install -r requirements.txt
    pause
    exit /b 1
)
cd /d "%~dp0blobby-face-controller"
python realtime_controller.py
pause
