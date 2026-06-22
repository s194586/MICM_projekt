@echo off
cd /d "%~dp0"
if not exist "MICM projekt\Scripts\activate.bat" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv "MICM projekt"
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
call "MICM projekt\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Could not activate the virtual environment.
    pause
    exit /b 1
)
cd /d "%~dp0blobby-face-controller"
python -m pip install --upgrade pip
if errorlevel 1 goto :install_error
pip install -r requirements.txt
if errorlevel 1 goto :install_error
python check_lab_ready.py
set "CHECK_EXIT=%ERRORLEVEL%"
pause
exit /b %CHECK_EXIT%

:install_error
echo [ERROR] Dependency installation failed.
pause
exit /b 1
