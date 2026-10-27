@echo off
REM Build on Windows with Python 3.10 or 3.11 (python.org, tick "Add to PATH")
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
pyinstaller --noconfirm --clean --windowed --name AutoRetouch ^
  --collect-all mediapipe --collect-all customtkinter ^
  auto_retouch.py
echo.
echo Done. Output: dist\AutoRetouch\AutoRetouch.exe
pause
