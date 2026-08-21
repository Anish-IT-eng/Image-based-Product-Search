@echo off
chcp 65001 >nul
echo.
echo  VisualFind - Frontend Server
echo  ==============================
echo.
echo  Serving frontend at: http://localhost:8080
echo  Make sure the backend (start_server.bat) is also running.
echo.
echo  Open your browser at: http://localhost:8080
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m http.server 8080 --directory frontend
