@echo off
chcp 65001 >nul
echo.
echo  VisualFind - Image-based Product Search
echo  ========================================
echo.
echo  Starting FastAPI backend on http://localhost:8000
echo  API docs:  http://localhost:8000/docs
echo.
echo  Open the frontend:  http://localhost:8080
echo  (start_frontend.bat must also be running)
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
