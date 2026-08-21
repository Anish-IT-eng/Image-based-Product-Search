@echo off
chcp 65001 >nul
echo.
echo  VisualFind - Building Search Index
echo  ====================================
echo.
echo  This will:
echo    1. Sample 1,600 product images from the catalog
echo    2. Extract ResNet50 embeddings (CPU, ~5-10 minutes)
echo    3. Build and save the FAISS index to backend\data\
echo.
echo  Progress will appear below...
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python backend\build_index.py

echo.
echo  Done! You can now start the server with start_server.bat
echo.
pause
