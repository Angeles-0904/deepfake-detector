@echo off
title DeepFake Detector - App
cd /d "D:\IX\detectorIA"

echo ============================================
echo    🚀 DeepFake Detector - Iniciando App
echo ============================================
echo.
echo  Modelo: outputs\checkpoints\best_model.pth
echo  Puerto: http://localhost:8501
echo.
echo  Presiona Ctrl+C para cerrar la app
echo ============================================
echo.

streamlit run app/streamlit_app.py --server.port 8501

pause
