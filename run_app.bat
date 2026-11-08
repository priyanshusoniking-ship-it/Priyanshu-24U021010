@echo off
title FeedTransformer AI - Personalized News Feed
echo =======================================================
echo Starting FeedTransformer AI Streamlit Dashboard...
echo =======================================================

REM Try activating virtual environment if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Launch Streamlit
streamlit run app.py

pause
