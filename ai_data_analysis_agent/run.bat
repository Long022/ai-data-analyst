@echo off
set NO_PROXY=*
set no_proxy=*
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=

echo ================================================
echo   AI Data Analysis Agent (DeepSeek V4 Pro)
echo ================================================
echo.
echo  Configure: copy .env.example to .env and edit
echo.

"%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run "%~dp0ai_data_analyst.py"
pause
