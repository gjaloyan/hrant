@echo off
REM Run backend (uvicorn) and frontend (vite) together.
REM Usage: dev.bat
setlocal
set ROOT=%~dp0
"%ROOT%.venv\Scripts\python.exe" "%ROOT%scripts\dev.py" %*
exit /b %ERRORLEVEL%
