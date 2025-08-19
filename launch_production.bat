@echo off
setlocal
REM [DEPRECATED NOTE]
REM This file is a thin wrapper. The canonical orchestrator is scripts\run_all.ps1
REM Use this for production/preview startup on Windows.

powershell -ExecutionPolicy Bypass -NoLogo -NoProfile -File "%~dp0scripts\run_all.ps1" -Mode prod -OpenBrowser -KeepAlive %*
endlocal

