@echo off
setlocal
REM [DEPRECATED NOTE]
REM This file is a thin wrapper. The canonical orchestrator is scripts/run_all.ps1
REM Use this for development startup on Windows.

powershell -ExecutionPolicy Bypass -NoLogo -NoProfile -File "%~dp0run_all.ps1" -Mode dev -OpenBrowser -KeepAlive %*
endlocal
