@echo off
rem Build the news briefing and save it to the data folder.
rem Usage: digest.bat              (today, KST)
rem        digest.bat 2026-09-29   (specific date)
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
uv run python -m src.run %1
pause
