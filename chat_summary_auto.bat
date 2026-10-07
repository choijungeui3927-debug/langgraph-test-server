@echo off
rem Daily chat summary (registered in Windows Task Scheduler).
rem Summarizes market dates that had new conversations since the last run.
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist logs mkdir logs
"%USERPROFILE%\.local\bin\uv.exe" run python -m src.chat_summary --auto >> logs\chat_summary_auto.log 2>&1
