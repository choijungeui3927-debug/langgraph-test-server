@echo off
rem Start the LangGraph Studio server. Closing this window stops the server.
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
uv run langgraph dev
pause
