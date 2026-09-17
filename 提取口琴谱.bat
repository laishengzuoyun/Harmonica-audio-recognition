@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "INPUT=%~1"
rem The dragged input is received as "%~1".
setlocal EnableDelayedExpansion
cd /d "!SCRIPT_DIR!"

if not defined INPUT goto no_input

set "VENV_ROOT=!SCRIPT_DIR!.tools\audio-venv"
set "VENV_PY=!VENV_ROOT!\Scripts\python.exe"
set "DEPS_READY=!VENV_ROOT!\.audio-deps-ready"

if exist "!VENV_PY!" goto check_dependencies
echo 正在创建项目本地 Python 3.12 环境，请稍候...
py -3.12 -m venv "!VENV_ROOT!"
if errorlevel 1 goto python_error

:check_dependencies
if exist "!DEPS_READY!" goto run_cli
echo 正在安装音频依赖，首次安装可能需要一些时间...
"!VENV_PY!" -m pip install --upgrade pip
if errorlevel 1 goto dependency_error
"!VENV_PY!" -m pip install -r "!SCRIPT_DIR!requirements-audio.txt"
if errorlevel 1 goto dependency_error
>"!DEPS_READY!" echo ready

:run_cli
echo 正在提取口琴谱。Demucs 首次运行可能需要联网下载模型，请耐心等待。
setlocal DisableDelayedExpansion
"%VENV_PY%" "%SCRIPT_DIR%scripts\extract_harmonica_score.py" "%INPUT%" --output "%SCRIPT_DIR%output" --keep-stems
set "CLI_CODE=%ERRORLEVEL%"
endlocal & set "CLI_CODE=%CLI_CODE%"
if not "%CLI_CODE%"=="0" goto cli_error

start "" explorer.exe "!SCRIPT_DIR!output"
echo 提取完成，结果已保存到："!SCRIPT_DIR!output"
pause
goto success

:no_input
echo 请将 MP3、WAV 或 FLAC 音频文件拖到本文件上运行。
pause
endlocal & endlocal & exit /b 2

:python_error
echo 创建 Python 环境失败。请确认已安装 Python 3.12，并可在命令行使用 py -3.12。
pause
endlocal & endlocal & exit /b 3

:dependency_error
echo 音频依赖安装失败。请检查网络连接或依赖配置后重试。
pause
endlocal & endlocal & exit /b 4

:cli_error
set "EXIT_CODE=!ERRORLEVEL!"
echo 提取失败，错误代码 %EXIT_CODE%。
pause
endlocal & endlocal & exit /b %EXIT_CODE%

:success
endlocal & endlocal & exit /b 0
