@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo 请将 MP3、WAV 或 FLAC 音频文件拖到本文件上运行。
    pause
    exit /b 2
)

set "VENV_ROOT=%~dp0.tools\audio-venv"
set "VENV_PY=%VENV_ROOT%\Scripts\python.exe"
set "DEPS_READY=%VENV_ROOT%\.audio-deps-ready"

if not exist "%VENV_PY%" (
    echo 正在创建项目本地 Python 3.12 环境，请稍候...
    py -3.12 -m venv "%VENV_ROOT%"
    if errorlevel 1 (
        echo 创建 Python 环境失败。请确认已安装 Python 3.12，并可在命令行使用 py -3.12。
        pause
        exit /b 3
    )
)

if not exist "%DEPS_READY%" (
    echo 正在安装音频依赖，首次安装可能需要一些时间...
    "%VENV_PY%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo pip 更新失败。请检查网络连接后重试。
        pause
        exit /b 4
    )
    "%VENV_PY%" -m pip install -r "%~dp0requirements-audio.txt"
    if errorlevel 1 (
        echo 音频依赖安装失败。请检查网络连接或依赖配置后重试。
        pause
        exit /b 4
    )
    >"%DEPS_READY%" echo ready
)

echo 正在提取口琴谱。Demucs 首次运行可能需要联网下载模型，请耐心等待。
"%VENV_PY%" "%~dp0scripts\extract_harmonica_score.py" "%~1" --output "%~dp0output" --keep-stems
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo 提取失败，错误代码 %EXIT_CODE%。
    pause
    endlocal & exit /b %EXIT_CODE%
)

start "" explorer.exe "%~dp0output"
echo 提取完成，结果已保存到：%~dp0output
pause
endlocal & exit /b 0
