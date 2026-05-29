@echo off
REM Launcher that sets the Isaac-Sim env vars then runs a Python script with the
REM Isaac-Lab conda env's Python. Works from BOTH cmd.exe and PowerShell, and
REM avoids the PowerShell script-execution-policy prompt that blocks .ps1 files.
REM
REM Usage (single line):
REM   scripts\run_isaaclab.bat scripts\isaaclab_visualize.py --task Isaac-Velocity-Rough-G1-v0 --num_envs 2 --method gatsr_full --train_steps 512 --run_steps 3000
REM
REM Override paths by setting GATSR_ISAACSIM_PATH / GATSR_CONDA_PY before calling.

setlocal EnableExtensions

if "%GATSR_ISAACSIM_PATH%"=="" (
    set "ISAACSIM=C:\isaac-sim"
) else (
    set "ISAACSIM=%GATSR_ISAACSIM_PATH%"
)
if "%GATSR_CONDA_PY%"=="" (
    set "CONDA_PY=C:\Users\reese\miniconda3\envs\isaaclab\python.exe"
) else (
    set "CONDA_PY=%GATSR_CONDA_PY%"
)

if not exist "%ISAACSIM%" (
    echo [ERROR] Isaac Sim not found at "%ISAACSIM%". Set GATSR_ISAACSIM_PATH.
    exit /b 1
)
if not exist "%CONDA_PY%" (
    echo [ERROR] Conda env python not found at "%CONDA_PY%". Set GATSR_CONDA_PY.
    exit /b 1
)

set "CARB_APP_PATH=%ISAACSIM%\kit"
set "EXP_PATH=%ISAACSIM%\apps"
set "ISAAC_PATH=%ISAACSIM%"
set "RESOURCE_NAME=IsaacSim"
if "%PYTHONPATH%"=="" (
    set "PYTHONPATH=%ISAACSIM%\site"
) else (
    set "PYTHONPATH=%PYTHONPATH%;%ISAACSIM%\site"
)

echo [run] Isaac Sim:  %ISAACSIM%
echo [run] Python:     %CONDA_PY%
echo [run] Args:       %*
echo.

"%CONDA_PY%" %*
exit /b %ERRORLEVEL%
