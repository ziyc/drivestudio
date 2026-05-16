@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Environment for Waymo-to-KITTI conversion and the modified 3D viewer.
rem Override before running if needed:
rem   set VIZ_ENV_NAME=waymo-kitti-viz

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"

if "%VIZ_ENV_NAME%"=="" set "VIZ_ENV_NAME=waymo-kitti-viz"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found in PATH.
    exit /b 1
)

echo [1/4] Creating or reusing conda env: %VIZ_ENV_NAME%
conda env list | findstr /R /C:"^%VIZ_ENV_NAME% " >nul 2>nul
if errorlevel 1 (
    conda create -n "%VIZ_ENV_NAME%" python=3.8 -y
    if errorlevel 1 exit /b 1
) else (
    echo Conda env already exists: %VIZ_ENV_NAME%
)

echo [2/4] Installing converter dependencies
call conda run -n "%VIZ_ENV_NAME%" python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
call conda run -n "%VIZ_ENV_NAME%" pip install tensorflow==2.11.* waymo-open-dataset-tf-2-11-0==1.6.0 opencv-python tqdm matplotlib
if errorlevel 1 exit /b 1

echo [3/4] Installing viewer dependencies
call conda run -n "%VIZ_ENV_NAME%" pip install numpy==1.21.3 vedo==2021.0.6 vtk==9.0.3 opencv-python==4.5.4.58 matplotlib==3.4.3
if errorlevel 1 exit /b 1

echo [4/4] Done
echo Activate with: conda activate %VIZ_ENV_NAME%
popd
endlocal
