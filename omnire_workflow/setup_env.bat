@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem DriveStudio / OmniRe environment bootstrap for Windows cmd.
rem Override before running if needed:
rem   set ENV_NAME=drivestudio
rem   set INSTALL_WAYMO=1

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if "%ENV_NAME%"=="" set "ENV_NAME=drivestudio"
if "%INSTALL_WAYMO%"=="" set "INSTALL_WAYMO=1"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found in PATH.
    echo Install Miniconda/Anaconda first, then rerun this script.
    exit /b 1
)

echo [1/6] Initializing git submodules
git submodule update --init --recursive
if errorlevel 1 exit /b 1

echo [2/6] Creating or updating conda env: %ENV_NAME%
conda env list | findstr /R /C:"^%ENV_NAME% " >nul 2>nul
if errorlevel 1 (
    conda create -n "%ENV_NAME%" python=3.9 -y
    if errorlevel 1 exit /b 1
) else (
    echo Conda env already exists: %ENV_NAME%
)

echo [3/6] Installing Python requirements
call conda run -n "%ENV_NAME%" python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
call conda run -n "%ENV_NAME%" pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [4/6] Installing DriveStudio rendering dependencies
call conda run -n "%ENV_NAME%" pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
if errorlevel 1 exit /b 1
call conda run -n "%ENV_NAME%" pip install git+https://github.com/facebookresearch/pytorch3d.git
if errorlevel 1 exit /b 1
call conda run -n "%ENV_NAME%" pip install git+https://github.com/NVlabs/nvdiffrast
if errorlevel 1 exit /b 1

echo [5/6] Installing local SMPL-X package
call conda run -n "%ENV_NAME%" pip install -e third_party\smplx
if errorlevel 1 exit /b 1

if "%INSTALL_WAYMO%"=="1" (
    echo [6/6] Installing Waymo Open Dataset toolkit
    call conda run -n "%ENV_NAME%" pip install waymo-open-dataset-tf-2-11-0==1.6.0
    if errorlevel 1 exit /b 1
) else (
    echo [6/6] Skipping Waymo toolkit because INSTALL_WAYMO=%INSTALL_WAYMO%
)

echo.
echo Environment setup completed.
echo Activate it with: conda activate %ENV_NAME%
popd
endlocal
