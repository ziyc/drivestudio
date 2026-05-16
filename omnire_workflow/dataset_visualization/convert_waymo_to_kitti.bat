@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"

if "%VIZ_ENV_NAME%"=="" set "VIZ_ENV_NAME=waymo-kitti-viz"
if "%KITTI_OUT%"=="" set "KITTI_OUT=OutPut\waymo_training_10scenes\scene_114\dataset_visualization\kitti"
if "%NUM_PROC%"=="" set "NUM_PROC=1"
if "%PREFIX%"=="" set "PREFIX="
if "%CAMERA_ID%"=="" set "CAMERA_ID=0"

set "EXTRA_ARGS="
if "%CONVERT_ALL%"=="1" set "EXTRA_ARGS=--convert_all"
if not "%TFRECORD_FILE%"=="" set "EXTRA_ARGS=%EXTRA_ARGS% --tfrecord_file ""%TFRECORD_FILE%"""
if not "%RAW_DIR%"=="" set "EXTRA_ARGS=%EXTRA_ARGS% --raw_dir ""%RAW_DIR%"""

call conda run -n "%VIZ_ENV_NAME%" python -m omnire_workflow.dataset_visualization.cli convert-one ^
    --kitti_out "%KITTI_OUT%" ^
    --prefix "%PREFIX%" ^
    --num_proc "%NUM_PROC%" ^
    --camera_id "%CAMERA_ID%" ^
    %EXTRA_ARGS%
if errorlevel 1 exit /b 1

popd
endlocal
