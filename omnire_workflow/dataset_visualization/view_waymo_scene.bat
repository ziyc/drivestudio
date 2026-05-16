@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"

if "%VIZ_ENV_NAME%"=="" set "VIZ_ENV_NAME=waymo-kitti-viz"
if "%SCENE_IDX%"=="" set "SCENE_IDX=114"
if "%PROJECT_OUTPUT%"=="" set "PROJECT_OUTPUT=OutPut\waymo_training_10scenes"
if "%CAMERA_ID%"=="" set "CAMERA_ID=0"
if "%CLASSES%"=="" set "CLASSES=Car"
if "%START_FRAME%"=="" set "START_FRAME=0"
if "%END_FRAME%"=="" set "END_FRAME=-1"

call conda run -n "%VIZ_ENV_NAME%" python -m omnire_workflow.dataset_visualization.cli view-scene ^
    --scene_idx "%SCENE_IDX%" ^
    --project_output "%PROJECT_OUTPUT%" ^
    --camera_id "%CAMERA_ID%" ^
    --classes "%CLASSES%" ^
    --start_frame "%START_FRAME%" ^
    --end_frame "%END_FRAME%"
if errorlevel 1 exit /b 1

popd
endlocal
