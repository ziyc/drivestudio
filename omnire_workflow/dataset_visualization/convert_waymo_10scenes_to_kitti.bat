@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"

if "%VIZ_ENV_NAME%"=="" set "VIZ_ENV_NAME=waymo-kitti-viz"
if "%RAW_DIR%"=="" set "RAW_DIR=data\waymo\raw"
if "%PROJECT_OUTPUT%"=="" set "PROJECT_OUTPUT=OutPut\waymo_training_10scenes"
if "%MAP_FILE%"=="" set "MAP_FILE=omnire_workflow\dataset_visualization\waymo_10scene_map.txt"
if "%NUM_PROC%"=="" set "NUM_PROC=1"
if "%PREFIX%"=="" set "PREFIX="
if "%CAMERA_ID%"=="" set "CAMERA_ID=0"

set "EXTRA_ARGS="
if "%OVERWRITE%"=="1" set "EXTRA_ARGS=--overwrite"

call conda run -n "%VIZ_ENV_NAME%" python -m omnire_workflow.dataset_visualization.cli convert-ten ^
    --raw_dir "%RAW_DIR%" ^
    --project_output "%PROJECT_OUTPUT%" ^
    --map_file "%MAP_FILE%" ^
    --prefix "%PREFIX%" ^
    --num_proc "%NUM_PROC%" ^
    --camera_id "%CAMERA_ID%" ^
    %EXTRA_ARGS%
if errorlevel 1 exit /b 1

popd
endlocal
