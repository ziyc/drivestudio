@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"

if "%PROJECT_OUTPUT%"=="" set "PROJECT_OUTPUT=OutPut\waymo_training_10scenes"
if "%MAP_FILE%"=="" set "MAP_FILE=omnire_workflow\dataset_visualization\waymo_10scene_map.txt"
if "%CAMERA_ID%"=="" set "CAMERA_ID=0"

python -m omnire_workflow.dataset_visualization.cli check-ten ^
    --project_output "%PROJECT_OUTPUT%" ^
    --map_file "%MAP_FILE%" ^
    --camera_id "%CAMERA_ID%"
if errorlevel 1 exit /b 1

popd
endlocal
