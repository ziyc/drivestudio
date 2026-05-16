@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"

if "%KITTI_ROOT%"=="" set "KITTI_ROOT=OutPut\waymo_training_10scenes\scene_114\dataset_visualization\kitti"
if "%CAMERA_ID%"=="" set "CAMERA_ID=0"

python -m omnire_workflow.dataset_visualization.cli check --kitti_root "%KITTI_ROOT%" --camera_id "%CAMERA_ID%"
if errorlevel 1 exit /b 1

popd
endlocal
