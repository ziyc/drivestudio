@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run preprocessing and OmniRe training over the ten Waymo raw scenes
rem currently expected under data\waymo\raw.
rem Common overrides:
rem   set ENV_NAME=drivestudio
rem   set SCENE_IDS=0 1 4 8 32 102 109 114 149 156
rem   set SKIP_PREPROCESS=1
rem   set CONFIG_FILE=configs\omnire_extended_cam.yaml
rem   set DATASET=waymo/3cams
rem   set END_TIMESTEP=150

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if "%ENV_NAME%"=="" set "ENV_NAME=drivestudio"
if "%SCENE_IDS%"=="" set "SCENE_IDS=0 1 4 8 32 102 109 114 149 156"
if "%WORKERS%"=="" set "WORKERS=4"
if "%START_TIMESTEP%"=="" set "START_TIMESTEP=0"
if "%END_TIMESTEP%"=="" set "END_TIMESTEP=-1"
if "%OUTPUT_ROOT%"=="" set "OUTPUT_ROOT=.\OutPut"
if "%PROJECT%"=="" set "PROJECT=waymo_training_10scenes"
if "%CONFIG_FILE%"=="" set "CONFIG_FILE=configs\omnire.yaml"
if "%DATASET%"=="" set "DATASET=waymo/3cams"

set "PYTHONPATH=%CD%"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found in PATH.
    exit /b 1
)

if not "%SKIP_PREPROCESS%"=="1" (
    echo [1/3] Preprocessing Waymo scenes: %SCENE_IDS%
    call conda run -n "%ENV_NAME%" python datasets/preprocess.py ^
        --data_root data/waymo/raw/ ^
        --target_dir data/waymo/processed ^
        --dataset waymo ^
        --split training ^
        --scene_ids %SCENE_IDS% ^
        --workers %WORKERS% ^
        --process_keys images lidar calib pose dynamic_masks objects
    if errorlevel 1 exit /b 1
) else (
    echo [1/3] Skipping preprocessing because SKIP_PREPROCESS=1
)

if not "%SEGFORMER_PATH%"=="" (
    if "%SEGFORMER_CHECKPOINT%"=="" set "SEGFORMER_CHECKPOINT=%SEGFORMER_PATH%\pretrained\segformer.b5.1024x1024.city.160k.pth"
    echo [2/3] Extracting sky and fine dynamic masks with SegFormer
    call conda run -n "%ENV_NAME%" python datasets/tools/extract_masks.py ^
        --data_root data/waymo/processed/training ^
        --segformer_path "%SEGFORMER_PATH%" ^
        --checkpoint "%SEGFORMER_CHECKPOINT%" ^
        --scene_ids %SCENE_IDS% ^
        --process_dynamic_mask
    if errorlevel 1 exit /b 1
) else (
    echo [2/3] Skipping SegFormer masks. Set SEGFORMER_PATH to enable mask extraction.
)

echo [3/3] Training OmniRe scenes
for %%S in (%SCENE_IDS%) do (
    set "RUN_NAME=scene_%%S"
    echo.
    echo ===== Training scene %%S as !RUN_NAME! =====
    call conda run -n "%ENV_NAME%" python tools/train.py ^
        --config_file "%CONFIG_FILE%" ^
        --output_root "%OUTPUT_ROOT%" ^
        --project "%PROJECT%" ^
        --run_name "!RUN_NAME!" ^
        dataset=%DATASET% ^
        data.scene_idx=%%S ^
        data.start_timestep=%START_TIMESTEP% ^
        data.end_timestep=%END_TIMESTEP%
    if errorlevel 1 exit /b 1
)

echo.
echo All requested scenes finished.
popd
endlocal
