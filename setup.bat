@echo off
setlocal enabledelayedexpansion

:: Logging function
set LOGTIME=%DATE% %TIME%
echo [%LOGTIME%] Starting setup...

:: Check required tools
where pixi >nul 2>nul || (
    echo [ERROR] pixi not found. Please install Pixi.
    exit /b 1
)
where gdown >nul 2>nul || (
    echo [ERROR] gdown not found. Please install gdown (via pixi).
    exit /b 1
)
where unzip >nul 2>nul || (
    echo [ERROR] unzip not found. Please install unzip utility.
    exit /b 1
)

:: Set up directories
set FILEDIR=%cd%
set MODELS_DIR=%FILEDIR%\models
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
cd /d "%MODELS_DIR%"

:: Detectron2 install
echo Installing detectron2...
pixi run pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git || (
    echo [ERROR] Failed to install detectron2.
    exit /b 1
)

:: Define model files and IDs
set RETRY_COUNT=3
set FILE_COUNT=5

set file1=rcnn_bet365.pth
set id1=1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH

set file2=faster_rcnn.yaml
set id2=1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW

set file3=resnetv2_rgb_new.pth.tar
set id3=1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS

set file4=expand_targetlist.zip
set id4=1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I

set file5=domain_map.pkl
set id5=1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1

:: Download loop
for /L %%i in (1,1,%FILE_COUNT%) do (
    call set FILENAME=%%file%%i%%
    call set FILEID=%%id%%i%%

    if exist "!FILENAME!" (
        echo !FILENAME! already exists. Skipping.
    ) else (
        set /A count=0
        :retry
        echo Downloading !FILENAME! (Attempt !count!/!RETRY_COUNT!)...
        pixi run gdown --id !FILEID! -O "!FILENAME!" && goto next
        set /A count+=1
        if !count! LSS %RETRY_COUNT% (
            timeout /t 2 >nul
            goto retry
        ) else (
            echo [ERROR] Failed to download !FILENAME! after %RETRY_COUNT% attempts.
            exit /b 1
        )
        :next
    )
)

:: Extract expand_targetlist.zip
echo Extracting expand_targetlist.zip...
unzip -o expand_targetlist.zip -d expand_targetlist || (
    echo [ERROR] Failed to unzip file.
    exit /b 1
)

cd expand_targetlist
if exist expand_targetlist\ (
    echo Flattening nested expand_targetlist directory...
    move expand_targetlist\*.* . >nul
    rmdir expand_targetlist
)
cd "%MODELS_DIR%"

echo [SUCCESS] Model setup and extraction complete.
