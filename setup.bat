@echo off
setlocal enabledelayedexpansion

:: 设置环境名称
set "ENV_NAME=phishpedia"
if not defined ENV_NAME set "ENV_NAME=phishpedia"

:: 设置重试次数
set RETRY_COUNT=3

:: 检查 Conda 是否安装
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :error_exit "Conda is not installed. Please install Conda and try again."
    exit /b 1
)

:: 检查环境是否存在
conda env list | find /i "%ENV_NAME%" >nul
if %ERRORLEVEL% equ 0 (
    echo Activating existing Conda environment: %ENV_NAME%
) else (
    echo Creating new Conda environment: %ENV_NAME% with Python 3.8
    conda create -y -n %ENV_NAME% python=3.8
    if %ERRORLEVEL% neq 0 (
        call :error_exit "Failed to create Conda environment."
        exit /b 1
    )
)

:: 激活环境
call conda activate %ENV_NAME%
if %ERRORLEVEL% neq 0 (
    call :error_exit "Failed to activate Conda environment."
    exit /b 1
)

:: 安装 gdown
call pip show gdown >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing gdown...
    pip install gdown
    echo %ERRORLEVEL%
    @REM if %ERRORLEVEL% neq 0 (
    @REM     call :error_exit "Failed to install gdown."
    @REM     exit /b 1
    @REM )
)

:: 检查 CUDA 可用性
nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo CUDA detected. Installing GPU-supported PyTorch and torchvision...
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install git+https://github.com/facebookresearch/detectron2.git
) else (
    echo No CUDA detected. Installing CPU-only PyTorch and torchvision...
    pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install git+https://github.com/facebookresearch/detectron2.git
)

:: 安装依赖
if exist requirements.txt (
    echo Installing requirements...
    pip install -r requirements.txt
) else (
    call :error_exit "requirements.txt not found."
    exit /b 1
)

:: 创建模型目录
set "MODELS_DIR=%CD%\models"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
cd "%MODELS_DIR%"



:: 下载所有模型文件
call :download_file "1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH" "rcnn_bet365.pth"
call :download_file "1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW" "faster_rcnn.yaml"
call :download_file "1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS" "resnetv2_rgb_new.pth.tar"
call :download_file "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I" "expand_targetlist.zip"
call :download_file "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1" "domain_map.pkl"

:: 解压文件
echo Extracting expand_targetlist.zip...
tar -xvzf expand_targetlist.zip
if %ERRORLEVEL% neq 0 (
    call :error_exit "Failed to extract expand_targetlist.zip"
    exit /b 1
)
rmdir /s /q "__MACOSX" 

:: 处理嵌套目录
cd expand_targetlist
if exist "expand_targetlist\" (
    echo Moving files from nested directory...
    move "expand_targetlist\*" "."
    rmdir "expand_targetlist"
)
del .DS_Store

cd ..

echo Installation completed successfully!
endlocal
exit /b 1

:: 错误处理函数
:error_exit
echo Error: %~1
exit /b 1

:: 下载模型文件函数
:download_file
set "file_id=%~1"
set "file_name=%~2"
set "retry_count=0"

:download_retry
if exist "%file_name%" (
    echo %file_name% already exists. Skipping download.
    goto :eof
)

echo Attempting to download %file_name% ^(Attempt !retry_count! of %RETRY_COUNT%^)
gdown --id "%file_id%" -O "%file_name%"
if %ERRORLEVEL% equ 0 (
    echo Successfully downloaded %file_name%
    goto :eof
)

set /a retry_count+=1
if !retry_count! lss %RETRY_COUNT% (
    echo Retrying download...
    timeout /t 2 >nul
    goto download_retry
) else (
    call :error_exit "Failed to download %file_name% after %RETRY_COUNT% attempts."
    exit /b 1
)