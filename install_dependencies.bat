@echo off
pushd %~dp0

echo Select version to install:
echo 1. CUDA (Fast processing if you have an NVIDIA GPU)
echo 2. CPU (Slow performance but smaller download)
echo 3. Download models only (Only needed if the models fail to download)
echo 4. Exit

choice /C 1234 /N /M "Enter your choice (1, 2, 3 or 4):"

if errorlevel 4 (
    echo Exiting...
    exit /b 0
) else if errorlevel 3 (
    .\python-3.12.8-embed-amd64\python.exe .\sammie\download_models.py
) else if errorlevel 2 (
    .\python-3.12.8-embed-amd64\python.exe -m pip install --upgrade pip --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install wheel --no-warn-script-location
    echo Uninstalling existing Pytorch if found
    .\python-3.12.8-embed-amd64\python.exe -m pip uninstall -y torch torchvision
    echo Installing CPU version of PyTorch
    .\python-3.12.8-embed-amd64\python.exe -m pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install -r .\requirements.txt --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe .\sammie\download_models.py
) else (
    .\python-3.12.8-embed-amd64\python.exe -m pip install --upgrade pip --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install wheel --no-warn-script-location
    echo Uninstalling existing Pytorch if found
    .\python-3.12.8-embed-amd64\python.exe -m pip uninstall -y torch torchvision
    echo Installing CUDA version of PyTorch
    .\python-3.12.8-embed-amd64\python.exe -m pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install -r .\requirements.txt --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe .\sammie\download_models.py
)

echo Completed.
Pause