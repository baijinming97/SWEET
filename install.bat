@echo off
REM SWEET Universal Installer for Windows
REM Double-click this file to install SWEET

title SWEET Installer - Windows
chcp 65001 >nul 2>&1

echo.
echo ============================================================
echo                SWEET Universal Installer                  
echo      SAM Widget for Edge Evaluation Tool        
echo ============================================================
echo.
echo Starting installation for Windows...
echo.

REM Change to SWEET directory
cd /d "%~dp0"

REM Function to check Python version
REM Try to run with different Python commands
echo Checking for Python...

REM Try py launcher first (recommended for Windows)
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    REM Check if version is 3.8+
    for /f "tokens=2" %%i in ('py -3 --version 2^>^&1') do set PYTHON_VERSION=%%i
    for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
        if %%a geq 3 (
            if %%b geq 8 (
                echo Found Python %PYTHON_VERSION%, starting installation...
                py -3 src\install.py
                goto :end
            ) else (
                echo Found Python %PYTHON_VERSION%, but need Python 3.8+
            )
        )
    )
)

REM Try python command
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
        if %%a geq 3 (
            if %%b geq 8 (
                echo Found Python %PYTHON_VERSION%, starting installation...
                python src\install.py
                goto :end
            ) else (
                echo Found Python %PYTHON_VERSION%, but need Python 3.8+
            )
        )
    )
)

REM Try python3 command
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python3 --version 2^>^&1') do set PYTHON_VERSION=%%i
    for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
        if %%a geq 3 (
            if %%b geq 8 (
                echo Found Python %PYTHON_VERSION%, starting installation...
                python3 src\install.py
                goto :end
            ) else (
                echo Found Python %PYTHON_VERSION%, but need Python 3.8+
            )
        )
    )
)

REM No Python found, try to install it
echo.
echo Python not found on your system.
echo.
echo SWEET will now download and install Python 3.12 for you.
echo This requires internet connection and may take a few minutes.
echo.
pause

echo Downloading Python 3.12...

REM Function to get Python URL from config file
call :get_python_url
echo Detected architecture: %PROCESSOR_ARCHITECTURE%
echo Download URL: %PYTHON_URL%

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile 'python_installer_temp.exe' -UserAgent 'Mozilla/5.0' } catch { Write-Host 'Download failed. Please check your internet connection.'; exit 1 }}"

if not exist python_installer_temp.exe (
    echo.
    echo Failed to download Python installer.
    echo Please check your internet connection and try again.
    echo.
    echo Alternatively, you can:
    echo 1. Install Python 3.8+ manually from https://python.org
    echo 2. Run this installer again
    echo.
    pause
    exit /b 1
)

echo Installing Python 3.12...
python_installer_temp.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_doc=0

REM Wait a bit for installation to complete
timeout /t 5 /nobreak >nul

REM Clean up installer
del python_installer_temp.exe

echo.
echo Python installation completed. Restarting SWEET installer...
echo.

REM Refresh PATH and try again
call refreshenv >nul 2>&1

REM Try different Python commands again
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python src\install.py
    goto :end
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    python3 src\install.py
    goto :end
)

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    py -3 src\install.py
    goto :end
)

REM If still no Python, show manual instructions
echo.
echo Python installation may not be complete or PATH not updated.
echo Please:
echo 1. Restart your computer
echo 2. Try running this installer again
echo.
echo Or manually install Python from https://python.org and try again.
echo.

:end
echo.
pause
exit /b 0

REM Function to get Python URL from config file
:get_python_url
set CONFIG_FILE=src\python_urls.txt
set PYTHON_URL=

REM Detect system architecture
set ARCH_KEY=X64
if "%PROCESSOR_ARCHITECTURE%"=="x86" set ARCH_KEY=X86
if "%PROCESSOR_ARCHITEW6432%"=="ARM64" set ARCH_KEY=ARM64

REM Try to read URL from config file
if exist "%CONFIG_FILE%" (
    for /f "tokens=1,2 delims==" %%a in ('type "%CONFIG_FILE%" ^| findstr /i "WINDOWS_%ARCH_KEY%="') do (
        set PYTHON_URL=%%b
    )
)

REM Fallback to hardcoded URLs if config file not found or URL not set
if "%PYTHON_URL%"=="" (
    if "%ARCH_KEY%"=="ARM64" (
        set PYTHON_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0-arm64.exe
    ) else if "%ARCH_KEY%"=="X86" (
        set PYTHON_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0.exe
    ) else (
        set PYTHON_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe
    )
)

exit /b 0