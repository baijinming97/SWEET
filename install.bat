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

REM Try to run with different Python commands
echo Checking for Python...

REM Try python command first
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Found Python, starting installation...
    python src\install.py
    goto :end
)

REM Try python3 command
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Found Python3, starting installation...
    python3 src\install.py
    goto :end
)

REM Try py launcher
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Found Python via py launcher, starting installation...
    py -3 src\install.py
    goto :end
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
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer_temp.exe' -UserAgent 'Mozilla/5.0' } catch { Write-Host 'Download failed. Please check your internet connection.'; exit 1 }}"

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