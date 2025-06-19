#!/bin/bash

# SWEET Universal Installer for Linux/macOS
# Double-click this file to install SWEET

clear
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              SWEET Universal Installer                       ║"
echo "║      SAM Widget for Edge Evaluation Tool          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "Starting installation for $(uname)..."
echo

# Change to SWEET directory
cd "$(dirname "$0")"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python on different systems
install_python() {
    echo "Python 3.8+ not found. Attempting to install..."
    echo
    
    # Detect system and install Python
    if command_exists apt-get; then
        echo "Detected Debian/Ubuntu system. Installing Python..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv python3-dev
        return $?
    elif command_exists yum; then
        echo "Detected RedHat/CentOS system. Installing Python..."
        sudo yum install -y python3 python3-pip python3-devel
        return $?
    elif command_exists dnf; then
        echo "Detected Fedora system. Installing Python..."
        sudo dnf install -y python3 python3-pip python3-devel
        return $?
    elif command_exists brew; then
        echo "Detected macOS with Homebrew. Installing Python..."
        brew install python3
        return $?
    elif command_exists pacman; then
        echo "Detected Arch Linux system. Installing Python..."
        sudo pacman -S python python-pip
        return $?
    else
        echo "Could not detect package manager."
        echo "Please install Python 3.8+ manually and run this installer again."
        echo
        echo "Installation instructions:"
        echo "- Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
        echo "- CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "- Fedora: sudo dnf install python3 python3-pip"
        echo "- macOS: brew install python3 (requires Homebrew)"
        echo "- Arch: sudo pacman -S python python-pip"
        return 1
    fi
}

# Try to find Python 3
echo "Checking for Python..."

if command_exists python3; then
    # Check Python 3 version
    python3_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    if [ $? -eq 0 ]; then
        python3_major=$(echo $python3_version | cut -d. -f1)
        python3_minor=$(echo $python3_version | cut -d. -f2)
        
        if [ "$python3_major" -eq 3 ] && [ "$python3_minor" -ge 8 ]; then
            echo "Found Python $python3_version, starting installation..."
            python3 install.py
            exit $?
        else
            echo "Python $python3_version found, but need Python 3.8+"
        fi
    fi
elif command_exists python; then
    # Check if python command points to Python 3
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    if [ $? -eq 0 ]; then
        python_major=$(echo $python_version | cut -d. -f1)
        python_minor=$(echo $python_version | cut -d. -f2)
        
        if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 8 ]; then
            echo "Found Python $python_version, starting installation..."
            python install.py
            exit $?
        else
            echo "Python $python_version found, but need Python 3.8+"
        fi
    fi
fi

# No suitable Python found, try to install
echo
echo "No suitable Python installation found."
echo "SWEET requires Python 3.8 or higher."
echo

# Ask user if they want to install Python
while true; do
    read -p "Would you like to install Python 3 now? (y/N): " choice
    case $choice in
        [Yy]* )
            if install_python; then
                echo
                echo "Python installation completed. Starting SWEET installer..."
                echo
                
                # Try again with python3
                if command_exists python3; then
                    python3 install.py
                    exit $?
                elif command_exists python; then
                    python install.py
                    exit $?
                else
                    echo "Python installation completed but command not found."
                    echo "You may need to restart your terminal or add Python to PATH."
                    echo "Please try running this installer again."
                fi
            else
                echo
                echo "Python installation failed."
                echo "Please install Python manually and run this installer again."
            fi
            break
            ;;
        [Nn]* | "" )
            echo
            echo "Installation cancelled."
            echo "Please install Python 3.8+ manually and run this installer again."
            echo
            echo "Quick installation commands:"
            echo "- Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
            echo "- CentOS/RHEL: sudo yum install python3 python3-pip"
            echo "- Fedora: sudo dnf install python3 python3-pip"
            echo "- macOS: brew install python3"
            echo "- Arch: sudo pacman -S python python-pip"
            break
            ;;
        * )
            echo "Please answer y for yes or n for no."
            ;;
    esac
done

echo
echo "Press Enter to exit..."
read