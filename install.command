#!/bin/bash

# SWEET Universal Installer for Linux/macOS
# Double-click this file to install SWEET

clear
echo "=============================================="
echo "          SWEET Universal Installer           "
echo "   SAM Widget for Edge Evaluation Tool        "
echo "=============================================="
echo
echo "Starting installation for $(uname)..."
echo

# Change to SWEET directory
cd "$(dirname "$0")"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to read Python URLs from config file
get_python_url() {
    local platform="$1"
    local arch="$2"
    local config_file="src/python_urls.txt"
    
    if [ -f "$config_file" ]; then
        # Look for the specific platform_arch combination
        local key="${platform}_${arch}"
        local url=$(grep "^${key}=" "$config_file" | cut -d'=' -f2-)
        if [ -n "$url" ]; then
            echo "$url"
            return 0
        fi
    fi
    
    # Fallback to hardcoded URLs if config file not found
    case "${platform}_${arch}" in
        "MACOS_ARM64")
            echo "https://www.python.org/ftp/python/3.12.0/python-3.12.0-macos11.pkg"
            ;;
        "MACOS_X64")
            echo "https://www.python.org/ftp/python/3.12.0/python-3.12.0-macosx10.9.pkg"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to download and install Python automatically
install_python() {
    echo "Python 3.8+ not found. Attempting to download and install..."
    echo
    
    # Detect system and download appropriate Python installer
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
    elif command_exists pacman; then
        echo "Detected Arch Linux system. Installing Python..."
        sudo pacman -S python python-pip
        return $?
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - download and install Python directly
        echo "Detected macOS. Downloading Python 3.12..."
        
        # Determine architecture and get URL from config
        if [[ $(uname -m) == "arm64" ]]; then
            PYTHON_URL=$(get_python_url "MACOS" "ARM64")
            echo "Downloading Python for Apple Silicon (M1/M2)..."
        else
            PYTHON_URL=$(get_python_url "MACOS" "X64")
            echo "Downloading Python for Intel Mac..."
        fi
        
        if [ -z "$PYTHON_URL" ]; then
            echo "Could not determine Python download URL."
            return 1
        fi
        
        # Download Python installer
        if command_exists curl; then
            curl -L "$PYTHON_URL" -o python_installer_temp.pkg
        elif command_exists wget; then
            wget "$PYTHON_URL" -O python_installer_temp.pkg
        else
            echo "Neither curl nor wget found. Cannot download Python."
            echo "Please install Python manually from https://python.org"
            return 1
        fi
        
        if [ ! -f "python_installer_temp.pkg" ]; then
            echo "Failed to download Python installer."
            echo "Please check your internet connection and try again."
            return 1
        fi
        
        echo "Installing Python 3.12..."
        sudo installer -pkg python_installer_temp.pkg -target /
        
        # Clean up
        rm -f python_installer_temp.pkg
        
        # Update PATH for current session
        export PATH="/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH"
        
        echo "Python installation completed."
        return 0
    elif command_exists brew; then
        echo "Detected macOS with Homebrew. Installing Python..."
        brew install python3
        return $?
    else
        echo "Attempting to download Python for your system..."
        
        # Try to download Python based on uname
        case "$(uname -s)" in
            Linux*)
                echo "Detected Linux system without package manager."
                echo "This installer cannot automatically install Python on this system."
                echo "Please install Python 3.8+ manually using your system's package manager."
                ;;
            Darwin*)
                echo "Detected macOS without Homebrew."
                echo "Will attempt direct download..."
                # Fallback to direct download using config file
                if [[ $(uname -m) == "arm64" ]]; then
                    PYTHON_URL=$(get_python_url "MACOS" "ARM64")
                else
                    PYTHON_URL=$(get_python_url "MACOS" "X64")
                fi
                
                if [ -n "$PYTHON_URL" ] && command_exists curl; then
                    curl -L "$PYTHON_URL" -o python_installer_temp.pkg
                    sudo installer -pkg python_installer_temp.pkg -target /
                    rm -f python_installer_temp.pkg
                    export PATH="/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH"
                    return 0
                fi
                ;;
            *)
                echo "Unknown operating system: $(uname -s)"
                ;;
        esac
        
        echo
        echo "Automatic installation failed."
        echo "Please install Python 3.8+ manually and run this installer again."
        echo
        echo "Installation instructions:"
        echo "- Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
        echo "- CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "- Fedora: sudo dnf install python3 python3-pip"
        echo "- macOS: Download from https://python.org/downloads/"
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
            python3 src/install.py
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
            python src/install.py
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
                    python3 src/install.py
                    exit $?
                elif command_exists python; then
                    python src/install.py
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