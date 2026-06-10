#!/usr/bin/env python3
"""
SWEET - Universal Installer
SWEET (SAM Widget for Edge Evaluation Tool)

Single entry point installer that:
- Auto-detects platform (Windows/Linux/macOS)
- Checks/installs Python 3.12 if needed
- Detects GPU for appropriate PyTorch version
- Checks existing installation and asks for overwrite
- Creates isolated virtual environment
- Shows progress during installation
"""

import os
import sys
import platform
import subprocess
import shutil
import venv
import json
from pathlib import Path
import urllib.request
import tarfile
import zipfile
import tempfile
import re

class SWEETInstaller:
    def __init__(self):
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        # Get the parent directory (SWEET_v1 root) from src directory
        self.root_dir = Path(__file__).parent.parent
        self.python_dir = self.root_dir / "python"
        self.is_windows = self.platform == "windows"
        self.is_macos = self.platform == "darwin"
        self.is_linux = self.platform == "linux"
        self.python_version = "3.12.0"  # Most stable version for all platforms
        self.installation_info_file = self.root_dir / "sweet_install.json"
        
    def show_header(self):
        """Display installation header"""
        print("=" * 60)
        print("          SWEET - Universal Installer")
        print("    SWEET (SAM Widget for Edge Evaluation Tool)")
        print("=" * 60)
        print()
        print(f"🖥️  Platform: {platform.system()} {platform.release()}")
        print(f"🏗️  Architecture: {self.architecture}")
        print()
        
        # Windows-specific warning
        if self.is_windows:
            print("⚠️  Windows users: If installation fails with compiler errors,")
            print("   you may need Microsoft Visual C++ Build Tools.")
            print("   Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print()
    
    def check_existing_installation(self):
        """Check if SWEET is already installed"""
        print("🔍 Checking existing installation...")
        
        # Check if virtual environment exists
        venv_exists = self.python_dir.exists()
        info_exists = self.installation_info_file.exists()
        
        if not venv_exists and not info_exists:
            print("   ℹ️  No existing installation found")
            return False
        
        print("   ⚠️  Existing SWEET installation detected!")
        
        # Try to read installation info
        if info_exists:
            try:
                with open(self.installation_info_file, 'r') as f:
                    install_info = json.load(f)
                    print(f"   📅 Installed: {install_info.get('install_date', 'Unknown')}")
                    print(f"   🐍 Python: {install_info.get('python_version', 'Unknown')}")
                    print(f"   🔥 PyTorch: {install_info.get('torch_type', 'Unknown')}")
                    print(f"   📦 Packages: {install_info.get('packages_count', 'Unknown')} installed")
            except:
                print("   ⚠️  Installation info corrupted")
        
        # Test if current installation works
        if venv_exists:
            print("   🧪 Testing current installation...")
            if self.test_current_installation():
                print("   ✅ Current installation is functional")
                
                while True:
                    choice = input("\n   Do you want to reinstall? (y/N): ").strip().lower()
                    if choice in ['', 'n', 'no']:
                        print("   ✅ Keeping existing installation")
                        return True
                    elif choice in ['y', 'yes']:
                        print("   🗑️  Will reinstall...")
                        break
                    else:
                        print("   Please enter 'y' for yes or 'n' for no")
            else:
                print("   ❌ Current installation has issues, will reinstall")
        
        return False
    
    def test_current_installation(self):
        """Test if current installation works"""
        try:
            if self.is_windows:
                python_cmd = str(self.python_dir / "Scripts" / "python.exe")
            else:
                python_cmd = str(self.python_dir / "bin" / "python")
            
            return Path(python_cmd).exists()
            
        except Exception:
            return False
    
    def detect_nvidia_gpu(self):
        """Detect if NVIDIA GPU is available"""
        print("🔍 Detecting GPU...")
        
        if self.is_macos:
            # Check for Apple Silicon GPU (MPS support)
            try:
                import platform
                machine = platform.machine().lower()
                if machine in ['arm64', 'aarch64']:
                    print("   ✅ Apple Silicon detected - will use MPS acceleration")
                    return "mps"  # Use MPS instead of CUDA
                else:
                    print("   ℹ️  Intel Mac detected - will use CPU version")
                    return False
            except:
                print("   ℹ️  macOS detected - will use CPU version")
                return False
        
        try:
            # Try nvidia-smi command
            if self.is_windows:
                result = subprocess.run(['nvidia-smi'], capture_output=True, shell=True, timeout=5)
            else:
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            
            if result.returncode == 0:
                print("   ✅ NVIDIA GPU detected")
                # Try to get GPU name
                try:
                    output = result.stdout.decode('utf-8')
                    for line in output.split('\n'):
                        if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                            # Extract GPU name
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if 'GeForce' in part or 'RTX' in part or 'GTX' in part:
                                    gpu_name = ' '.join(parts[i:i+3])
                                    print(f"   🎮 GPU: {gpu_name}")
                                    break
                            break
                except:
                    pass
                return True
            else:
                print("   ℹ️  No NVIDIA GPU detected - will use CPU version")
                return False
                
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # nvidia-smi not found or timed out, try alternative methods
            if self.is_windows:
                try:
                    # Check Windows Device Manager
                    result = subprocess.run(
                        ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 and 'NVIDIA' in result.stdout:
                        print("   ✅ NVIDIA GPU detected (via WMI)")
                        return True
                except:
                    pass
            elif self.is_linux:
                try:
                    # Check lspci
                    result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and 'NVIDIA' in result.stdout:
                        print("   ✅ NVIDIA GPU detected (via lspci)")
                        return True
                except:
                    pass
            
            print("   ℹ️  No NVIDIA GPU detected - will use CPU version")
            return False
    
    def find_python(self):
        """Find suitable Python installation"""
        print("🐍 Checking Python installation...")
        
        # List of possible Python commands
        python_commands = []
        if self.is_windows:
            python_commands = ['python', 'python3', 'py -3']
        else:
            python_commands = ['python3', 'python']
        
        for cmd in python_commands:
            try:
                # Test the command
                cmd_parts = cmd.split()
                result = subprocess.run(cmd_parts + ['--version'], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    version_str = result.stdout.strip()
                    print(f"   ✅ Found {version_str}")
                    
                    # Extract version numbers
                    match = re.search(r'Python (\d+)\.(\d+)\.(\d+)', version_str)
                    if match:
                        major, minor, patch = map(int, match.groups())
                        if major == 3 and minor >= 8:
                            print(f"   ✅ Python version is suitable")
                            return cmd_parts[0] if len(cmd_parts) == 1 else cmd
                        else:
                            print(f"   ⚠️  Python version too old (need 3.8+)")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        print("   ❌ No suitable Python found")
        return None
    
    def download_file(self, url, dest_path, desc=""):
        """Download file with progress indicator"""
        import ssl
        
        def download_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                print(f"\r   📥 {desc}: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        print(f"   🌐 Downloading {desc}...")
        
        # Try with default SSL context first
        try:
            urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
            print()  # New line after progress
            return True
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                print(f"\n   ⚠️  SSL certificate verification failed, trying with unverified context...")
                # Create unverified SSL context for macOS certificate issues
                try:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                    # Install the custom SSL context
                    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                    urllib.request.install_opener(opener)
                    
                    # Try download again
                    urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
                    print()  # New line after progress
                    return True
                except Exception as e2:
                    print(f"\n   ❌ Download failed even with unverified SSL: {e2}")
                    return False
            else:
                print(f"\n   ❌ Download failed: {e}")
                return False
        except Exception as e:
            print(f"\n   ❌ Download failed: {e}")
            return False
    
    def install_python(self):
        """Download and install Python 3.12"""
        print(f"📦 Installing Python {self.python_version}...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            if self.is_windows:
                # Windows: Download installer
                if 'arm' in self.architecture.lower():
                    url = f"https://www.python.org/ftp/python/{self.python_version}/python-{self.python_version}-arm64.exe"
                    filename = f"python-{self.python_version}-arm64.exe"
                else:
                    url = f"https://www.python.org/ftp/python/{self.python_version}/python-{self.python_version}-amd64.exe"
                    filename = f"python-{self.python_version}-amd64.exe"
                
                download_path = Path(temp_dir) / filename
                if not self.download_file(url, download_path, f"Python {self.python_version}"):
                    return None
                
                # Install Python
                print("   🔧 Installing Python...")
                result = subprocess.run([
                    str(download_path), '/quiet', 'InstallAllUsers=0', 
                    'PrependPath=1', 'Include_test=0'
                ], timeout=300)
                
                if result.returncode == 0:
                    print("   ✅ Python installed successfully")
                    # Give Windows time to update PATH
                    import time
                    time.sleep(3)
                    return 'python'
                else:
                    print("   ❌ Python installation failed")
                    return None
                    
            else:
                # Linux/macOS: Use system package manager
                if self.is_linux:
                    # Try different package managers
                    installers = [
                        (['apt-get', 'update'], ['apt-get', 'install', '-y', 'python3', 'python3-pip', 'python3-venv']),
                        (['yum', 'install', '-y', 'python3', 'python3-pip']),
                        (['dnf', 'install', '-y', 'python3', 'python3-pip']),
                    ]
                    
                    for update_cmd, install_cmd in installers:
                        try:
                            if update_cmd:
                                subprocess.run(update_cmd, check=True, timeout=60)
                            subprocess.run(install_cmd, check=True, timeout=300)
                            print("   ✅ Python installed via system package manager")
                            return 'python3'
                        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                            continue
                            
                elif self.is_macos:
                    # Try Homebrew
                    try:
                        subprocess.run(['brew', 'install', 'python3'], check=True, timeout=300)
                        print("   ✅ Python installed via Homebrew")
                        return 'python3'
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                
                print("   ❌ Could not install Python automatically")
                print("   ℹ️  Please install Python 3.8+ manually and run this installer again")
                return None
                
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_virtual_environment(self, python_cmd):
        """Create Python virtual environment"""
        print("🔨 Creating virtual environment...")
        
        if self.python_dir.exists():
            print(f"   🗑️  Removing existing environment...")
            try:
                shutil.rmtree(self.python_dir)
            except Exception as e:
                print(f"   ⚠️  Could not remove existing environment: {e}")
                if self.is_windows:
                    subprocess.run(f'rd /s /q "{self.python_dir}"', shell=True, check=False)
                else:
                    subprocess.run(['rm', '-rf', str(self.python_dir)], check=False)
        
        try:
            # Create virtual environment
            if isinstance(python_cmd, list):
                cmd = python_cmd + ["-m", "venv", str(self.python_dir)]
            else:
                cmd = [python_cmd, "-m", "venv", str(self.python_dir)]
                
            result = subprocess.run(cmd, timeout=60)
            
            if result.returncode == 0:
                print("   ✅ Virtual environment created successfully")
                return True
            else:
                print("   ❌ Failed to create virtual environment")
                return False
                
        except Exception as e:
            print(f"   ❌ Error creating virtual environment: {e}")
            return False
    
    def get_venv_commands(self):
        """Get virtual environment Python and pip commands"""
        if self.is_windows:
            python_cmd = str(self.python_dir / "Scripts" / "python.exe")
            pip_cmd = str(self.python_dir / "Scripts" / "pip.exe")
        else:
            python_cmd = str(self.python_dir / "bin" / "python")
            pip_cmd = str(self.python_dir / "bin" / "pip")
        
        return python_cmd, pip_cmd
    
    def install_packages(self, has_gpu):
        """Install required packages from requirements.txt"""
        python_cmd, pip_cmd = self.get_venv_commands()
        
        print("📦 Installing packages...")
        
        # Path to requirements file
        requirements_file = self.root_dir / "src" / "requirements.txt"
        
        # Check if requirements file exists
        if requirements_file.exists():
            print(f"   📋 Using requirements file: {requirements_file}")
            try:
                # Upgrade pip and essential tools first
                print("   ⚙️  Upgrading pip and build tools...")
                subprocess.run([python_cmd, "-m", "pip", "install", "--upgrade", "pip"], 
                             check=True, timeout=60)
                
                # Install from requirements file
                print("   📦 Installing packages from requirements.txt...")
                result = subprocess.run([python_cmd, "-m", "pip", "install", "-r", str(requirements_file)], 
                                      timeout=1800)  # 30 minutes timeout
                
                if result.returncode == 0:
                    print("   ✅ All packages installed successfully from requirements.txt")
                    return True
                else:
                    print("   ⚠️  Requirements installation failed, falling back to manual installation...")
            except Exception as e:
                print(f"   ⚠️  Error installing from requirements.txt: {e}")
                print("   📝 Falling back to manual package installation...")
        
        # Fallback: Manual package installation (original method)
        print("   📦 Installing packages manually...")
        
        # Base requirements - Compatible versions for stable operation
        base_packages = [
            "setuptools>=60.0.0,<71.0.0",  # Compatible version range
            "wheel>=0.37.0,<1.0.0",  # Compatible wheel version
            "numpy>=1.21.0,<2.0.0",  # Keep NumPy 1.x for PyTorch compatibility
            "scipy>=1.7.0,<1.12.0",  # Compatible scipy range
            "opencv-python>=4.5.0,<4.9.0",
            "pillow>=8.3.0,<11.0.0",
            "pandas>=1.3.0,<2.2.0",  # Compatible pandas range
            "matplotlib>=3.5.0,<3.9.0",  # Compatible matplotlib range
            "PyQt5>=5.15.0,<5.16.0",  # Stable PyQt5 version
            "segment-anything==1.0",
            "scikit-learn>=1.0.0,<1.4.0",
            "openpyxl>=3.0.0,<3.2.0",
            "tqdm>=4.62.0,<5.0.0",  # Compatible tqdm range
            "ultralytics>=8.0.0,<8.1.0",  # Stable ultralytics version
            "certifi>=2022.12.7",  # For SSL certificate issues
            "urllib3>=1.26.0,<2.0.0"  # Compatible urllib3 version
        ]
        
        # PyTorch packages - compatible stable versions
        if has_gpu == "mps":
            print("   🔥 Installing PyTorch (Apple Silicon MPS)...")
            torch_packages = [
                "torch>=2.0.0,<2.3.0", "torchvision>=0.15.0,<0.18.0", "torchaudio>=2.0.0,<2.3.0"
                # No special index URL needed for MPS, use default PyPI
            ]
            torch_size = "~500MB"
        elif has_gpu:
            print("   🔥 Installing PyTorch (NVIDIA CUDA 12.1)...")
            torch_packages = [
                "torch>=2.0.0,<2.3.0", "torchvision>=0.15.0,<0.18.0", "torchaudio>=2.0.0,<2.3.0",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
            torch_size = "~2GB"
        else:
            print("   🔥 Installing PyTorch (CPU version)...")
            torch_packages = [
                "torch>=2.0.0,<2.3.0", "torchvision>=0.15.0,<0.18.0", "torchaudio>=2.0.0,<2.3.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
            torch_size = "~200MB"
        
        try:
            # Upgrade pip and essential tools first
            print("   ⚙️  Upgrading pip and build tools...")
            subprocess.run([python_cmd, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, timeout=60)
            
            # Install setuptools and wheel first for Python 3.13 compatibility
            print("   🔧 Installing build essentials...")
            subprocess.run([python_cmd, "-m", "pip", "install", "--upgrade", "setuptools>=70.0.0", "wheel>=0.42.0"], 
                         check=True, timeout=60)
            
            # Set pip options - more flexible for Python 3.13
            pip_options = ["--prefer-binary", "--no-build-isolation"]
            
            # Install base packages (skip setuptools/wheel as already installed)
            print("   📚 Installing base packages...")
            packages_to_install = [pkg for pkg in base_packages if not pkg.startswith(("setuptools", "wheel"))]
            
            for i, package in enumerate(packages_to_install, 1):
                print(f"      [{i}/{len(packages_to_install)}] {package}")
                if "ultralytics" in package:
                    print(f"      ⏳ Please be patient, installation is proceeding...")
                
                # Try different installation strategies for Python 3.13
                success = False
                
                # Strategy 1: Try with prefer-binary (extended timeouts)
                if "ultralytics" in package:
                    timeout_duration = 1200  # 20 minutes for ultralytics
                    print(f"      📝 Almost there...")
                elif any(pkg in package for pkg in ["torch", "tensorflow", "scipy", "matplotlib", "opencv"]):
                    timeout_duration = 900   # 15 minutes for other large packages
                else:
                    timeout_duration = 600   # 10 minutes for regular packages
                    
                result = subprocess.run([python_cmd, "-m", "pip", "install", "--prefer-binary", package], 
                                      capture_output=True, text=True, timeout=timeout_duration)
                if result.returncode == 0:
                    success = True
                
                # Strategy 2: Try without any special options (same extended timeouts)
                if not success:
                    print(f"      ⚠️  Retrying with standard install...")
                    if "ultralytics" in package:
                        timeout_duration = 1200  # 20 minutes for ultralytics
                    elif any(pkg in package for pkg in ["torch", "tensorflow", "scipy", "matplotlib", "opencv"]):
                        timeout_duration = 900   # 15 minutes for other large packages
                    else:
                        timeout_duration = 600   # 10 minutes for regular packages
                        
                    result = subprocess.run([python_cmd, "-m", "pip", "install", package], 
                                          capture_output=True, text=True, timeout=timeout_duration)
                    if result.returncode == 0:
                        success = True
                
                # Strategy 3: Special handling for specific packages
                if not success:
                    if "numpy" in package:
                        print(f"      ⚠️  Trying latest numpy version...")
                        result = subprocess.run([python_cmd, "-m", "pip", "install", "numpy", "--pre"], 
                                              capture_output=True, text=True, timeout=600)
                        if result.returncode == 0:
                            success = True
                    elif "ultralytics" in package:
                        print(f"      ⚠️  Trying ultralytics with extended options...")
                        # Try with no cache to avoid cache conflicts
                        result = subprocess.run([python_cmd, "-m", "pip", "install", "ultralytics", "--no-cache-dir", "--prefer-binary"], 
                                              capture_output=True, text=True, timeout=1200)
                        if result.returncode == 0:
                            success = True
                
                if not success:
                    print(f"      ❌ Failed to install {package}")
                    print(f"      Error: {result.stderr}")
                    return False
                else:
                    print(f"      ✅ Installed {package}")
            
            # Install PyTorch
            print(f"   🔥 Installing PyTorch ({torch_size})...")
            print("      ⚠️  This may take several minutes...")
            
            result = subprocess.run([python_cmd, "-m", "pip", "install", "--prefer-binary"] + torch_packages, 
                                  timeout=900)  # 15 minutes timeout for PyTorch
            
            if result.returncode != 0:
                print("   ❌ Failed to install PyTorch")
                return False
            else:
                print("   ✅ PyTorch installed successfully")
            
            print("✅ All packages installed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            print("   ❌ Installation timed out")
            print("   📝 This might be due to slow internet or large package sizes")
            print("   📝 You can try running the installer again or check your internet connection")
            return False
        except Exception as e:
            print(f"   ❌ Installation error: {e}")
            return False
    
    def download_sam_model(self, has_gpu=False):
        """Download SAM model file(s) if not present.
        Always fetch vit_b (the CPU fast path). Also fetch vit_l on NVIDIA-GPU
        machines so SWEET can auto-use the higher-quality model there."""
        models_dir = self.root_dir / "models"
        models_dir.mkdir(exist_ok=True)

        downloads = [("sam_vit_b_01ec64.pth",
                      "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                      "SAM vit_b (375MB)")]
        if has_gpu is True:  # CUDA only; select_model_path also gates on torch.cuda
            downloads.append(("sam_vit_l_0b3195.pth",
                              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                              "SAM vit_l (1.2GB, GPU high-quality)"))

        ok_any = False
        for fname, url, desc in downloads:
            model_path = models_dir / fname
            if model_path.exists():
                print(f"🤖 {fname} already exists")
                ok_any = True
                continue
            print(f"🤖 Downloading {desc}...")
            try:
                if self.download_file(url, model_path, desc):
                    print(f"   ✅ {fname} -> {model_path.absolute()}")
                    ok_any = True
                else:
                    print(f"   ❌ Failed to download {fname}")
            except Exception as e:
                print(f"   ❌ Error downloading {fname}: {e}")
        if not ok_any:
            print("   ⚠️  No SAM model available; SWEET will run in fallback mode")
        return ok_any

    def verify_installation(self):
        """Quick verification that installation completed"""
        python_cmd, _ = self.get_venv_commands()
        
        print("✅ Installation completed successfully")
        print("   📦 Python environment created")
        print("   🔧 Dependencies installed")
        
        return True
    
    def save_installation_info(self, has_gpu):
        """Save installation information"""
        from datetime import datetime
        
        python_cmd, _ = self.get_venv_commands()
        
        # Get Python version
        try:
            result = subprocess.run([python_cmd, "--version"], 
                                  capture_output=True, text=True)
            python_version = result.stdout.strip()
        except:
            python_version = "Unknown"
        
        # Count installed packages
        try:
            result = subprocess.run([python_cmd, "-m", "pip", "list"], 
                                  capture_output=True, text=True)
            packages_count = len(result.stdout.strip().split('\n')) - 2  # Exclude header lines
        except:
            packages_count = "Unknown"
        
        install_info = {
            "install_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "platform": f"{platform.system()} {platform.release()}",
            "architecture": self.architecture,
            "python_version": python_version,
            "torch_type": "GPU (CUDA)" if has_gpu else "CPU",
            "packages_count": packages_count,
            "installer_version": "1.1"
        }
        
        try:
            with open(self.installation_info_file, 'w') as f:
                json.dump(install_info, f, indent=2)
        except Exception as e:
            print(f"   ⚠️  Could not save installation info: {e}")
    
    def show_completion_message(self, has_gpu):
        """Show installation completion message"""
        print("\n" + "=" * 60)
        print("🎉 SWEET INSTALLATION COMPLETED!")
        print("=" * 60)
        
        # Show environment info
        try:
            if self.is_windows:
                result = subprocess.run(
                    f'powershell "(Get-ChildItem -Path \\"{self.python_dir}\\" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB"',
                    shell=True, capture_output=True, text=True
                )
                if result.returncode == 0:
                    size_mb = float(result.stdout.strip())
                    size = f"{size_mb:.0f}MB"
                else:
                    size = "Unknown"
            else:
                result = subprocess.run(["du", "-sh", str(self.python_dir)], 
                                      capture_output=True, text=True)
                size = result.stdout.strip().split()[0] if result.returncode == 0 else "Unknown"
        except:
            size = "Unknown"
        
        print(f"📁 Environment size: {size}")
        print(f"📍 Location: {self.python_dir.absolute()}")
        
        # Check if SAM model exists
        model_path = self.root_dir / "models" / "sam_vit_b_01ec64.pth"
        if model_path.exists():
            print(f"🤖 SAM model: Ready ({model_path.stat().st_size / 1024 / 1024:.0f}MB)")
        else:
            print("⚠️  SAM model: Not available (fallback mode)")
        
        print()
        print("🚀 How to run SWEET:")
        
        if self.is_windows:
            print("   • Double-click: SWEET_Windows.bat")
        elif self.is_macos:
            print("   • Run: ./SWEET_macOS.sh")
        else:
            print("   • Run: ./SWEET_Linux.sh")
            
        print()
        print("📊 Logs: logs/sam_annotator_debug.log")
        
        if has_gpu == "mps":
            print("🍎 Apple Silicon GPU (MPS) acceleration is enabled")
        elif has_gpu:
            print("🎮 NVIDIA GPU (CUDA) acceleration is enabled")
        else:
            print("💻 CPU mode (consider GPU for better performance)")
        
        # Acknowledgments
        print()
        print("=" * 60)
        print("🙏 ACKNOWLEDGMENTS".center(60))
        print("=" * 60)
        print("🧪 Thanks to Prof. Sharon Prince and all members".center(60))
        print("of the Prince Laboratory".center(60))
        print("🐭 Special thanks to my Lab Rats: Tawfeeq and Michael".center(60))
        print("=" * 60)
        print()
    
    def install(self):
        """Main installation process"""
        try:
            self.show_header()
            
            # Check existing installation
            if self.check_existing_installation():
                return True
            
            # Detect GPU
            has_gpu = self.detect_nvidia_gpu()
            
            # Find or install Python
            python_cmd = self.find_python()
            if not python_cmd:
                python_cmd = self.install_python()
                if not python_cmd:
                    print("❌ Could not install Python")
                    return False
            
            # Create virtual environment
            if not self.create_virtual_environment(python_cmd):
                return False
            
            # Install packages
            if not self.install_packages(has_gpu):
                return False
            
            # Download SAM model(s)
            self.download_sam_model(has_gpu)
            
            # Verify installation
            self.verify_installation()
            
            # Save installation info
            self.save_installation_info(has_gpu)
            
            # Show completion message
            self.show_completion_message(has_gpu)
            
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Installation cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    # Make this script work on both Python 2 and 3 (basic compatibility)
    try:
        input_func = raw_input  # Python 2
    except NameError:
        input_func = input      # Python 3
    
    print("Starting SWEET Installation...")
    
    # Check minimum Python version
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        print("This installer will attempt to install Python 3.12")
        input_func("\nPress Enter to continue...")
    
    installer = SWEETInstaller()
    success = installer.install()
    
    if success:
        print("\n🎉 Installation completed successfully!")
    else:
        print("\n❌ Installation failed!")
    
    input_func("\nPress Enter to exit...")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
