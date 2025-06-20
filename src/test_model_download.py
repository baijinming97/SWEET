#!/usr/bin/env python3
"""
Test script for SAM model download functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from install import SWEETInstaller

def test_model_download():
    """Test the model download functionality"""
    print("Testing SAM model download...")
    
    installer = SWEETInstaller()
    
    # Create a backup if model already exists  
    models_dir = installer.root_dir / "models"
    model_path = models_dir / "sam_vit_b_01ec64.pth"
    backup_path = models_dir / "sam_vit_b_01ec64.pth.backup"
    
    model_existed = False
    if model_path.exists():
        print(f"Model already exists, creating backup...")
        model_path.rename(backup_path)
        model_existed = True
    
    try:
        # Test the download
        result = installer.download_sam_model()
        
        if result:
            print("✅ Model download test passed!")
            
            # Check file size
            if model_path.exists():
                size_mb = model_path.stat().st_size / 1024 / 1024
                print(f"📊 Downloaded file size: {size_mb:.1f}MB")
                
                # Expected size is around 375MB
                if 300 <= size_mb <= 400:
                    print("✅ File size looks correct")
                else:
                    print("⚠️  File size seems unusual for SAM model")
            else:
                print("❌ File was not created")
                
        else:
            print("❌ Model download test failed")
            
    finally:
        # Restore backup if it existed
        if model_existed and backup_path.exists():
            if model_path.exists():
                model_path.unlink()  # Remove downloaded file
            backup_path.rename(model_path)  # Restore original
            print("🔄 Original model file restored")

if __name__ == "__main__":
    test_model_download()