#!/bin/bash

# Linux SWEET Launcher  
# SAM Widget for Edge Evaluation Tool

clear
echo "============================================================"
echo "              SWEET - Linux Edition"
echo "      SAM Widget for Edge Evaluation Tool"
echo "============================================================"
echo

# Change to SWEET directory
cd "$(dirname "$0")"

# Check if environment exists
if [ ! -f "python/bin/python" ]; then
    echo "❌ Python environment not found!"
    echo
    echo "Please run setup first:"
    echo "   python3 install"
    read -p "Press Enter to exit..."
    exit 1
fi

# Environment info
echo "🐧 Platform: Linux"
python/bin/python -c "
import torch
print('🔥 PyTorch:', torch.__version__)
print('⚡ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🎮 GPU count:', torch.cuda.device_count())
    print('🎮 GPU name:', torch.cuda.get_device_name(0))
else:
    print('💻 CPU cores:', torch.get_num_threads())
"
echo

# Language selection
echo "Please select language / 请选择语言:"
echo
echo "[1] 🇺🇸 English"
echo "[2] 🇨🇳 中文"
echo

read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo
        echo "🚀 Starting SWEET in English..."
        python/bin/python src/sam_annotator_english.py
        ;;
    2)
        echo
        echo "🚀 启动中文版SWEET..."
        python/bin/python src/sam_annotator_chinese.py
        ;;
    *)
        echo
        echo "❌ Invalid choice. Defaulting to English..."
        python/bin/python src/sam_annotator_english.py
        ;;
esac

echo
echo "📊 Check logs/sam_annotator.log for performance details"
read -p "Press Enter to exit..."