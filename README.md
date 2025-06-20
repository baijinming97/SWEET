# SWEET - SAM Widget for Edge Evaluation Tool

🎯 AI-powered image segmentation and area calculation using Segment Anything Model (SAM)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English Version

### 📖 Overview

SWEET is an intelligent tool that helps you:
1. **Segment objects** in images with just a few clicks
2. **Calculate area percentages** of selected regions
3. **Export results** for further analysis

Perfect for:
- 🔬 Scientific research (microscopy analysis)
- 🏗️ Engineering analysis  
- 📸 Image processing
- 📊 Data visualization

### 🚀 Quick Start

#### Installation (One-Click Install)

**Windows:**
- Double-click `install.bat` ✅

**Linux/macOS:**
- Double-click `install.command` ✅

The installer will automatically:
- Detect your system
- Install Python if needed
- Configure GPU acceleration
- Install all dependencies

#### Running SWEET (One-Click Launch)

**Windows:** Double-click `SWEET_Windows.bat` 🚀

**Linux:** Double-click `SWEET_Linux.sh` 🚀

**macOS:** Double-click `SWEET_macOS.command` 🚀

### 📋 Usage Tutorial

#### Step 1: Load Images
- Click **"Load Dir"** button
- Select a folder containing your images
- Images will be loaded automatically

#### Step 2: Annotate Objects
- **Left Click** 🖱️ - Add positive points (green) to mark objects
- **Right Click** 🖱️ - Add negative points (red) to exclude areas
- The annotation count will update in real-time

#### Step 3: Batch Process
- Click **"Start Batch Segmentation"** 🚀
- SWEET will process all images in the folder
- Progress will be shown during processing

#### Step 4: View Results
- **Segmentation Images**: Masked overlay images saved in the same directory
  - Original images with green segmentation masks
  - Use for accuracy verification or paper figures
- **CSV Results**: `segmentation_results.csv` file containing:
  - Image names
  - Coverage percentage (area ratio)
  - Confidence scores
  - Annotation point counts

![alt text](image.png)
#### Example Output
```csv
image_name,coverage_ratio,confidence,positive_points,negative_points
SW_0038_TRANS.tif,17.3879,0.989792,5,2
```

### 🎮 Keyboard Shortcuts
- **Space**: Generate mask
- **S**: Save comparison image
- **A/D**: Previous/Next image
- **C**: Clear annotations

### 💡 Features

- 🎯 **Smart Segmentation** - AI-powered object detection
- 📊 **Area Calculation** - Precise percentage calculations
- 🔥 **GPU Acceleration** - NVIDIA CUDA & Apple Silicon support
- 🖼️ **Batch Processing** - Process entire folders at once
- 📈 **Export Results** - CSV files for data analysis
- 🌐 **Multi-language** - English/Chinese support

### 💻 System Requirements

- **OS:** Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **RAM:** 8GB minimum (16GB recommended)
- **GPU:** NVIDIA GPU with CUDA (optional) or Apple Silicon
- **Storage:** 2-4GB free space
- **Python:** 3.8+ (auto-installed if missing)

### 🔧 Troubleshooting

- 🐛 **Issues?** Check `logs/sam_annotator_debug.log`
- 💡 **GPU not detected?** Install latest NVIDIA/AMD drivers
- 🔧 **Windows error?** Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- 📂 **Permission denied?** Right-click → Run as Administrator

---

<a name="chinese"></a>
## 中文版本

### 📖 概述

SWEET 是一个智能工具，帮助您：
1. **分割图像中的对象** - 只需几次点击
2. **计算区域百分比** - 精确计算选定区域占比
3. **导出分析结果** - 便于进一步研究

适用于：
- 🔬 科学研究（显微镜图像分析）
- 🏗️ 工程分析
- 📸 图像处理
- 📊 数据可视化

### 🚀 快速开始

#### 安装（一键安装）

**Windows系统:**
- 双击 `install.bat` ✅

**Linux/macOS系统:**
- 双击 `install.command` ✅

安装程序将自动：
- 检测您的系统
- 安装所需的Python
- 配置GPU加速
- 安装所有依赖项

#### 运行SWEET（一键启动）

**Windows:** 双击 `SWEET_Windows.bat` 🚀

**Linux:** 双击 `SWEET_Linux.sh` 🚀

**macOS:** 双击 `SWEET_macOS.command` 🚀

### 📋 使用教程

#### 步骤1：加载图像
- 点击 **"Load Dir"（加载目录）** 按钮
- 选择包含图像的文件夹
- 图像将自动加载

#### 步骤2：标注对象
- **鼠标左键** 🖱️ - 添加正向标注点（绿色）标记目标对象
- **鼠标右键** 🖱️ - 添加负向标注点（红色）排除区域
- 标注计数会实时更新显示

#### 步骤3：批量处理
- 点击 **"Start Batch Segmentation"（开始批量分割）** 🚀
- SWEET将处理文件夹中的所有图像
- 处理过程中会显示进度

#### 步骤4：查看结果
- **分割图像**：在同一目录下保存掩码叠加图像
  - 原始图像上叠加绿色分割掩码
  - 可用于准确性验证或论文配图
- **CSV结果**：生成 `segmentation_results.csv` 文件，包含：
  - 图像名称
  - 覆盖百分比（面积比）
  - 置信度分数
  - 标注点数量

#### 输出示例
```csv
image_name,coverage_ratio,confidence,positive_points,negative_points
SW_0038_TRANS.tif,17.3879,0.989792,5,2
```

### 🎮 快捷键
- **空格键**: 生成掩码
- **S**: 保存对比图像
- **A/D**: 上一张/下一张图像
- **C**: 清除标注

### 💡 功能特点

- 🎯 **智能分割** - AI驱动的对象检测
- 📊 **面积计算** - 精确的百分比计算
- 🔥 **GPU加速** - 支持NVIDIA CUDA和Apple Silicon
- 🖼️ **批量处理** - 一次处理整个文件夹
- 📈 **导出结果** - CSV文件便于数据分析
- 🌐 **多语言支持** - 中英文界面

### 💻 系统要求

- **操作系统:** Windows 10+、Ubuntu 18.04+、macOS 10.15+
- **内存:** 最低8GB（推荐16GB）
- **显卡:** NVIDIA GPU（可选）或Apple Silicon
- **存储空间:** 2-4GB可用空间
- **Python:** 3.8+（如果缺失会自动安装）

### 🔧 故障排除

- 🐛 **遇到问题？** 查看 `logs/sam_annotator_debug.log`
- 💡 **GPU未检测到？** 安装最新的NVIDIA/AMD驱动
- 🔧 **Windows错误？** 安装 [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- 📂 **权限拒绝？** 右键 → 以管理员身份运行

### 📁 文件结构

```
SWEET/
├── install.bat          # Windows安装器（双击）
├── install.command      # Linux/macOS安装器（双击）
├── install.py           # 核心安装脚本
├── SWEET_Windows.bat    # Windows启动器（双击）
├── SWEET_Linux.sh       # Linux启动器（双击）
├── SWEET_macOS.command  # macOS启动器（双击）
├── src/                 # 源代码
├── python/              # 虚拟环境（自动创建）
├── models/              # AI模型（自动下载）
└── logs/                # 调试日志
```

### 🚫 无需命令行！

所有操作都可通过双击完成：
- ✅ 双击安装器进行安装
- ✅ 双击启动器运行程序
- ✅ 无需终端或命令提示符

---

**SWEET v1.0** - Making AI segmentation accessible to everyone 🎉