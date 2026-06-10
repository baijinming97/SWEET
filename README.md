# SWEET — SAM Widget for Edge Evaluation Tool

SWEET is an interactive tool for segmenting objects in images and measuring their area, built on Meta's Segment Anything Model (SAM). Mark a few points, batch‑process a folder, and get mask overlays plus a CSV of area percentages. Designed for microscopy such as wound‑healing / cell‑migration (scratch) assays.

English | [中文](#中文)

<div align="center">
  <img src="https://github.com/user-attachments/assets/fb4db1a6-fd7b-4341-8c54-443052f3cc44" width="1000" alt="Download SWEET">
</div>

## Install

1. Download or clone this repository.
2. Run the one‑click installer — it sets up Python, dependencies, and the SAM model automatically:
   - **Windows:** double‑click `install.bat`
   - **macOS / Linux:** double‑click `install.command`

No command line required.

## Run

- **Windows:** `SWEET_Windows.bat`
- **macOS:** `SWEET_macOS.command`
- **Linux:** `SWEET_Linux.sh`

Choose English or Chinese when prompted.

## Usage

**1. Load Dir** — choose a folder of images (`.tif`).

<div align="center">
  <img src="https://github.com/user-attachments/assets/7512258f-545a-4948-ac34-2852ad22bc17" width="1000" alt="Load Images">
</div>

**2. Annotate** — left‑click adds a positive point (green, inside the target); right‑click adds a negative point (red, to exclude). Use **A / D** to move between images and mark each one.

**3. Start Batch Segmentation** — processes every annotated image.

<div align="left">
  <img src="https://github.com/user-attachments/assets/72bc2483-eae7-4e9e-8a72-b8b83e1b557c" width="300" alt="Batch Process">
</div>

**4. Results** — each image gets a `*_segmented.png` overlay, plus a `segmentation_results.csv` (area % + confidence) in the same folder.

<div align="center">
  <img src="https://github.com/user-attachments/assets/9bdce82a-2ed0-4a8e-a3e3-abd3c1021c86" width="420" alt="Result 1">
  <img src="https://github.com/user-attachments/assets/44a46c32-aefe-48af-a5d1-fb2f48a4d142" width="420" alt="Result 2">
  <br>
  <img src="https://github.com/user-attachments/assets/f62ff47d-1449-4ad3-97be-fbf158b9ff45" width="600" alt="CSV Data">
</div>

**Shortcuts:** A / D = previous / next · C = clear points · Space = run batch.

**Advanced Settings** (collapsible panel): contrast boost (CLAHE), fill debris inside the gap, edge/debris smoothing, and red‑point strength. SWEET automatically uses the higher‑quality `vit_l` model on an NVIDIA GPU and the faster `vit_b` model on CPU.

## Features

- Point‑prompted segmentation (SAM) with one‑click batch processing
- Area‑percentage export to CSV
- Boundary‑safe negative points and debris‑aware masks
- Automatic GPU (CUDA / Apple MPS) acceleration, CPU fallback
- Cross‑platform (Windows / macOS / Linux), English & Chinese UI
- Reads non‑ASCII paths and filenames with spaces

## Requirements

Windows 10+ / macOS 10.15+ / Ubuntu 18.04+ · 8 GB RAM (16 GB recommended) · ~3 GB free disk. Python is installed automatically; an NVIDIA GPU is optional. Logs: `logs/sam_annotator.log`.

## Citation

Image segmentation and area calculation were performed using SWEET [1], based on the Segment Anything Model (SAM) [2].

1. *SWEET: SAM Widget for Edge Evaluation Tool*, GitHub repository, 2025. https://github.com/baijinming97/SWEET
2. A. Kirillov et al., "Segment Anything," ICCV 2023, pp. 4015–4026.

See [Releases](https://github.com/baijinming97/SWEET/releases) for version history.

---

<a name="中文"></a>

# 中文

SWEET 是一个交互式图像分割与面积测量工具，基于 Meta 的 Segment Anything Model (SAM)。标注几个点、批量处理整个文件夹，即可得到分割叠加图和面积百分比的 CSV。适用于划痕 / 细胞迁移等显微图像分析。

## 安装

1. 下载或克隆本仓库。
2. 运行一键安装器（自动配置 Python、依赖和 SAM 模型）：
   - **Windows：** 双击 `install.bat`
   - **macOS / Linux：** 双击 `install.command`

无需命令行。

## 运行

- **Windows：** `SWEET_Windows.bat`
- **macOS：** `SWEET_macOS.command`
- **Linux：** `SWEET_Linux.sh`

启动时选择中文或英文。

## 使用

**1. Load Dir** —— 选择图像文件夹（`.tif`）。

<div align="center">
  <img src="https://github.com/user-attachments/assets/7512258f-545a-4948-ac34-2852ad22bc17" width="1000" alt="加载图像">
</div>

**2. 标注** —— 左键加正向点（绿色，标在目标内部），右键加负向点（红色，排除区域）。用 **A / D** 切换图像，逐张标注。

**3. Start Batch Segmentation** —— 对所有已标注图像批量分割。

<div align="left">
  <img src="https://github.com/user-attachments/assets/72bc2483-eae7-4e9e-8a72-b8b83e1b557c" width="300" alt="批量处理">
</div>

**4. 结果** —— 每张图生成 `*_segmented.png` 叠加图，并在同目录输出 `segmentation_results.csv`（面积 % + 置信度）。

<div align="center">
  <img src="https://github.com/user-attachments/assets/9bdce82a-2ed0-4a8e-a3e3-abd3c1021c86" width="420" alt="结果 1">
  <img src="https://github.com/user-attachments/assets/44a46c32-aefe-48af-a5d1-fb2f48a4d142" width="420" alt="结果 2">
  <br>
  <img src="https://github.com/user-attachments/assets/f62ff47d-1449-4ad3-97be-fbf158b9ff45" width="600" alt="CSV 数据">
</div>

**快捷键：** A / D = 上一张 / 下一张 · C = 清除标注 · Space = 批量分割。

**高级设置**（可折叠面板）：对比增强 (CLAHE)、填充裂缝内碎片、边缘 / 碎片平滑、红点强度。SWEET 会在 NVIDIA GPU 上自动使用更高质量的 `vit_l`，无 GPU 时使用更快的 `vit_b`。

## 功能

- 基于点提示的 SAM 分割，一键批量处理
- 面积百分比导出 CSV
- 边界安全的负向点、碎片感知的分割
- 自动 GPU（CUDA / Apple MPS）加速，无 GPU 时回退 CPU
- 跨平台（Windows / macOS / Linux），中英文界面
- 支持非 ASCII 路径与含空格文件名

## 系统要求

Windows 10+ / macOS 10.15+ / Ubuntu 18.04+ · 内存 8 GB（推荐 16 GB）· 可用磁盘约 3 GB。Python 自动安装；NVIDIA GPU 可选。日志：`logs/sam_annotator.log`。

## 引用

图像分割和面积计算使用 SWEET [1] 完成，基于 Segment Anything Model (SAM) [2]。

1. *SWEET: SAM Widget for Edge Evaluation Tool*, GitHub, 2025. https://github.com/baijinming97/SWEET
2. A. Kirillov et al., "Segment Anything," ICCV 2023, pp. 4015–4026.

版本历史见 [Releases](https://github.com/baijinming97/SWEET/releases)。
