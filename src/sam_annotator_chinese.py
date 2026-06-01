import sys
import os
import json
import numpy as np
import time
import logging
from datetime import datetime
from qt_bootstrap import configure_qt_plugins

configure_qt_plugins()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QProgressBar, QProgressDialog, QDialog)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QKeyEvent
import cv2
import pandas as pd
import glob
from PIL import Image, ImageDraw, ImageFont

# 确保日志目录存在
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置详细日志
log_file = os.path.join(log_dir, 'sam_annotator.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchSegmentationDialog(QDialog):
    """批量分割进度对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量分割进度")
        self.setFixedSize(500, 200)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # 进度信息
        self.status_label = QLabel("准备开始...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #34495e;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 时间信息
        self.time_label = QLabel("估计剩余时间: 计算中...")
        self.time_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(self.time_label)
        
        # 当前处理的图片
        self.current_image_label = QLabel("当前图片: 无")
        self.current_image_label.setStyleSheet("font-size: 12px; color: #2c3e50;")
        layout.addWidget(self.current_image_label)
        
        # 取消按钮
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; color: white; border: none;
                padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
        self.cancelled = False
        self.cancel_button.clicked.connect(self.cancel)
        
    def cancel(self):
        self.cancelled = True
        self.reject()
        
    def update_progress(self, current, total, current_image, elapsed_time):
        """更新进度"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        percentage = (current / total * 100) if total > 0 else 0
        self.status_label.setText(f"正在处理... {current}/{total} ({percentage:.1f}%)")
        self.current_image_label.setText(f"当前图片: {os.path.basename(current_image)}")
        
        # 估算剩余时间
        if current > 0 and elapsed_time > 0:
            avg_time_per_image = elapsed_time / current
            remaining_images = total - current
            estimated_remaining = avg_time_per_image * remaining_images
            
            minutes = int(estimated_remaining // 60)
            seconds = int(estimated_remaining % 60)
            self.time_label.setText(f"估计剩余时间: {minutes}:{seconds:02d}")
        else:
            self.time_label.setText("估计剩余时间: 计算中...")

# Import SAM dependencies from utils
try:
    logger.info("开始加载SAM依赖...")
    start_time = time.time()
    from utils import sam_model_registry, SamPredictor, largest_component, label_to_color_image
    import torch
    SAM_AVAILABLE = True
    logger.info(f"SAM依赖加载成功，耗时: {time.time() - start_time:.2f}秒")
except ImportError as e:
    SAM_AVAILABLE = False
    logger.error(f"SAM不可用: {e}")


class SAMModelWrapper:
    """优化的SAM模型包装器，支持预加载"""
    _instance = None
    
    def __new__(cls, model_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path=None):
        if self._initialized:
            return
            
        logger.info("初始化SAM模型...")
        start_time = time.time()
        
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_image_hash = None
        self.current_image_path = None  # 当前已设置的图像路径
        # 默认使用压缩模式 (512px)，可通过precise_mode参数控制
        self.max_dimension = 512
        self.precise_mode = False
        
        if model_path and SAM_AVAILABLE and os.path.exists(model_path):
            try:
                logger.info(f"从 {model_path} 加载SAM模型")
                logger.info(f"设备: {self.device} (CUDA可用: {torch.cuda.is_available()})")
                logger.info(f"图像处理尺寸: {self.max_dimension}px (精确模式: {self.precise_mode})")
                if self.device == "cpu":
                    logger.warning("使用CPU运行SAM会较慢，建议使用压缩模式")
                    logger.info("如需更好质量可安装CUDA版本PyTorch")
                
                model_type = "_".join(model_path.split("_")[1:3])
                sam = sam_model_registry[model_type](checkpoint=model_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                
                load_time = time.time() - start_time
                logger.info(f"SAM模型加载成功，耗时: {load_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"SAM模型加载失败: {e}")
                self.predictor = None
        else:
            logger.warning("SAM模型文件未找到或依赖缺失")
            
        self._initialized = True
            
    def set_image(self, image):
        if self.predictor:
            try:
                # 快速哈希：使用图像shape和部分像素数据
                hash_start = time.time()
                # 使用图像的形状和四个角的像素值创建快速哈希
                quick_hash = (
                    image.shape,
                    tuple(image[0, 0]),  # 左上角
                    tuple(image[0, -1]),  # 右上角
                    tuple(image[-1, 0]),  # 左下角
                    tuple(image[-1, -1]),  # 右下角
                    image[image.shape[0]//2, image.shape[1]//2].tobytes()  # 中心点
                )
                image_hash = hash(quick_hash)
                hash_time = time.time() - hash_start
                logger.debug(f"图像哈希计算耗时: {hash_time:.3f}秒")
                
                if self.current_image_hash != image_hash:
                    # 检查是否需要缩放图像
                    h, w = image.shape[:2]
                    self.original_size = (h, w)
                    self.scale_factor = 1.0
                    
                    # 确保图像具有正确的数据类型
                    if image.dtype != np.uint8:
                        logger.debug(f"转换图像数据类型从 {image.dtype} 到 uint8")
                        image = image.astype(np.uint8)
                    
                    # 确保图像具有正确的形状 (H, W, 3)
                    if len(image.shape) != 3 or image.shape[2] != 3:
                        logger.error(f"无效的图像形状: {image.shape}, 期望 (H, W, 3)")
                        return False
                    
                    # 根据精确模式决定是否缩放
                    if self.precise_mode or self.max_dimension is None:
                        # 精确模式：使用原图
                        sam_start = time.time()
                        self.predictor.set_image(image)
                        sam_time = time.time() - sam_start
                        logger.debug(f"SAM设置原始图像耗时: {sam_time:.3f}秒 (精确模式)")
                    elif max(h, w) > self.max_dimension:
                        # 压缩模式：缩放图像
                        self.scale_factor = self.max_dimension / max(h, w)
                        new_h = int(h * self.scale_factor)
                        new_w = int(w * self.scale_factor)
                        
                        # 缩放图像
                        resize_start = time.time()
                        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        resize_time = time.time() - resize_start
                        logger.debug(f"图像缩放耗时: {resize_time:.3f}秒 ({w}x{h} -> {new_w}x{new_h})")
                        
                        # 确保缩放后的图像具有正确的数据类型
                        if resized_image.dtype != np.uint8:
                            logger.debug(f"转换缩放图像数据类型从 {resized_image.dtype} 到 uint8")
                            resized_image = resized_image.astype(np.uint8)
                        
                        # 使用缩放后的图像
                        sam_start = time.time()
                        self.predictor.set_image(resized_image)
                        sam_time = time.time() - sam_start
                        logger.debug(f"SAM设置缩放图像耗时: {sam_time:.3f}秒 (压缩模式)")
                    else:
                        # 图像已经足够小，直接使用
                        sam_start = time.time()
                        self.predictor.set_image(image)
                        sam_time = time.time() - sam_start
                        logger.debug(f"SAM设置原始图像耗时: {sam_time:.3f}秒 (无需缩放)")
                    
                    self.current_image_hash = image_hash
                else:
                    logger.debug("跳过SAM图像设置（相同图像）")
                return True
            except Exception as e:
                logger.error(f"SAM设置图像错误: {e}")
                return False
        return False
        
    def predict(self, positive_points, negative_points):
        if not self.predictor:
            return None, 0.0
            
        try:
            start_time = time.time()
            
            all_points = positive_points + negative_points
            labels = [1] * len(positive_points) + [0] * len(negative_points)
            
            if len(all_points) == 0:
                return None, 0.0
            
            # 如果图像被缩放了，需要缩放点坐标（仅在压缩模式下）
            if hasattr(self, 'scale_factor') and self.scale_factor != 1.0 and not self.precise_mode:
                scaled_points = [(int(x * self.scale_factor), int(y * self.scale_factor)) 
                                for x, y in all_points]
                input_points = np.array(scaled_points)
            else:
                input_points = np.array(all_points)
            
            input_labels = np.array(labels)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            # 应用最大连通组件过滤
            if SAM_AVAILABLE:
                try:
                    best_mask = largest_component(best_mask).astype(bool)
                except Exception as e:
                    logger.warning(f"最大连通组件处理失败: {e}")
            
            # 如果图像被缩放了，需要将mask放大回原始尺寸（仅在压缩模式下）
            if hasattr(self, 'scale_factor') and self.scale_factor != 1.0 and not self.precise_mode and hasattr(self, 'original_size'):
                resize_start = time.time()
                h, w = self.original_size
                best_mask_resized = cv2.resize(best_mask.astype(np.uint8), (w, h), 
                                              interpolation=cv2.INTER_NEAREST)
                resize_time = time.time() - resize_start
                logger.debug(f"掩码放大耗时: {resize_time:.3f}秒")
                best_mask = best_mask_resized.astype(bool)
            
            predict_time = time.time() - start_time
            logger.info(f"SAM预测完成，耗时: {predict_time:.3f}秒，置信度: {best_score:.3f}")
            
            return best_mask.astype(np.uint8) * 255, float(best_score)
            
        except Exception as e:
            logger.error(f"SAM预测错误: {e}")
            return None, 0.0


class FastImageLabel(QLabel):
    pointsUpdated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        logger.info("初始化图像标签组件...")
        
        self.original_image = None
        self.cached_base_pixmap = None
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.image_rect = None
        self.sam_predictor = None
        self.show_mask = False
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.last_image_path = None
        
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("border: 3px solid #2c3e50; background-color: #ecf0f1; border-radius: 5px;")
        # 不设置焦点策略，让主窗口处理键盘事件
        self.setFocusPolicy(Qt.NoFocus)
        
        # 初始化SAM
        # SAM模型延迟加载，仅在批量分割时初始化
        self.sam_model_path = "models/sam_vit_b_01ec64.pth"
        self.sam_predictor = None
        logger.info("SAM模型将在批量分割时加载")
        
    def set_image(self, image_path):
        """优化的图像加载"""
        logger.info(f"开始加载图像: {image_path}")
        start_time = time.time()
        
        # 注释掉缓存检查，确保每次都重新加载图像
        # 这解决了切换图片需要按两次键的问题
        # if self.last_image_path == image_path:
        #     logger.debug("跳过加载（相同图像路径）")
        #     return True
            
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return False
            
        # 加载图像
        try:
            load_start = time.time()
            self.original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if self.original_image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False
            
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            load_time = time.time() - load_start
            logger.debug(f"OpenCV图像读取耗时: {load_time:.3f}秒")
            
        except Exception as e:
            logger.error(f"图像加载异常: {e}")
            return False
        
        self.last_image_path = image_path
        
        # 跳过SAM图像设置，仅在批量分割时处理
        logger.debug("跳过SAM图像设置，将在批量分割时处理")
        
        # 重置状态
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.show_mask = False
        self.cached_base_pixmap = None
        
        # 更新显示
        display_start = time.time()
        self.update_display_fast()
        display_time = time.time() - display_start
        logger.debug(f"显示更新耗时: {display_time:.3f}秒")
        
        total_time = time.time() - start_time
        logger.info(f"图像加载完成，总耗时: {total_time:.3f}秒")
        return True
        
    def update_display_fast(self):
        """超快显示更新"""
        if self.original_image is None:
            return
            
        start_time = time.time()
        
        # 需要时创建基础像素图
        if self.cached_base_pixmap is None:
            self.create_base_pixmap()
            
        # 快速点覆盖
        self.draw_points_overlay()
        
        update_time = time.time() - start_time
        logger.debug(f"显示更新耗时: {update_time:.3f}秒")
        
    def create_base_pixmap(self):
        """创建缓存的基础像素图"""
        start_time = time.time()
        
        display_image = self.original_image.copy()
        
        # 应用掩码覆盖（如果存在）
        if self.show_mask and self.current_mask is not None:
            try:
                mask_start = time.time()
                if SAM_AVAILABLE:
                    mask_binary = (self.current_mask > 0).astype(np.uint8)
                    color_mask = label_to_color_image(mask_binary)
                    if color_mask.size > 0:
                        display_image = cv2.addWeighted(display_image, 0.7, color_mask, 0.3, 0.0)
                else:
                    # 后备方案
                    overlay_color = np.zeros_like(display_image)
                    overlay_color[self.current_mask > 0] = [0, 255, 0]
                    display_image = cv2.addWeighted(display_image, 0.7, overlay_color, 0.3, 0)
                
                mask_time = time.time() - mask_start
                logger.debug(f"掩码覆盖耗时: {mask_time:.3f}秒")
            except Exception as e:
                logger.error(f"掩码覆盖错误: {e}")
        
        # 转换为QPixmap
        try:
            convert_start = time.time()
            height, width, channel = display_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            convert_time = time.time() - convert_start
            logger.debug(f"QPixmap转换耗时: {convert_time:.3f}秒")
        except Exception as e:
            logger.error(f"QPixmap转换错误: {e}")
            return
        
        # 缩放以适应
        try:
            scale_start = time.time()
            widget_size = self.size()
            self.cached_base_pixmap = pixmap.scaled(widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scale_time = time.time() - scale_start
            logger.debug(f"像素图缩放耗时: {scale_time:.3f}秒")
        except Exception as e:
            logger.error(f"像素图缩放错误: {e}")
            return
        
        # 更新图像矩形和缩放比例
        x_offset = (widget_size.width() - self.cached_base_pixmap.width()) // 2
        y_offset = (widget_size.height() - self.cached_base_pixmap.height()) // 2
        self.image_rect = (x_offset, y_offset, self.cached_base_pixmap.width(), self.cached_base_pixmap.height())
        
        self.scale_x = self.cached_base_pixmap.width() / width
        self.scale_y = self.cached_base_pixmap.height() / height
        
        create_time = time.time() - start_time
        logger.debug(f"基础像素图创建耗时: {create_time:.3f}秒")
        
    def draw_points_overlay(self):
        """快速点绘制"""
        if self.cached_base_pixmap is None:
            return
            
        start_time = time.time()
        
        final_pixmap = QPixmap(self.cached_base_pixmap)
        painter = QPainter(final_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制正向点
        painter.setPen(QPen(QColor(0, 255, 0), 4))
        painter.setBrush(QColor(0, 255, 0, 180))
        for point in self.positive_points:
            x = int(point[0] * self.scale_x)
            y = int(point[1] * self.scale_y)
            painter.drawEllipse(QPoint(x, y), 8, 8)
            
        # 绘制负向点
        painter.setPen(QPen(QColor(255, 0, 0), 4))
        painter.setBrush(QColor(255, 0, 0, 180))
        for point in self.negative_points:
            x = int(point[0] * self.scale_x)
            y = int(point[1] * self.scale_y)
            painter.drawEllipse(QPoint(x, y), 8, 8)
            
        painter.end()
        self.setPixmap(final_pixmap)
        
        draw_time = time.time() - start_time
        logger.debug(f"点绘制耗时: {draw_time:.3f}秒")
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        logger.debug("窗口大小改变，清除缓存")
        self.cached_base_pixmap = None
        if self.original_image is not None:
            self.update_display_fast()
        
    def mousePressEvent(self, event):
        if self.original_image is None or self.image_rect is None:
            return
            
        start_time = time.time()
        
        # 快速坐标转换
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        img_x, img_y, img_w, img_h = self.image_rect
        if (click_x < img_x or click_x >= img_x + img_w or 
            click_y < img_y or click_y >= img_y + img_h):
            return
            
        relative_x = click_x - img_x
        relative_y = click_y - img_y
        
        orig_x = int(relative_x / self.scale_x)
        orig_y = int(relative_y / self.scale_y)
        
        orig_x = max(0, min(orig_x, self.original_image.shape[1] - 1))
        orig_y = max(0, min(orig_y, self.original_image.shape[0] - 1))
        
        # 检查是否点击在现有点附近（取消功能）
        click_radius = 15  # 点击半径
        point_removed = False
        
        if event.button() == Qt.LeftButton:
            # 左键：检查是否点击红点（取消红点）
            for i, (nx, ny) in enumerate(self.negative_points):
                if abs(orig_x - nx) < click_radius and abs(orig_y - ny) < click_radius:
                    self.negative_points.pop(i)
                    logger.info(f"取消负向点: ({nx}, {ny})")
                    point_removed = True
                    break
            
            # 如果没有取消点，则添加绿点
            if not point_removed:
                self.positive_points.append([orig_x, orig_y])
                logger.info(f"添加正向点: ({orig_x}, {orig_y})")
                
        elif event.button() == Qt.RightButton:
            # 右键：检查是否点击绿点（取消绿点）
            for i, (px, py) in enumerate(self.positive_points):
                if abs(orig_x - px) < click_radius and abs(orig_y - py) < click_radius:
                    self.positive_points.pop(i)
                    logger.info(f"取消正向点: ({px}, {py})")
                    point_removed = True
                    break
            
            # 如果没有取消点，则添加红点
            if not point_removed:
                self.negative_points.append([orig_x, orig_y])
                logger.info(f"添加负向点: ({orig_x}, {orig_y})")
        
        # 快速更新 - 只重绘点
        self.draw_points_overlay()
        self.pointsUpdated.emit()
        
        # 移除强制焦点设置，避免双击问题
        # 主窗口已配置StrongFocus，不需要每次点击后重新设置焦点
        
        click_time = time.time() - start_time
        logger.debug(f"鼠标点击处理耗时: {click_time:.3f}秒")
        
    def generate_sam_mask(self):
        if len(self.positive_points) == 0:
            return False
            
        logger.info("开始生成SAM掩码...")
        start_time = time.time()
        
        if self.sam_predictor and self.sam_predictor.predictor:
            mask, score = self.sam_predictor.predict(self.positive_points, self.negative_points)
            if mask is not None:
                self.current_mask = mask
                self.show_mask = True
                self.cached_base_pixmap = None  # 强制重新生成
                
                total_time = time.time() - start_time
                logger.info(f"SAM掩码生成成功，总耗时: {total_time:.3f}秒，置信度: {score:.3f}")
                
                self.update_display_fast()
                return True
        
        # 后备方案
        logger.info("使用后备掩码生成方法...")
        self.generate_fallback_mask()
        return True
            
    def generate_fallback_mask(self):
        start_time = time.time()
        
        h, w = self.original_image.shape[:2]
        self.current_mask = np.zeros((h, w), dtype=np.uint8)
        
        for px, py in self.positive_points:
            cv2.circle(self.current_mask, (px, py), 50, 255, -1)
        
        for nx, ny in self.negative_points:
            cv2.circle(self.current_mask, (nx, ny), 40, 0, -1)
        
        self.current_mask = cv2.GaussianBlur(self.current_mask, (11, 11), 0)
        self.current_mask = (self.current_mask > 128).astype(np.uint8) * 255
        self.show_mask = True
        self.cached_base_pixmap = None
        
        fallback_time = time.time() - start_time
        logger.info(f"后备掩码生成完成，耗时: {fallback_time:.3f}秒")
        
        self.update_display_fast()
            
    def clear_points(self):
        logger.info("清除所有标记点")
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.show_mask = False
        self.cached_base_pixmap = None
        self.update_display_fast()

    def get_mask_coverage(self):
        if self.current_mask is not None:
            total_pixels = self.current_mask.shape[0] * self.current_mask.shape[1]
            mask_pixels = np.sum(self.current_mask > 0)
            coverage = (mask_pixels / total_pixels) * 100
            logger.debug(f"掩码覆盖率: {coverage:.2f}%")
            return coverage
        return 0.0


class WoundAnnotatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("初始化主窗口...")
        
        self.setWindowTitle("SWEET - SAM Widget for Edge Evaluation Tool")
        self.setGeometry(50, 50, 1500, 900)
        
        # 图像管理 - 单张加载模式
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        self.results_data = {}
        self.image_annotations = {}  # 存储所有图片的标记点 {image_path: {'positive': [...], 'negative': [...]}}
        self.data_directory = None  # 保存用户选择的数据目录
        
        # 状态消息计时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.clear_status_message)
        
        self.init_ui()
        
        # 确保主窗口能接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 设置所有按钮不获取焦点，让主窗口处理键盘事件
        for widget in self.findChildren(QPushButton):
            widget.setFocusPolicy(Qt.NoFocus)
        
        self.setFocus()
        
        logger.info("主窗口初始化完成")
        
        # 界面初始化完成
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        
        # 左面板 - 图像显示
        self.image_label = FastImageLabel()
        self.image_label.main_window = self  # 保存主窗口引用
        self.image_label.pointsUpdated.connect(self.on_points_updated)
        main_layout.addWidget(self.image_label, 4)
        
        # 右面板 - 控件
        right_panel = QVBoxLayout()
        
        # 标题 - 已移除
        # title = QLabel("SWEET")
        # title.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("""
        #     font-size: 20px; font-weight: bold; padding: 15px; 
        #     background-color: #3498db; color: white; border-radius: 8px;
        # """)
        # right_panel.addWidget(title)
        
        # 状态消息
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            background-color: #27ae60; color: white; font-weight: bold;
            padding: 8px; border-radius: 4px; margin: 5px;
            font-size: 12px;
        """)
        self.status_label.setVisible(False)
        right_panel.addWidget(self.status_label)
        
        # 信息显示
        self.info_label = QLabel("选择目录开始标注")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            background-color: #ecf0f1; border: 2px solid #34495e; border-radius: 8px;
            padding: 15px; margin: 5px; font-size: 13px; color: #2c3e50;
        """)
        right_panel.addWidget(self.info_label)
        
        # 模型状态 - 已移除
        # self.model_status = QLabel()
        # self.update_model_status()
        # right_panel.addWidget(self.model_status)
        
        # 按钮样式 - 深色背景，白色文字，高对比度
        button_style = """
            QPushButton {
                font-size: 13px; font-weight: bold; padding: 10px; margin: 3px;
                border: 2px solid; border-radius: 8px; color: white;
                background-color: #2c3e50; border-color: #34495e;
            }
            QPushButton:hover { 
                background-color: #34495e; opacity: 0.9; 
            }
            QPushButton:disabled {
                background-color: #95a5a6; border-color: #7f8c8d; color: #7f8c8d;
            }
        """
        
        # 加载按钮
        load_layout = QHBoxLayout()
        
        self.load_dir_button = QPushButton("📁 加载目录")
        self.load_dir_button.clicked.connect(self.load_directory)
        self.load_dir_button.setStyleSheet(button_style + """
            background-color: #27ae60; border-color: #229954;
        """)
        load_layout.addWidget(self.load_dir_button)
        
        # 精确模式开关（默认OFF，使用压缩模式）
        self.precise_mode_button = QPushButton("🎯 精确模式: OFF")
        self.precise_mode_button.clicked.connect(self.toggle_precise_mode)
        self.precise_mode_button.setStyleSheet(button_style + """
            background-color: #95a5a6; border-color: #7f8c8d;
        """)
        load_layout.addWidget(self.precise_mode_button)
        
        # 默认使用压缩模式
        self.precise_mode = False
        
        right_panel.addLayout(load_layout)
        
        # 导航
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ 上一张 (A)")
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setEnabled(False)
        self.prev_button.setStyleSheet(button_style)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("下一张 (D) ▶")
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setEnabled(False)
        self.next_button.setStyleSheet(button_style)
        nav_layout.addWidget(self.next_button)
        right_panel.addLayout(nav_layout)
        
        # 标记状态显示
        self.mark_status = QLabel("📍 状态: 未开始")
        self.mark_status.setAlignment(Qt.AlignCenter)
        self.mark_status.setStyleSheet("""
            font-size: 14px; font-weight: bold; padding: 8px; margin: 3px;
            background-color: #34495e; color: white; border-radius: 5px;
        """)
        right_panel.addWidget(self.mark_status)
        
        # 清除按钮
        self.clear_button = QPushButton("🗑️ 清除标记 (C)")
        self.clear_button.clicked.connect(self.clear_points)
        self.clear_button.setEnabled(False)
        self.clear_button.setStyleSheet(button_style + """
            background-color: #95a5a6; border-color: #7f8c8d;
        """)
        right_panel.addWidget(self.clear_button)
        
        # 批量分割按钮
        self.batch_segment_button = QPushButton("🚀 开始批量分割")
        self.batch_segment_button.clicked.connect(self.start_batch_segmentation)
        self.batch_segment_button.setEnabled(False)
        self.batch_segment_button.setStyleSheet(button_style + """
            background-color: #e74c3c; border-color: #c0392b;
            font-size: 16px; padding: 15px;
        """)
        right_panel.addWidget(self.batch_segment_button)
        
        right_panel.addStretch()
        
        # 添加右面板
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(380)
        main_layout.addWidget(right_widget)
        
    def keyPressEvent(self, event):
        # 按键事件处理
        if event.key() == Qt.Key_A:
            # 处理A键
            self.prev_image()
            event.accept()  # 明确接受事件，防止传播
        elif event.key() == Qt.Key_D:
            # 处理D键
            self.next_image()
            event.accept()  # 明确接受事件，防止传播
        elif event.key() == Qt.Key_C:
            self.clear_points()
            event.accept()
        elif event.key() == Qt.Key_Space:
            # 触发批量分割
            if self.batch_segment_button.isEnabled():
                self.start_batch_segmentation()
            event.accept()
        else:
            super().keyPressEvent(event)
        
    def update_model_status(self):
        # 模型状态显示已移除
        pass
    
    def toggle_precise_mode(self):
        """切换精确模式"""
        if self.precise_mode:
            # 切换到压缩模式
            self.precise_mode = False
            self.precise_mode_button.setText("🎯 精确模式: OFF")
            self.precise_mode_button.setStyleSheet("""
                QPushButton {
                    font-size: 13px; font-weight: bold; padding: 10px; margin: 3px;
                    border: 2px solid; border-radius: 8px; color: white;
                    background-color: #95a5a6; border-color: #7f8c8d;
                }
            """)
            # 切换到压缩模式
        else:
            # 切换到精确模式
            self.precise_mode = True
            self.precise_mode_button.setText("🎯 精确模式: ON")
            self.precise_mode_button.setStyleSheet("""
                QPushButton {
                    font-size: 13px; font-weight: bold; padding: 10px; margin: 3px;
                    border: 2px solid; border-radius: 8px; color: white;
                    background-color: #e74c3c; border-color: #c0392b;
                }
            """)
            # 切换到精确模式
    
    def auto_save_current_points(self):
        """自动保存当前图片的标记点（静默保存，无提示）"""
        if not self.current_image_path:
            return
            
        if self.image_label.positive_points or self.image_label.negative_points:
            self.image_annotations[self.current_image_path] = {
                'positive': self.image_label.positive_points.copy(),
                'negative': self.image_label.negative_points.copy()
            }
            
            num_points = len(self.image_label.positive_points) + len(self.image_label.negative_points)
            # 自动保存标记
        else:
            # 如果没有点，移除该图片的标记
            if self.current_image_path in self.image_annotations:
                del self.image_annotations[self.current_image_path]
                logger.debug(f"移除标记: {os.path.basename(self.current_image_path)}")
        
        # 更新状态（静默更新，无消息提示）
        self.update_mark_status()
    
    def save_current_points(self):
        """保存当前图片的标记点"""
        if not self.current_image_path:
            return
            
        if self.image_label.positive_points or self.image_label.negative_points:
            self.image_annotations[self.current_image_path] = {
                'positive': self.image_label.positive_points.copy(),
                'negative': self.image_label.negative_points.copy()
            }
            
            num_points = len(self.image_label.positive_points) + len(self.image_label.negative_points)
            logger.info(f"保存标记: {os.path.basename(self.current_image_path)} - {num_points}个点")
            self.show_status_message(f"✅ 已保存 {num_points} 个标记点", 2000)
            self.update_mark_status()
        else:
            # 如果没有点，移除该图片的标记
            if self.current_image_path in self.image_annotations:
                del self.image_annotations[self.current_image_path]
            self.show_status_message("⚠️ 没有标记点可保存", 2000)
            self.update_mark_status()
    
    def update_mark_status(self):
        """更新标记状态显示"""
        total_images = len(self.image_list) if self.image_list else 0
        marked_images = len(self.image_annotations)
        
        if total_images == 0:
            status_text = "📍 状态: 未加载"
            color = "#95a5a6"
        elif marked_images == 0:
            status_text = f"📍 状态: 0/{total_images}"
            color = "#e74c3c"
        elif marked_images == total_images:
            status_text = f"📍 状态: {marked_images}/{total_images} ✓"
            color = "#27ae60"
        else:
            status_text = f"📍 状态: {marked_images}/{total_images}"
            color = "#f39c12"
        
        self.mark_status.setText(status_text)
        self.mark_status.setStyleSheet(f"""
            font-size: 13px; font-weight: bold; padding: 8px; margin: 3px;
            background-color: {color}; color: white; border-radius: 5px;
        """)
        
        # 更新批量分割按钮状态
        self.batch_segment_button.setEnabled(marked_images > 0)
        
    def load_directory(self):
        """加载目录"""
        # 检查是否有未分割的标记图片
        if self.image_annotations:
            # 检查是否已经进行过分割
            has_segmented = any(
                os.path.exists(img_path.replace('.tif', '_segmented.png'))
                for img_path in self.image_annotations.keys()
            )
            
            if not has_segmented:
                # 还没有进行分割，提醒用户
                reply = QMessageBox.warning(
                    self, "警告", 
                    f"您有 {len(self.image_annotations)} 张已标记但未分割的图片。\n\n"
                    "切换目录将丢失这些标记。是否继续？",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    return
        
        # 用户选择加载目录
        default_dir = os.path.join(os.getcwd(), "data")
        directory = QFileDialog.getExistingDirectory(self, "选择目录", default_dir)
        if not directory:
            return
            
        # 清除之前的标记，防止串扰
        self.image_annotations.clear()
        self.results_data.clear()
        
        # 保存数据目录路径
        self.data_directory = directory
        logger.info(f"选择了目录: {directory}")
        start_time = time.time()
        
        # 快速查找图片文件
        self.show_status_message("🔍 正在扫描目录...", persistent=True)
        QApplication.processEvents()
        
        image_extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']  # 只加载原始图片，不加载对比图
        image_list = []
        
        # 使用set来避免重复文件
        unique_files = set()
        for ext in image_extensions:
            found_files = glob.glob(os.path.join(directory, ext))
            # 过滤掉对比图片并添加到set中避免重复
            for file in found_files:
                if 'comparison' not in os.path.basename(file).lower():
                    unique_files.add(file)
        
        # 转换为列表
        image_list = list(unique_files)
        
        image_list.sort()
        scan_time = time.time() - start_time
        logger.info(f"目录扫描完成，找到 {len(image_list)} 张图片，耗时: {scan_time:.3f}秒")
        
        if image_list:
            self.image_list = image_list
            self.current_index = 0
            
            # 只加载第一张图片
            load_start = time.time()
            self.load_current_image()
            load_time = time.time() - load_start
            logger.info(f"首张图片加载耗时: {load_time:.3f}秒")
            
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            # 标记加载完成后可进行批量分割
            pass
            
            total_time = time.time() - start_time
            self.show_status_message(f"✅ 目录加载完成：{len(image_list)} 张图片（总耗时: {total_time:.2f}秒）", 4000)
        else:
            self.show_status_message("❌ 目录中未找到图片文件", 3000)
            
    def load_current_image(self):
        """加载当前图片"""
        if 0 <= self.current_index < len(self.image_list):
            logger.info(f"加载图片 {self.current_index + 1}/{len(self.image_list)}")
            start_time = time.time()
            
            # 清除缓存的路径，确保图片切换时会重新加载
            self.image_label.last_image_path = None
            self.current_image_path = self.image_list[self.current_index]
            
            if self.image_label.set_image(self.current_image_path):
                self.clear_button.setEnabled(True)
                # 自动保存，无需手动保存按钮
                
                # 加载之前保存的标记点
                if self.current_image_path in self.image_annotations:
                    annotation = self.image_annotations[self.current_image_path]
                    self.image_label.positive_points = annotation['positive'].copy()
                    self.image_label.negative_points = annotation['negative'].copy()
                    self.image_label.update_display_fast()
                    
                    num_points = len(annotation['positive']) + len(annotation['negative'])
                    logger.info(f"加载已保存的标记: {num_points}个点")
                
                self.update_info()
                self.update_mark_status()
                
                load_time = time.time() - start_time
                image_name = os.path.basename(self.current_image_path)
                logger.info(f"图片加载完成: {image_name}，耗时: {load_time:.3f}秒")
                
                # 更新导航按钮状态
                self.prev_button.setEnabled(self.current_index > 0)
                self.next_button.setEnabled(self.current_index < len(self.image_list) - 1)
                
    def prev_image(self):
        if self.current_index > 0:
            logger.info("切换到上一张图片")
            # 保存当前图片的标记点
            self.auto_save_current_points()
            self.current_index -= 1
            self.load_current_image()
            # 确保窗口保持焦点
            self.setFocus()
            
    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            logger.info("切换到下一张图片")
            # 保存当前图片的标记点
            self.auto_save_current_points()
            self.current_index += 1
            self.load_current_image()
            # 确保窗口保持焦点
            self.setFocus()
    
    def start_batch_segmentation(self):
        """开始批量分割处理"""
        if not self.image_annotations:
            QMessageBox.warning(self, "警告", "没有标记的图片可以分割！")
            return
            
        reply = QMessageBox.question(
            self, "确认批量分割", 
            f"将对 {len(self.image_annotations)} 张已标记的图片进行分割处理。\n\n"
            "这个过程可能需要较长时间，是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # 初始化SAM模型（如果还没有）
        if self.image_label.sam_predictor is None:
            logger.info("正在加载SAM模型...")
            QApplication.processEvents()
            
            try:
                self.image_label.sam_predictor = SAMModelWrapper(self.image_label.sam_model_path)
                # 设置精确模式
                self.image_label.sam_predictor.precise_mode = self.precise_mode
                self.image_label.sam_predictor.max_dimension = None if self.precise_mode else 512
                    
                logger.info(f"SAM模型加载完成，精确模式: {'ON' if self.precise_mode else 'OFF'}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"SAM模型加载失败: {str(e)}")
                return
        
        # 创建进度对话框
        progress_dialog = BatchSegmentationDialog(self)
        progress_dialog.show()
        
        # 执行批量分割
        results = []
        start_time = time.time()
        
        total_images = len(self.image_annotations)
        current_count = 0
        
        for image_path, annotation in self.image_annotations.items():
            if progress_dialog.cancelled:
                break
                
            current_count += 1
            elapsed_time = time.time() - start_time
            
            # 更新进度
            progress_dialog.update_progress(current_count, total_images, image_path, elapsed_time)
            QApplication.processEvents()
            
            # 处理单张图片
            result = self.process_single_image(image_path, annotation)
            if result:
                results.append(result)
                
        progress_dialog.close()
        
        if not progress_dialog.cancelled and results:
            # 保存结果
            self.save_batch_results(results)
            
            total_time = time.time() - start_time
            QMessageBox.information(
                self, "批量分割完成", 
                f"成功处理 {len(results)} 张图片\n"
                f"总耗时: {total_time:.1f} 秒\n"
                f"平均每张: {total_time/len(results):.1f} 秒\n\n"
                "结果已保存到 segmentation_results.csv（仅CSV格式）"
            )
        elif progress_dialog.cancelled:
            QMessageBox.information(self, "操作取消", "批量分割已取消")
    
    def process_single_image(self, image_path, annotation):
        """处理单张图片的分割"""
        try:
            logger.info(f"处理图片: {os.path.basename(image_path)}")
            
            # 加载图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 设置SAM图像（如果需要）
            if self.image_label.sam_predictor:
                success = self.image_label.sam_predictor.set_image(image_rgb)
                if not success:
                    logger.error(f"SAM设置图像失败: {image_path}")
                    return None
                
                # 进行分割
                mask, score = self.image_label.sam_predictor.predict(
                    annotation['positive'], 
                    annotation['negative']
                )
                
                if mask is not None:
                    # 计算覆盖率
                    coverage_rate = np.mean(mask > 0) * 100
                    
                    # 保存对比图
                    self.save_comparison_image_for_path(image_path, image_rgb, mask)
                    
                    return {
                        'image_name': os.path.basename(image_path),
                        'coverage_rate': coverage_rate,
                        'confidence_score': score,
                        'positive_points': len(annotation['positive']),
                        'negative_points': len(annotation['negative'])
                    }
                    
        except Exception as e:
            logger.error(f"处理图片时出错 {image_path}: {e}")
            
        return None
    
    def save_comparison_image_for_path(self, image_path, image_rgb, mask):
        """为指定路径的图片保存对比图"""
        try:
            # 创建对比图
            if SAM_AVAILABLE:
                color_mask = label_to_color_image((mask > 0).astype(np.uint8))
            else:
                # 简单的红色掩码
                color_mask = np.zeros_like(image_rgb)
                color_mask[:, :, 0] = (mask > 0) * 255
            
            # 叠加掩码
            alpha = 0.3
            result = image_rgb.copy()
            mask_area = mask > 0
            result[mask_area] = (1 - alpha) * result[mask_area] + alpha * color_mask[mask_area]
            
            # 保存
            output_path = image_path.replace('.tif', '_segmented.png')
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            logger.debug(f"保存分割结果: {output_path}")
            
        except Exception as e:
            logger.error(f"保存对比图失败: {e}")
    
    def save_batch_results(self, results):
        """保存批量处理结果到CSV"""
        try:
            df = pd.DataFrame(results)
            
            # 获取目录名作为后缀
            dir_suffix = ""
            if self.data_directory:
                dir_name = os.path.basename(self.data_directory)
                dir_suffix = f"_{dir_name}"
            
            # 保存到数据目录中
            if self.data_directory:
                output_file = os.path.join(self.data_directory, f"segmentation_results{dir_suffix}.csv")
            else:
                output_file = f"segmentation_results{dir_suffix}.csv"
                
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"批量分割结果保存到: {output_file}")
            
            # Excel saving removed - only save CSV files
            logger.info("结果仅保存为CSV格式")
            
        except Exception as e:
            logger.error(f"保存结果文件失败: {e}")
            
    def generate_mask(self):
        # 实时分割已移除，提示用户使用批量分割
        QMessageBox.information(
            self, "提示", 
            "实时分割已改为批量模式！\n\n"
            "请完成所有图片标记后，点击 '🚀 开始批量分割' 按钮。"
        )
        
    def save_comparison_image(self):
        """保存对比图"""
        if self.image_label.current_mask is None:
            QMessageBox.warning(self, "警告", "无掩码可保存。请先生成掩码。")
            return
            
        if not self.current_image_path:
            return
            
        logger.info("开始保存对比图...")
        start_time = time.time()
        
        try:
            base_name = os.path.splitext(self.current_image_path)[0]
            comparison_path = base_name + "_comparison.png"
            
            # 获取原始图像
            original = self.image_label.original_image.copy()
            
            # 创建覆盖图像
            if SAM_AVAILABLE and hasattr(self.image_label, 'current_mask'):
                mask_binary = (self.image_label.current_mask > 0).astype(np.uint8)
                color_mask = label_to_color_image(mask_binary)
                overlay = cv2.addWeighted(original, 0.7, color_mask, 0.3, 0.0)
            else:
                overlay = original.copy()
                overlay_color = np.zeros_like(original)
                overlay_color[self.image_label.current_mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(original, 0.7, overlay_color, 0.3, 0)
            
            # 创建并排对比
            h, w = original.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = original
            comparison[:, w:] = overlay
            
            # 添加文字标签
            try:
                cv2.putText(comparison, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(comparison, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                cv2.putText(comparison, "SAM Segmentation", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(comparison, "SAM Segmentation", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            except:
                pass
            
            # 保存对比图
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(comparison_path, comparison_bgr)
            
            # 保存结果数据
            image_name = os.path.basename(self.current_image_path)
            coverage = self.image_label.get_mask_coverage()
            self.results_data[image_name] = coverage
            
            save_time = time.time() - start_time
            logger.info(f"对比图保存完成，耗时: {save_time:.3f}秒")
            
            self.show_status_message(
                f"💾 已保存: {os.path.basename(comparison_path)} (覆盖率: {coverage:.2f}%)", 
                4000
            )
            
        except Exception as e:
            logger.error(f"保存对比图失败: {e}")
            QMessageBox.critical(self, "错误", f"保存对比图失败: {str(e)}")
            
    def save_all_results(self):
        if not self.results_data:
            QMessageBox.warning(self, "警告", "无结果可保存。")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "sam_annotation_results.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                df = pd.DataFrame([
                    {"Image": name, "Coverage_Percentage": coverage}
                    for name, coverage in self.results_data.items()
                ])
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
                logger.info(f"结果保存到CSV: {file_path}")
                self.show_status_message(f"📊 结果已保存: {len(self.results_data)} 张图片", 3000)
                
                QMessageBox.information(self, "成功", 
                    f"结果保存成功!\n\n文件: {file_path}\n图片数: {len(self.results_data)}")
                    
            except Exception as e:
                logger.error(f"保存CSV失败: {e}")
                QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")
            
    def clear_points(self):
        self.image_label.clear_points()
        self.update_info()
        self.update_mark_status()
        self.show_status_message("🗑️ 标记点已清除", 1500)
        
    def show_status_message(self, message, duration=3000, persistent=False):
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        if not persistent:
            self.status_timer.start(duration)
        
    def clear_status_message(self):
        self.status_label.setVisible(False)
        self.status_timer.stop()
        
    def on_points_updated(self):
        self.update_info()
        # 自动保存当前标记
        self.auto_save_current_points()
        # 有标记点时启用清除按钮
        if len(self.image_label.positive_points) > 0:
            self.clear_button.setEnabled(True)
        
    def update_info(self):
        positive_count = len(self.image_label.positive_points)
        negative_count = len(self.image_label.negative_points)
        
        info_text = ""
        
        if self.current_image_path:
            image_name = os.path.basename(self.current_image_path)
            info_text += f"📷 {image_name}\n"
            if len(self.image_list) > 1:
                info_text += f"🔢 {self.current_index + 1} / {len(self.image_list)}\n\n"
            else:
                info_text += "🔢 单张图片模式\n\n"
        
        info_text += f"📍 标记点:\n"
        info_text += f"• ✅ 正向: {positive_count}\n"
        info_text += f"• ❌ 负向: {negative_count}\n"
        
        if self.image_label.current_mask is not None:
            coverage = self.image_label.get_mask_coverage()
            info_text += f"• 📊 覆盖率: {coverage:.2f}%\n"
            
        info_text += f"\n📈 已处理: {len(self.results_data)} 张\n"
        info_text += f"\n⌨️ 快捷键:\n"
        info_text += f"• SPACE: 开始批量分割\n"
        info_text += f"• A/D: 上/下一张\n"
        info_text += f"• C: 清除标记点\n\n"
        info_text += f"🖱️ 点击操作:\n"
        info_text += f"• 左键: 添加绿点或取消红点\n"
        info_text += f"• 右键: 添加红点或取消绿点"
        
        self.info_label.setText(info_text)


def main():
    logger.info("程序启动...")
    start_time = time.time()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    if not SAM_AVAILABLE:
        reply = QMessageBox.question(
            None, "SAM依赖缺失", 
            "SAM依赖不可用。使用后备模式继续?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
    
    window = WoundAnnotatorUI()
    window.show()
    
    startup_time = time.time() - start_time
    logger.info(f"程序启动完成，总耗时: {startup_time:.2f}秒")
    logger.info("日志记录已启用，文件: sam_annotator_debug.log")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
