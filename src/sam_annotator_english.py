import sys
import os
import json
import numpy as np
import time
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QProgressBar, QProgressDialog, QDialog)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QKeyEvent
import cv2
import pandas as pd
import glob
from PIL import Image, ImageDraw, ImageFont

# Ensure logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup detailed logging
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
    """Batch segmentation progress dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Segmentation Progress")
        self.setFixedSize(500, 200)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Progress information
        self.status_label = QLabel("Preparing to start...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Progress bar
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
        
        # Time information
        self.time_label = QLabel("Estimated remaining time: Calculating...")
        self.time_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(self.time_label)
        
        # Current processing image
        self.current_image_label = QLabel("Current image: None")
        self.current_image_label.setStyleSheet("font-size: 12px; color: #2c3e50;")
        layout.addWidget(self.current_image_label)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
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
        """Update progress"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        percentage = (current / total * 100) if total > 0 else 0
        self.status_label.setText(f"Processing... {current}/{total} ({percentage:.1f}%)")
        self.current_image_label.setText(f"Current image: {os.path.basename(current_image)}")
        
        # Estimate remaining time
        if current > 0 and elapsed_time > 0:
            avg_time_per_image = elapsed_time / current
            remaining_images = total - current
            estimated_remaining = avg_time_per_image * remaining_images
            
            minutes = int(estimated_remaining // 60)
            seconds = int(estimated_remaining % 60)
            self.time_label.setText(f"Estimated remaining time: {minutes}:{seconds:02d}")
        else:
            self.time_label.setText("Estimated remaining time: Calculating...")

# Import SAM dependencies from utils
try:
    logger.info("Starting to load SAM dependencies...")
    start_time = time.time()
    from utils import sam_model_registry, SamPredictor, largest_component, label_to_color_image
    import torch
    SAM_AVAILABLE = True
    logger.info(f"SAM dependencies loaded successfully, time taken: {time.time() - start_time:.2f} seconds")
except ImportError as e:
    SAM_AVAILABLE = False
    logger.error(f"SAM not available: {e}")


class SAMModelWrapper:
    """Optimized SAM model wrapper with preloading support"""
    _instance = None
    
    def __new__(cls, model_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path=None):
        if self._initialized:
            return
            
        logger.info("Initializing SAM model...")
        start_time = time.time()
        
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_image_hash = None
        self.current_image_path = None  # Current set image path
        # Default to compressed mode (512px), can be controlled by precise_mode parameter
        self.max_dimension = 512
        self.precise_mode = False
        
        if model_path and SAM_AVAILABLE and os.path.exists(model_path):
            try:
                logger.info(f"Loading SAM model from {model_path}")
                logger.info(f"Device: {self.device} (CUDA available: {torch.cuda.is_available()})")
                logger.info(f"Image processing size: {self.max_dimension}px (Precise mode: {self.precise_mode})")
                if self.device == "cpu":
                    logger.warning("Running SAM on CPU will be slower, recommend using compressed mode")
                    logger.info("For better quality, consider installing CUDA version of PyTorch")
                
                model_type = "_".join(model_path.split("_")[1:3])
                sam = sam_model_registry[model_type](checkpoint=model_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                
                load_time = time.time() - start_time
                logger.info(f"SAM model loaded successfully, time taken: {load_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"SAM model loading failed: {e}")
                self.predictor = None
        else:
            logger.warning("SAM model file not found or dependencies missing")
            
        self._initialized = True
            
    def set_image(self, image):
        if self.predictor:
            try:
                # Fast hash: using image shape and partial pixel data
                hash_start = time.time()
                # Create fast hash using image shape and pixel values from four corners
                quick_hash = (
                    image.shape,
                    tuple(image[0, 0]),  # Top-left corner
                    tuple(image[0, -1]),  # Top-right corner
                    tuple(image[-1, 0]),  # Bottom-left corner
                    tuple(image[-1, -1]),  # Bottom-right corner
                    image[image.shape[0]//2, image.shape[1]//2].tobytes()  # Center point
                )
                image_hash = hash(quick_hash)
                hash_time = time.time() - hash_start
                logger.debug(f"Image hash calculation time: {hash_time:.3f} seconds")
                
                if self.current_image_hash != image_hash:
                    # Check if image needs scaling
                    h, w = image.shape[:2]
                    self.original_size = (h, w)
                    self.scale_factor = 1.0
                    
                    # Ensure image has correct dtype before processing
                    if image.dtype != np.uint8:
                        logger.debug(f"Converting image dtype from {image.dtype} to uint8")
                        image = image.astype(np.uint8)
                    
                    # Ensure image has correct shape (H, W, 3)
                    if len(image.shape) != 3 or image.shape[2] != 3:
                        logger.error(f"Invalid image shape: {image.shape}, expected (H, W, 3)")
                        return False
                    
                    # Decide scaling based on precise mode
                    if self.precise_mode or self.max_dimension is None:
                        # Precise mode: use original image
                        sam_start = time.time()
                        self.predictor.set_image(image)
                        sam_time = time.time() - sam_start
                        logger.debug(f"SAM set original image time: {sam_time:.3f} seconds (precise mode)")
                    elif max(h, w) > self.max_dimension:
                        # Compressed mode: scale image
                        self.scale_factor = self.max_dimension / max(h, w)
                        new_h = int(h * self.scale_factor)
                        new_w = int(w * self.scale_factor)
                        
                        # Scale image
                        resize_start = time.time()
                        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        resize_time = time.time() - resize_start
                        logger.debug(f"Image resize time: {resize_time:.3f} seconds ({w}x{h} -> {new_w}x{new_h})")
                        
                        # Ensure resized image has correct dtype
                        if resized_image.dtype != np.uint8:
                            logger.debug(f"Converting resized image dtype from {resized_image.dtype} to uint8")
                            resized_image = resized_image.astype(np.uint8)
                        
                        # Use scaled image
                        sam_start = time.time()
                        self.predictor.set_image(resized_image)
                        sam_time = time.time() - sam_start
                        logger.debug(f"SAM set scaled image time: {sam_time:.3f} seconds (compressed mode)")
                    else:
                        # Image is already small enough, use directly
                        sam_start = time.time()
                        self.predictor.set_image(image)
                        sam_time = time.time() - sam_start
                        logger.debug(f"SAM set original image time: {sam_time:.3f} seconds (no scaling needed)")
                    
                    self.current_image_hash = image_hash
                else:
                    logger.debug("Skipping SAM image setting (same image)")
                return True
            except Exception as e:
                logger.error(f"SAM set image error: {e}")
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
            
            # If image was scaled, need to scale point coordinates (only in compressed mode)
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
            
            # Apply largest connected component filtering
            if SAM_AVAILABLE:
                try:
                    best_mask = largest_component(best_mask).astype(bool)
                except Exception as e:
                    logger.warning(f"Largest connected component processing failed: {e}")
            
            # If image was scaled, need to resize mask back to original size (only in compressed mode)
            if hasattr(self, 'scale_factor') and self.scale_factor != 1.0 and not self.precise_mode and hasattr(self, 'original_size'):
                resize_start = time.time()
                h, w = self.original_size
                best_mask_resized = cv2.resize(best_mask.astype(np.uint8), (w, h), 
                                              interpolation=cv2.INTER_NEAREST)
                resize_time = time.time() - resize_start
                logger.debug(f"Mask resize time: {resize_time:.3f} seconds")
                best_mask = best_mask_resized.astype(bool)
            
            predict_time = time.time() - start_time
            logger.info(f"SAM prediction completed, time taken: {predict_time:.3f} seconds, confidence: {best_score:.3f}")
            
            return best_mask.astype(np.uint8) * 255, float(best_score)
            
        except Exception as e:
            logger.error(f"SAM prediction error: {e}")
            return None, 0.0


class FastImageLabel(QLabel):
    pointsUpdated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing image label component...")
        
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
        # Don't set focus policy, let main window handle keyboard events
        self.setFocusPolicy(Qt.NoFocus)
        
        # Initialize SAM
        # SAM model lazy loading, only initialize during batch segmentation
        self.sam_model_path = "models/sam_vit_b_01ec64.pth"
        self.sam_predictor = None
        logger.info("SAM model will be loaded during batch segmentation")
        
    def set_image(self, image_path):
        """Optimized image loading"""
        logger.info(f"Starting to load image: {image_path}")
        start_time = time.time()
        
        # Commented out cache check to ensure image is reloaded every time
        # This fixes the issue of needing to press keys twice to switch images
        # if self.last_image_path == image_path:
        #     logger.debug("Skipping load (same image path)")
        #     return True
            
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return False
            
        # Load image
        try:
            load_start = time.time()
            self.original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if self.original_image is None:
                logger.error(f"Unable to read image: {image_path}")
                return False
            
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            load_time = time.time() - load_start
            logger.debug(f"OpenCV image read time: {load_time:.3f} seconds")
            
        except Exception as e:
            logger.error(f"Image loading exception: {e}")
            return False
        
        self.last_image_path = image_path
        
        # Skip SAM image setting, only process during batch segmentation
        logger.debug("Skipping SAM image setting, will be processed during batch segmentation")
        
        # Reset state
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.show_mask = False
        self.cached_base_pixmap = None
        
        # Update display
        display_start = time.time()
        self.update_display_fast()
        display_time = time.time() - display_start
        logger.debug(f"Display update time: {display_time:.3f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Image loading completed, total time: {total_time:.3f} seconds")
        return True
        
    def update_display_fast(self):
        """Ultra-fast display update"""
        if self.original_image is None:
            return
            
        start_time = time.time()
        
        # Create base pixmap when needed
        if self.cached_base_pixmap is None:
            self.create_base_pixmap()
            
        # Fast point overlay
        self.draw_points_overlay()
        
        update_time = time.time() - start_time
        logger.debug(f"Display update time: {update_time:.3f} seconds")
        
    def create_base_pixmap(self):
        """Create cached base pixmap"""
        start_time = time.time()
        
        display_image = self.original_image.copy()
        
        # Apply mask overlay (if exists)
        if self.show_mask and self.current_mask is not None:
            try:
                mask_start = time.time()
                if SAM_AVAILABLE:
                    mask_binary = (self.current_mask > 0).astype(np.uint8)
                    color_mask = label_to_color_image(mask_binary)
                    if color_mask.size > 0:
                        display_image = cv2.addWeighted(display_image, 0.7, color_mask, 0.3, 0.0)
                else:
                    # Fallback solution
                    overlay_color = np.zeros_like(display_image)
                    overlay_color[self.current_mask > 0] = [0, 255, 0]
                    display_image = cv2.addWeighted(display_image, 0.7, overlay_color, 0.3, 0)
                
                mask_time = time.time() - mask_start
                logger.debug(f"Mask overlay time: {mask_time:.3f} seconds")
            except Exception as e:
                logger.error(f"Mask overlay error: {e}")
        
        # Convert to QPixmap
        try:
            convert_start = time.time()
            height, width, channel = display_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            convert_time = time.time() - convert_start
            logger.debug(f"QPixmap conversion time: {convert_time:.3f} seconds")
        except Exception as e:
            logger.error(f"QPixmap conversion error: {e}")
            return
        
        # Scale to fit
        try:
            scale_start = time.time()
            widget_size = self.size()
            self.cached_base_pixmap = pixmap.scaled(widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scale_time = time.time() - scale_start
            logger.debug(f"Pixmap scaling time: {scale_time:.3f} seconds")
        except Exception as e:
            logger.error(f"Pixmap scaling error: {e}")
            return
        
        # Update image rectangle and scale factors
        x_offset = (widget_size.width() - self.cached_base_pixmap.width()) // 2
        y_offset = (widget_size.height() - self.cached_base_pixmap.height()) // 2
        self.image_rect = (x_offset, y_offset, self.cached_base_pixmap.width(), self.cached_base_pixmap.height())
        
        self.scale_x = self.cached_base_pixmap.width() / width
        self.scale_y = self.cached_base_pixmap.height() / height
        
        create_time = time.time() - start_time
        logger.debug(f"Base pixmap creation time: {create_time:.3f} seconds")
        
    def draw_points_overlay(self):
        """Fast point drawing"""
        if self.cached_base_pixmap is None:
            return
            
        start_time = time.time()
        
        final_pixmap = QPixmap(self.cached_base_pixmap)
        painter = QPainter(final_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw positive points
        painter.setPen(QPen(QColor(0, 255, 0), 4))
        painter.setBrush(QColor(0, 255, 0, 180))
        for point in self.positive_points:
            x = int(point[0] * self.scale_x)
            y = int(point[1] * self.scale_y)
            painter.drawEllipse(QPoint(x, y), 8, 8)
            
        # Draw negative points
        painter.setPen(QPen(QColor(255, 0, 0), 4))
        painter.setBrush(QColor(255, 0, 0, 180))
        for point in self.negative_points:
            x = int(point[0] * self.scale_x)
            y = int(point[1] * self.scale_y)
            painter.drawEllipse(QPoint(x, y), 8, 8)
            
        painter.end()
        self.setPixmap(final_pixmap)
        
        draw_time = time.time() - start_time
        logger.debug(f"Point drawing time: {draw_time:.3f} seconds")
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        logger.debug("Window size changed, clearing cache")
        self.cached_base_pixmap = None
        if self.original_image is not None:
            self.update_display_fast()
        
    def mousePressEvent(self, event):
        if self.original_image is None or self.image_rect is None:
            return
            
        start_time = time.time()
        
        # Fast coordinate conversion
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
        
        # Check if clicking near existing points (cancel functionality)
        click_radius = 15  # Click radius
        point_removed = False
        
        if event.button() == Qt.LeftButton:
            # Left click: check if clicking red point (cancel red point)
            for i, (nx, ny) in enumerate(self.negative_points):
                if abs(orig_x - nx) < click_radius and abs(orig_y - ny) < click_radius:
                    self.negative_points.pop(i)
                    logger.info(f"Remove negative point: ({nx}, {ny})")
                    point_removed = True
                    break
            
            # If no point removed, add green point
            if not point_removed:
                self.positive_points.append([orig_x, orig_y])
                logger.info(f"Add positive point: ({orig_x}, {orig_y})")
                
        elif event.button() == Qt.RightButton:
            # Right click: check if clicking green point (cancel green point)
            for i, (px, py) in enumerate(self.positive_points):
                if abs(orig_x - px) < click_radius and abs(orig_y - py) < click_radius:
                    self.positive_points.pop(i)
                    logger.info(f"Remove positive point: ({px}, {py})")
                    point_removed = True
                    break
            
            # If no point removed, add red point
            if not point_removed:
                self.negative_points.append([orig_x, orig_y])
                logger.info(f"Add negative point: ({orig_x}, {orig_y})")
        
        # Fast update - only redraw points
        self.draw_points_overlay()
        self.pointsUpdated.emit()
        
        # Remove forced focus setting to avoid double-click issues
        # Main window already configured with StrongFocus, no need to reset focus after each click
        
        click_time = time.time() - start_time
        logger.debug(f"Mouse click processing time: {click_time:.3f} seconds")
        
    def generate_sam_mask(self):
        if len(self.positive_points) == 0:
            return False
            
        logger.info("Starting to generate SAM mask...")
        start_time = time.time()
        
        if self.sam_predictor and self.sam_predictor.predictor:
            mask, score = self.sam_predictor.predict(self.positive_points, self.negative_points)
            if mask is not None:
                self.current_mask = mask
                self.show_mask = True
                self.cached_base_pixmap = None  # Force regeneration
                
                total_time = time.time() - start_time
                logger.info(f"SAM mask generation successful, total time: {total_time:.3f} seconds, confidence: {score:.3f}")
                
                self.update_display_fast()
                return True
        
        # Fallback solution
        logger.info("Using fallback mask generation method...")
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
        logger.info(f"Fallback mask generation completed, time taken: {fallback_time:.3f} seconds")
        
        self.update_display_fast()
            
    def clear_points(self):
        logger.info("Clear all annotation points")
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
            logger.debug(f"Mask coverage rate: {coverage:.2f}%")
            return coverage
        return 0.0


class WoundAnnotatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing main window...")
        
        self.setWindowTitle("SWEET - SAM Widget for Edge Evaluation Tool")
        self.setGeometry(50, 50, 1500, 900)
        
        # Image management - single image loading mode
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        self.results_data = {}
        self.image_annotations = {}  # Store annotation points for all images {image_path: {'positive': [...], 'negative': [...]}}
        self.data_directory = None  # Save user selected data directory
        
        # Status message timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.clear_status_message)
        
        self.init_ui()
        
        # Ensure main window can receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Set all buttons to not get focus, let main window handle keyboard events
        for widget in self.findChildren(QPushButton):
            widget.setFocusPolicy(Qt.NoFocus)
        
        self.setFocus()
        
        logger.info("Main window initialization completed")
        
        # UI initialization completed
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        
        # Left panel - image display
        self.image_label = FastImageLabel()
        self.image_label.main_window = self  # Save main window reference
        self.image_label.pointsUpdated.connect(self.on_points_updated)
        main_layout.addWidget(self.image_label, 4)
        
        # Right panel - controls
        right_panel = QVBoxLayout()
        
        # Title - removed
        # title = QLabel("SWEET")
        # title.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("""
        #     font-size: 20px; font-weight: bold; padding: 15px; 
        #     background-color: #2c3e50; color: white; border-radius: 8px;
        # """)
        # right_panel.addWidget(title)
        
        # Status message
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
        
        # Information display
        self.info_label = QLabel("Select directory to start annotation")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            background-color: #ecf0f1; border: 2px solid #34495e; border-radius: 8px;
            padding: 15px; margin: 5px; font-size: 13px; color: #2c3e50;
        """)
        right_panel.addWidget(self.info_label)
        
        # Model status - removed
        # self.model_status = QLabel()
        # self.update_model_status()
        # right_panel.addWidget(self.model_status)
        
        # Button style - dark background, white text, high contrast
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
        
        # Load buttons
        load_layout = QHBoxLayout()
        
        self.load_dir_button = QPushButton("📁 Load Dir")
        self.load_dir_button.clicked.connect(self.load_directory)
        self.load_dir_button.setStyleSheet(button_style + """
            background-color: #27ae60; border-color: #229954;
        """)
        load_layout.addWidget(self.load_dir_button)
        
        # Precise mode toggle (default OFF, use compressed mode)
        self.precise_mode_button = QPushButton("🎯 Precise: OFF")
        self.precise_mode_button.clicked.connect(self.toggle_precise_mode)
        self.precise_mode_button.setStyleSheet(button_style + """
            background-color: #95a5a6; border-color: #7f8c8d;
        """)
        load_layout.addWidget(self.precise_mode_button)
        
        # Default to compressed mode
        self.precise_mode = False
        
        right_panel.addLayout(load_layout)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous (A)")
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setEnabled(False)
        self.prev_button.setStyleSheet(button_style)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next (D) ▶")
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setEnabled(False)
        self.next_button.setStyleSheet(button_style)
        nav_layout.addWidget(self.next_button)
        right_panel.addLayout(nav_layout)
        
        # Annotation status display
        self.mark_status = QLabel("📍 Status: Not started")
        self.mark_status.setAlignment(Qt.AlignCenter)
        self.mark_status.setStyleSheet("""
            font-size: 14px; font-weight: bold; padding: 8px; margin: 3px;
            background-color: #34495e; color: white; border-radius: 5px;
        """)
        right_panel.addWidget(self.mark_status)
        
        # Clear button
        self.clear_button = QPushButton("🗑️ Clear Annotations (C)")
        self.clear_button.clicked.connect(self.clear_points)
        self.clear_button.setEnabled(False)
        self.clear_button.setStyleSheet(button_style + """
            background-color: #95a5a6; border-color: #7f8c8d;
        """)
        right_panel.addWidget(self.clear_button)
        
        # Batch segmentation button
        self.batch_segment_button = QPushButton("🚀 Start Batch Segmentation")
        self.batch_segment_button.clicked.connect(self.start_batch_segmentation)
        self.batch_segment_button.setEnabled(False)
        self.batch_segment_button.setStyleSheet(button_style + """
            background-color: #e74c3c; border-color: #c0392b;
            font-size: 16px; padding: 15px;
        """)
        right_panel.addWidget(self.batch_segment_button)
        
        right_panel.addStretch()
        
        # Add right panel
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(380)
        main_layout.addWidget(right_widget)
        
    def keyPressEvent(self, event):
        # Key event handling
        if event.key() == Qt.Key_A:
            # Process A key
            self.prev_image()
            event.accept()  # Explicitly accept event to prevent propagation
        elif event.key() == Qt.Key_D:
            # Process D key
            self.next_image()
            event.accept()  # Explicitly accept event to prevent propagation
        elif event.key() == Qt.Key_C:
            self.clear_points()
            event.accept()
        elif event.key() == Qt.Key_Space:
            # Trigger batch segmentation
            if self.batch_segment_button.isEnabled():
                self.start_batch_segmentation()
            event.accept()
        else:
            super().keyPressEvent(event)
        
    def update_model_status(self):
        # Model status display removed
        pass
    
    def toggle_precise_mode(self):
        """Toggle precise mode"""
        if self.precise_mode:
            # Switch to compressed mode
            self.precise_mode = False
            self.precise_mode_button.setText("🎯 Precise: OFF")
            self.precise_mode_button.setStyleSheet("""
                QPushButton {
                    font-size: 13px; font-weight: bold; padding: 10px; margin: 3px;
                    border: 2px solid; border-radius: 8px; color: white;
                    background-color: #95a5a6; border-color: #7f8c8d;
                }
            """)
            # Switched to compressed mode
        else:
            # Switch to precise mode
            self.precise_mode = True
            self.precise_mode_button.setText("🎯 Precise: ON")
            self.precise_mode_button.setStyleSheet("""
                QPushButton {
                    font-size: 13px; font-weight: bold; padding: 10px; margin: 3px;
                    border: 2px solid; border-radius: 8px; color: white;
                    background-color: #e74c3c; border-color: #c0392b;
                }
            """)
            # Switched to precise mode
    
    def auto_save_current_points(self):
        """Auto-save current image annotation points (silent save, no prompt)"""
        if not self.current_image_path:
            return
            
        if self.image_label.positive_points or self.image_label.negative_points:
            self.image_annotations[self.current_image_path] = {
                'positive': self.image_label.positive_points.copy(),
                'negative': self.image_label.negative_points.copy()
            }
            
            num_points = len(self.image_label.positive_points) + len(self.image_label.negative_points)
            logger.debug(f"Auto-save annotations: {os.path.basename(self.current_image_path)} - {num_points} points")
        else:
            # If no points, remove this image's annotations
            if self.current_image_path in self.image_annotations:
                del self.image_annotations[self.current_image_path]
                logger.debug(f"Remove annotations: {os.path.basename(self.current_image_path)}")
        
        # Update status (silent update, no message prompt)
        self.update_mark_status()
    
    def save_current_points(self):
        """Save current image annotation points"""
        if not self.current_image_path:
            return
            
        if self.image_label.positive_points or self.image_label.negative_points:
            self.image_annotations[self.current_image_path] = {
                'positive': self.image_label.positive_points.copy(),
                'negative': self.image_label.negative_points.copy()
            }
            
            num_points = len(self.image_label.positive_points) + len(self.image_label.negative_points)
            logger.info(f"Save annotations: {os.path.basename(self.current_image_path)} - {num_points} points")
            self.show_status_message(f"✅ Saved {num_points} annotation points", 2000)
            self.update_mark_status()
        else:
            # If no points, remove this image's annotations
            if self.current_image_path in self.image_annotations:
                del self.image_annotations[self.current_image_path]
            self.show_status_message("⚠️ No annotation points to save", 2000)
            self.update_mark_status()
    
    def update_mark_status(self):
        """Update annotation status display"""
        total_images = len(self.image_list) if self.image_list else 0
        marked_images = len(self.image_annotations)
        
        if total_images == 0:
            status_text = "📍 Status: No images"
            color = "#95a5a6"
        elif marked_images == 0:
            status_text = f"📍 Status: 0/{total_images}"
            color = "#e74c3c"
        elif marked_images == total_images:
            status_text = f"📍 Status: {marked_images}/{total_images} ✓"
            color = "#27ae60"
        else:
            status_text = f"📍 Status: {marked_images}/{total_images}"
            color = "#f39c12"
        
        self.mark_status.setText(status_text)
        self.mark_status.setStyleSheet(f"""
            font-size: 13px; font-weight: bold; padding: 8px; margin: 3px;
            background-color: {color}; color: white; border-radius: 5px;
        """)
        
        # Update batch segmentation button status
        self.batch_segment_button.setEnabled(marked_images > 0)
        
    def load_directory(self):
        """Load directory"""
        # Check if there are unsegmented marked images
        if self.image_annotations:
            # Check if any segmentation has been done
            has_segmented = any(
                os.path.exists(img_path.replace('.tif', '_segmented.png'))
                for img_path in self.image_annotations.keys()
            )
            
            if not has_segmented:
                # No segmentation done yet, warn user
                reply = QMessageBox.warning(
                    self, "Warning", 
                    f"You have {len(self.image_annotations)} marked images that haven't been segmented.\n\n"
                    "Switching directories will lose these marks. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    return
        
        # User selecting directory
        default_dir = os.path.join(os.getcwd(), "data")
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", default_dir)
        if not directory:
            return
            
        # Clear previous annotations to prevent cross-contamination
        self.image_annotations.clear()
        self.results_data.clear()
        
        # Save data directory path
        self.data_directory = directory
        logger.info(f"Selected directory: {directory}")
        start_time = time.time()
        
        # Quick scan for image files
        self.show_status_message("🔍 Scanning directory...", persistent=True)
        QApplication.processEvents()
        
        image_extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']  # Only load original images, not comparison images
        image_list = []
        
        # Use set to avoid duplicate files
        unique_files = set()
        for ext in image_extensions:
            found_files = glob.glob(os.path.join(directory, ext))
            # Filter out comparison images and add to set to avoid duplicates
            for file in found_files:
                if 'comparison' not in os.path.basename(file).lower():
                    unique_files.add(file)
        
        # Convert to list
        image_list = list(unique_files)
        
        image_list.sort()
        scan_time = time.time() - start_time
        logger.info(f"Directory scan completed, found {len(image_list)} images, time taken: {scan_time:.3f} seconds")
        
        if image_list:
            self.image_list = image_list
            self.current_index = 0
            
            # Only load first image
            load_start = time.time()
            self.load_current_image()
            load_time = time.time() - load_start
            logger.info(f"First image load time: {load_time:.3f} seconds")
            
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            # After loading is complete, batch segmentation can be performed
            pass
            
            total_time = time.time() - start_time
            self.show_status_message(f"✅ Directory loaded: {len(image_list)} images (total time: {total_time:.2f} seconds)", 4000)
        else:
            self.show_status_message("❌ No image files found in directory", 3000)
            
    def load_current_image(self):
        """Load current image"""
        if 0 <= self.current_index < len(self.image_list):
            logger.info(f"Loading image {self.current_index + 1}/{len(self.image_list)}")
            start_time = time.time()
            
            # Clear cached path to ensure image reloads when switching
            self.image_label.last_image_path = None
            self.current_image_path = self.image_list[self.current_index]
            
            if self.image_label.set_image(self.current_image_path):
                self.clear_button.setEnabled(True)
                # Auto-save, no need for manual save button
                
                # Load previously saved annotation points
                if self.current_image_path in self.image_annotations:
                    annotation = self.image_annotations[self.current_image_path]
                    self.image_label.positive_points = annotation['positive'].copy()
                    self.image_label.negative_points = annotation['negative'].copy()
                    self.image_label.update_display_fast()
                    
                    num_points = len(annotation['positive']) + len(annotation['negative'])
                    logger.info(f"Load saved annotations: {num_points} points")
                
                self.update_info()
                self.update_mark_status()
                
                load_time = time.time() - start_time
                image_name = os.path.basename(self.current_image_path)
                logger.info(f"Image loading completed: {image_name}, time taken: {load_time:.3f} seconds")
                
                # Update navigation button status
                self.prev_button.setEnabled(self.current_index > 0)
                self.next_button.setEnabled(self.current_index < len(self.image_list) - 1)
                
    def prev_image(self):
        if self.current_index > 0:
            logger.info("Switch to previous image")
            # Save current image annotation points
            self.auto_save_current_points()
            self.current_index -= 1
            self.load_current_image()
            # Ensure window maintains focus
            self.setFocus()
            
    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            logger.info("Switch to next image")
            # Save current image annotation points
            self.auto_save_current_points()
            self.current_index += 1
            self.load_current_image()
            # Ensure window maintains focus
            self.setFocus()
    
    def start_batch_segmentation(self):
        """Start batch segmentation processing"""
        if not self.image_annotations:
            QMessageBox.warning(self, "Warning", "No annotated images to segment!")
            return
            
        reply = QMessageBox.question(
            self, "Confirm Batch Segmentation", 
            f"Will perform segmentation on {len(self.image_annotations)} annotated images.\n\n"
            "This process may take a long time, continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # Initialize SAM model (if not already initialized)
        if self.image_label.sam_predictor is None:
            logger.info("Loading SAM model...")
            QApplication.processEvents()
            
            try:
                self.image_label.sam_predictor = SAMModelWrapper(self.image_label.sam_model_path)
                # Set precise mode
                self.image_label.sam_predictor.precise_mode = self.precise_mode
                self.image_label.sam_predictor.max_dimension = None if self.precise_mode else 512
                    
                logger.info(f"SAM model loaded, precise mode: {'ON' if self.precise_mode else 'OFF'}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"SAM model loading failed: {str(e)}")
                return
        
        # Create progress dialog
        progress_dialog = BatchSegmentationDialog(self)
        progress_dialog.show()
        
        # Execute batch segmentation
        results = []
        start_time = time.time()
        
        total_images = len(self.image_annotations)
        current_count = 0
        
        for image_path, annotation in self.image_annotations.items():
            if progress_dialog.cancelled:
                break
                
            current_count += 1
            elapsed_time = time.time() - start_time
            
            # Update progress
            progress_dialog.update_progress(current_count, total_images, image_path, elapsed_time)
            QApplication.processEvents()
            
            # Process single image
            result = self.process_single_image(image_path, annotation)
            if result:
                results.append(result)
                
        progress_dialog.close()
        
        if not progress_dialog.cancelled and results:
            # Save results
            self.save_batch_results(results)
            
            total_time = time.time() - start_time
            QMessageBox.information(
                self, "Batch Segmentation Complete", 
                f"Successfully processed {len(results)} images\n"
                f"Total time: {total_time:.1f} seconds\n"
                f"Average per image: {total_time/len(results):.1f} seconds\n\n"
                "Results saved to segmentation_results.csv (CSV format only)"
            )
        elif progress_dialog.cancelled:
            QMessageBox.information(self, "Operation Cancelled", "Batch segmentation cancelled")
    
    def process_single_image(self, image_path, annotation):
        """Process single image segmentation"""
        try:
            logger.info(f"Processing image: {os.path.basename(image_path)}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Unable to read image: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set SAM image (if needed)
            if self.image_label.sam_predictor:
                success = self.image_label.sam_predictor.set_image(image_rgb)
                if not success:
                    logger.error(f"SAM set image failed: {image_path}")
                    return None
                
                # Perform segmentation
                mask, score = self.image_label.sam_predictor.predict(
                    annotation['positive'], 
                    annotation['negative']
                )
                
                if mask is not None:
                    # Calculate coverage rate
                    coverage_rate = np.mean(mask > 0) * 100
                    
                    # Save comparison image
                    self.save_comparison_image_for_path(image_path, image_rgb, mask)
                    
                    return {
                        'image_name': os.path.basename(image_path),
                        'coverage_rate': coverage_rate,
                        'confidence_score': score,
                        'positive_points': len(annotation['positive']),
                        'negative_points': len(annotation['negative'])
                    }
                    
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            
        return None
    
    def save_comparison_image_for_path(self, image_path, image_rgb, mask):
        """Save comparison image for specified path"""
        try:
            # Create comparison image
            if SAM_AVAILABLE:
                color_mask = label_to_color_image((mask > 0).astype(np.uint8))
            else:
                # Simple red mask
                color_mask = np.zeros_like(image_rgb)
                color_mask[:, :, 0] = (mask > 0) * 255
            
            # Overlay mask
            alpha = 0.3
            result = image_rgb.copy()
            mask_area = mask > 0
            result[mask_area] = (1 - alpha) * result[mask_area] + alpha * color_mask[mask_area]
            
            # Save
            output_path = image_path.replace('.tif', '_segmented.png')
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            logger.debug(f"Save segmentation result: {output_path}")
            
        except Exception as e:
            logger.error(f"Save comparison image failed: {e}")
    
    def save_batch_results(self, results):
        """Save batch processing results to CSV"""
        try:
            df = pd.DataFrame(results)
            
            # Get directory name for suffix
            dir_suffix = ""
            if self.data_directory:
                dir_name = os.path.basename(self.data_directory)
                dir_suffix = f"_{dir_name}"
            
            # Save to data directory
            if self.data_directory:
                output_file = os.path.join(self.data_directory, f"segmentation_results{dir_suffix}.csv")
            else:
                output_file = f"segmentation_results{dir_suffix}.csv"
                
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"Batch segmentation results saved to: {output_file}")
            
            # Excel saving removed - only save CSV files
            logger.info("Results saved in CSV format only")
            
        except Exception as e:
            logger.error(f"Save results file failed: {e}")
            
    def generate_mask(self):
        # Real-time segmentation removed, prompt user to use batch segmentation
        QMessageBox.information(
            self, "Hint", 
            "Real-time segmentation changed to batch mode!\n\n"
            "Please complete all image annotations first, then click '🚀 Start Batch Segmentation' button."
        )
        
    def save_comparison_image(self):
        """Save comparison image"""
        if self.image_label.current_mask is None:
            QMessageBox.warning(self, "Warning", "No mask to save. Please generate mask first.")
            return
            
        if not self.current_image_path:
            return
            
        logger.info("Starting to save comparison image...")
        start_time = time.time()
        
        try:
            base_name = os.path.splitext(self.current_image_path)[0]
            comparison_path = base_name + "_comparison.png"
            
            # Get original image
            original = self.image_label.original_image.copy()
            
            # Create overlay image
            if SAM_AVAILABLE and hasattr(self.image_label, 'current_mask'):
                mask_binary = (self.image_label.current_mask > 0).astype(np.uint8)
                color_mask = label_to_color_image(mask_binary)
                overlay = cv2.addWeighted(original, 0.7, color_mask, 0.3, 0.0)
            else:
                overlay = original.copy()
                overlay_color = np.zeros_like(original)
                overlay_color[self.image_label.current_mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(original, 0.7, overlay_color, 0.3, 0)
            
            # Create side-by-side comparison
            h, w = original.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = original
            comparison[:, w:] = overlay
            
            # Add text labels
            try:
                cv2.putText(comparison, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(comparison, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                cv2.putText(comparison, "SAM Segmentation", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(comparison, "SAM Segmentation", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            except:
                pass
            
            # Save comparison image
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(comparison_path, comparison_bgr)
            
            # Save result data
            image_name = os.path.basename(self.current_image_path)
            coverage = self.image_label.get_mask_coverage()
            self.results_data[image_name] = coverage
            
            save_time = time.time() - start_time
            logger.info(f"Comparison image save completed, time taken: {save_time:.3f} seconds")
            
            self.show_status_message(
                f"💾 Saved: {os.path.basename(comparison_path)} (Coverage rate: {coverage:.2f}%)", 
                4000
            )
            
        except Exception as e:
            logger.error(f"Save comparison image failed: {e}")
            QMessageBox.critical(self, "Error", f"Save comparison image failed: {str(e)}")
            
    def save_all_results(self):
        if not self.results_data:
            QMessageBox.warning(self, "Warning", "No results to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "sam_annotation_results.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                df = pd.DataFrame([
                    {"Image": name, "Coverage_Percentage": coverage}
                    for name, coverage in self.results_data.items()
                ])
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
                logger.info(f"Results saved to CSV: {file_path}")
                self.show_status_message(f"📊 Results saved: {len(self.results_data)} images", 3000)
                
                QMessageBox.information(self, "Success", 
                    f"Results saved successfully!\n\nFile: {file_path}\nImages: {len(self.results_data)}")
                    
            except Exception as e:
                logger.error(f"Save CSV failed: {e}")
                QMessageBox.critical(self, "Error", f"Save results failed: {str(e)}")
            
    def clear_points(self):
        self.image_label.clear_points()
        self.update_info()
        self.update_mark_status()
        self.show_status_message("🗑️ Annotation points cleared", 1500)
        
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
        # Auto-save current annotations
        self.auto_save_current_points()
        # Enable clear button when there are annotation points
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
                info_text += "🔢 Single image mode\n\n"
        
        info_text += f"📍 Annotation Points:\n"
        info_text += f"• ✅ Positive: {positive_count}\n"
        info_text += f"• ❌ Negative: {negative_count}\n"
        
        if self.image_label.current_mask is not None:
            coverage = self.image_label.get_mask_coverage()
            info_text += f"• 📊 Coverage Rate: {coverage:.2f}%\n"
            
        info_text += f"\n📈 Processed: {len(self.results_data)} images\n"
        info_text += f"\n⌨️ Shortcuts:\n"
        info_text += f"• SPACE: Start batch segmentation\n"
        info_text += f"• A/D: Previous/Next\n"
        info_text += f"• C: Clear annotations\n\n"
        info_text += f"🖱️ Mouse Operations:\n"
        info_text += f"• Left click: Add green point or cancel red point\n"
        info_text += f"• Right click: Add red point or cancel green point"
        
        self.info_label.setText(info_text)


def main():
    logger.info("Program starting...")
    start_time = time.time()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    if not SAM_AVAILABLE:
        reply = QMessageBox.question(
            None, "SAM Dependencies Missing", 
            "SAM dependencies unavailable. Continue with fallback mode?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
    
    window = WoundAnnotatorUI()
    window.show()
    
    startup_time = time.time() - start_time
    logger.info(f"Program startup completed, total time: {startup_time:.2f} seconds")
    logger.info("Logging enabled, file: sam_annotator.log")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
