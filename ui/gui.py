#!/usr/bin/env python3
"""
Photography Editor GUI

Main GUI interface for the Photography Editor application.
Displays raw images in a grid layout and allows image selection.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image as PILImage
from PIL.ExifTags import TAGS, GPSTAGS
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QScrollArea, QMessageBox, QSpacerItem,
    QSizePolicy, QStackedWidget, QListWidget, QListWidgetItem, QGroupBox,
    QSlider, QCheckBox, QButtonGroup
)
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QFont, QImage, QPainter, QColor, QImageReader
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer

# Get the path to the data/raw folder
PROJECT_ROOT = Path(__file__).parent.parent
RAW_IMAGES_PATH = PROJECT_ROOT / "data" / "raw"
TECHNIQUES_PATH = PROJECT_ROOT / "techniques"

# This function is responsible for loading an image file and applying the correct orientation
# input: image_path
# output: Image with correct orientation applied
def load_pixmap_with_orientation(image_path: str) -> QPixmap:
    """Load image with EXIF orientation applied."""
    reader = QImageReader(image_path)
    reader.setAutoTransform(True)
    image = reader.read()
    return QPixmap.fromImage(image)


# This function is responsible for extracting EXIF and image metadata from a photo. Opens image with PIL, 
# extracts EXIF tags, and organizes metadata into structured categories for documentation.
# Input: Path to an image file
# Output: Dict[] dictionary with basic information
def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract EXIF and other metadata from an image."""
    metadata = {
        "source_file": Path(image_path).name,
        "source_path": str(image_path),
        "extraction_date": datetime.now().isoformat(),
    }
    
    try:
        with PILImage.open(image_path) as img:
            # Basic image info
            metadata["image_format"] = img.format
            metadata["image_size"] = f"{img.width} x {img.height}"
            metadata["color_mode"] = img.mode
            
            # Extract EXIF data
            exif_data = img._getexif()
            if exif_data:
                exif_info = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except:
                            value = str(value)
                    
                    exif_info[tag_name] = value
                
                # Extract key imaging information
                metadata["camera"] = {
                    "make": exif_info.get("Make", "Unknown"),
                    "model": exif_info.get("Model", "Unknown"),
                    "software": exif_info.get("Software", "Unknown"),
                }
                
                metadata["lens"] = {
                    "f_stop": exif_info.get("FNumber", "Unknown"),
                    "aperture": exif_info.get("ApertureValue", "Unknown"),
                    "focal_length": exif_info.get("FocalLength", "Unknown"),
                    "focal_length_35mm": exif_info.get("FocalLengthIn35mmFilm", "Unknown"),
                    "lens_make": exif_info.get("LensMake", "Unknown"),
                    "lens_model": exif_info.get("LensModel", "Unknown"),
                }
                
                metadata["exposure"] = {
                    "shutter_speed": exif_info.get("ExposureTime", "Unknown"),
                    "iso": exif_info.get("ISOSpeedRatings", "Unknown"),
                    "exposure_program": exif_info.get("ExposureProgram", "Unknown"),
                    "exposure_mode": exif_info.get("ExposureMode", "Unknown"),
                    "exposure_bias": exif_info.get("ExposureBiasValue", "Unknown"),
                    "metering_mode": exif_info.get("MeteringMode", "Unknown"),
                }
                
                metadata["lighting"] = {
                    "white_balance": exif_info.get("WhiteBalance", "Unknown"),
                    "light_source": exif_info.get("LightSource", "Unknown"),
                    "flash": exif_info.get("Flash", "Unknown"),
                }
                
                metadata["capture_info"] = {
                    "datetime_original": exif_info.get("DateTimeOriginal", "Unknown"),
                    "datetime_digitized": exif_info.get("DateTimeDigitized", "Unknown"),
                    "orientation": exif_info.get("Orientation", "Unknown"),
                }
                
                # Store full EXIF for reference
                metadata["full_exif"] = exif_info
                
    except Exception as e:
        metadata["error"] = f"Failed to extract metadata: {str(e)}"
    
    return metadata


# Save metadata dictionary to a JSON file. Writes metadata to JSON with indentation for readability
# Input: Metadara dictionary to save
# Output: path to saved JSON file
def save_metadata_file(metadata: Dict[str, Any], output_path: Path) -> Path:
    """Save metadata to a JSON file."""
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata_path


# This section handles the initial page where you can select an image to edit. Ie: Initialize a clickable image card widget for the gallery.
class ImageCard(QWidget):
    
    clicked = pyqtSignal(str)  # Emits the image path when clicked
    
    # Initialize a clickable image card widget for the gallery.
    # Input: Path to image file, parent widget
    # Output: None
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.selected = False
        self.original_pixmap = load_pixmap_with_orientation(self.image_path)
        self.init_ui()
    
    # Set up the UI components for the image card.
    def init_ui(self):
        """Initialize the UI for the image card."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(5)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(220, 220)
        self.image_label.setStyleSheet("background-color: white;")
        layout.addWidget(self.image_label, 1)
        
        # Create filename label
        filename = Path(self.image_path).name
        self.filename_label = QLabel(filename)
        self.filename_label.setFont(QFont('Arial', 10))
        self.filename_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.filename_label)
        
        self.setLayout(layout)
        self.update_image_preview()
        self.update_style()
    
    # Refresh the displayed image preview with 1:1 aspect ratio crop.
    # Crops image to square from center, scales to fit label dimensions using smooth transformation.
    def update_image_preview(self):
        """Update the image preview with 1:1 crop."""
        # Get the available space for the image
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        
        if label_width <= 0 or label_height <= 0:
            return
        
        # Crop image to 1:1 ratio from center
        cropped_pixmap = self.crop_to_square(self.original_pixmap)
        
        # Scale to fill the label completely
        scaled_pixmap = cropped_pixmap.scaledToWidth(
            label_width,
            Qt.SmoothTransformation
        )
        
        # If the height doesn't match after scaling to width, scale to height
        if scaled_pixmap.height() < label_height:
            scaled_pixmap = cropped_pixmap.scaledToHeight(
                label_height,
                Qt.SmoothTransformation
            )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    # Crop a pixmap to 1:1 aspect ratio from the center.
    # Input: Original Image
    # Output: Cropped square image
    def crop_to_square(self, pixmap: QPixmap) -> QPixmap:
        width = pixmap.width()
        height = pixmap.height()
        
        # Determine the size of the square (use the smaller dimension)
        square_size = min(width, height)
        
        # Calculate the crop position to center the crop
        x = (width - square_size) // 2
        y = (height - square_size) // 2
        
        # Crop the pixmap
        return pixmap.copy(x, y, square_size, square_size)
    
    # Handle widget resize events to update image display. Triggered when card is resized, calls update_image_preview() to rescale image.
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_preview()
    
    # Update card border styling based on selection state. Applies blue border (3px solid #0078d4) when selected, transparent border otherwise.
    def update_style(self):
        if self.selected:
            self.setStyleSheet("""
                QWidget {
                    border: 3px solid #0078d4;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    border: 3px solid transparent;
                    border-radius: 5px;
                }
            """)
    
    # Handle mouse click on the card.
    # input: Mouse event
    # Output: None
    def mousePressEvent(self, event):
        self.clicked.emit(self.image_path)
    
    # Set the selection state of the card.Updates internal state and triggers style update to show/hide selection border.
    # Input: Selected bool of whwether card should be selected
    def set_selected(self, selected: bool):
        self.selected = selected
        self.update_style()


# Initialize a QLabel with painting/masking capabilities for inpainting.
class MaskableImageLabel(QLabel):

    # Initialize a QLabel with painting/masking capabilities for inpainting.
    def __init__(self, parent=None):
        super().__init__(parent)
        self.painting_enabled = False
        self.brush_size = 20
        self.apply_brush_callback = None
        self.show_brush_preview = False
        self.last_mouse_pos = None
        self.mask_size = None
        self.setMouseTracking(True)

    # Enable or disable painting mode. Toggles painting mode and brush preview visibility, triggers redraw.
    def set_painting_enabled(self, enabled: bool):
        self.painting_enabled = enabled
        self.show_brush_preview = enabled
        self.update()

    # Set the brush radius for painting.
    def set_brush_size(self, size: int):
        self.brush_size = max(1, size)

    # Register callback function for brush strokes. Stores callback that will be invoked during mouse drag to apply mask.
    def set_apply_brush_callback(self, callback):
        self.apply_brush_callback = callback

    # Set the dimensions of the underlying mask array.
    def set_mask_size(self, width: int, height: int):
        self.mask_size = (width, height)

    # Convert mouse position to pixmap coordinates. Accounts for label centering offset, validates bounds, returns scaled coordinates.
    def _map_to_pixmap_coords(self, pos):
        pixmap = self.pixmap()
        if pixmap is None:
            return None

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        if pixmap_width == 0 or pixmap_height == 0:
            return None

        x_offset = max(0, (self.width() - pixmap_width) // 2)
        y_offset = max(0, (self.height() - pixmap_height) // 2)

        x = pos.x() - x_offset
        y = pos.y() - y_offset

        if x < 0 or y < 0 or x >= pixmap_width or y >= pixmap_height:
            return None

        return x, y, pixmap_width, pixmap_height

    # Apply brush stroke at mouse position. Maps coordinates and invokes callback to paint mask if painting is enabled.
    def _apply_brush(self, pos):
        if not self.painting_enabled or self.apply_brush_callback is None:
            return

        mapped = self._map_to_pixmap_coords(pos)
        if mapped is None:
            return

        x, y, pixmap_width, pixmap_height = mapped
        self.apply_brush_callback(x, y, pixmap_width, pixmap_height, self.brush_size)

    # Handle mouse button press.
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._apply_brush(event.pos())
        super().mousePressEvent(event)

    # Handle mouse movement. Updates brush preview position, applies brush if left button is held.
    def mouseMoveEvent(self, event):
        if self.show_brush_preview:
            self.last_mouse_pos = event.pos()
            self.update()
        if event.buttons() & Qt.LeftButton:
            self._apply_brush(event.pos())
        super().mouseMoveEvent(event)

    # Handle mouse enter/leave events. Shows/hides brush preview cursor when mouse enters/leaves widget.
    def enterEvent(self, event):
        if self.painting_enabled:
            self.show_brush_preview = True
            self.update()
        super().enterEvent(event)
    
    # Handle mouse enter/leave events. Shows/hides brush preview cursor when mouse enters/leaves widget.
    def leaveEvent(self, event):
        self.show_brush_preview = False
        self.last_mouse_pos = None
        self.update()
        super().leaveEvent(event)

    # Custom paint handler to draw brush preview. Draws red circular outline at mouse position to preview brush size (only when painting enabled).
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.show_brush_preview or self.last_mouse_pos is None:
            return

        mapped = self._map_to_pixmap_coords(self.last_mouse_pos)
        if mapped is None:
            return

        x, y, pixmap_width, pixmap_height = mapped
        if pixmap_width == 0 or pixmap_height == 0:
            return

        if self.mask_size is None:
            return

        mask_width, mask_height = self.mask_size
        scale_x = mask_width / pixmap_width
        scale_y = mask_height / pixmap_height
        radius_mask = self.brush_size * (scale_x + scale_y) / 4
        radius = max(1, int((radius_mask / scale_x + radius_mask / scale_y) / 2))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(255, 0, 0, 200))
        painter.setBrush(Qt.NoBrush)

        x_offset = max(0, (self.width() - pixmap_width) // 2)
        y_offset = max(0, (self.height() - pixmap_height) // 2)
        painter.drawEllipse(x_offset + x - radius, y_offset + y - radius, radius * 2, radius * 2)
        painter.end()

# Responsible for the page for editing a selected image
class ImageEditPage(QWidget):
    
    back_clicked = pyqtSignal()
    
    # nitialize the image editing interface.
    # Loads original image, initializes edit state, creates UI with dual-pane view and technique panel.
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.original_pixmap = load_pixmap_with_orientation(image_path)
        self.edited_pixmap = self.original_pixmap.copy()
        self.selected_technique = None
        self.slider_widgets = {} 
        self.white_balance_method = None
        self.zoom_factor = 1.0
        self.inpainting_mask = None
        self.inpainting_enabled = False
        self.edit_history = []
        self.last_applied_params = {}
        self.white_balance_toggle = None
        self.super_resolution_method = None
        self.super_resolution_toggle = None
        self.init_ui()
    
    # Builds the complete editing interface layout.
    def init_ui(self):
        """Initialize the edit page UI."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Left side - images (66% width)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # Original image section
        original_label = QLabel("ORIGINAL")
        original_label.setFont(QFont('Arial', 11, QFont.Bold))
        original_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(original_label)
        
        self.original_image_display = QLabel()
        self.original_image_display.setAlignment(Qt.AlignCenter)
        self.original_image_display.setStyleSheet("border: 1px solid #cccccc;")

        self.original_scroll_area = QScrollArea()
        self.original_scroll_area.setWidgetResizable(True)
        self.original_scroll_area.setWidget(self.original_image_display)
        self.original_scroll_area.setMinimumHeight(250)
        left_layout.addWidget(self.original_scroll_area, 1)
        
        # Edited image section
        edited_label = QLabel("EDITED")
        edited_label.setFont(QFont('Arial', 11, QFont.Bold))
        edited_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(edited_label)
        
        self.edited_image_display = MaskableImageLabel()
        self.edited_image_display.setAlignment(Qt.AlignCenter)
        self.edited_image_display.setStyleSheet("border: 1px solid #cccccc;")
        self.edited_image_display.set_apply_brush_callback(self.apply_inpainting_brush)

        self.edited_scroll_area = QScrollArea()
        self.edited_scroll_area.setWidgetResizable(True)
        self.edited_scroll_area.setWidget(self.edited_image_display)
        self.edited_scroll_area.setMinimumHeight(250)
        left_layout.addWidget(self.edited_scroll_area, 1)

        # Sync scroll between original and edited views
        self._sync_scroll = False
        self.original_scroll_area.verticalScrollBar().valueChanged.connect(
            lambda value: self.sync_scrollbars(value, source="original")
        )
        self.edited_scroll_area.verticalScrollBar().valueChanged.connect(
            lambda value: self.sync_scrollbars(value, source="edited")
        )
        self.original_scroll_area.horizontalScrollBar().valueChanged.connect(
            lambda value: self.sync_scrollbars(value, source="original", horizontal=True)
        )
        self.edited_scroll_area.horizontalScrollBar().valueChanged.connect(
            lambda value: self.sync_scrollbars(value, source="edited", horizontal=True)
        )
        
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, 2)  # 66% width
        
        # Right side - techniques and parameters (33% width)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        techniques_label = QLabel("TECHNIQUES")
        techniques_label.setFont(QFont('Arial', 11, QFont.Bold))
        techniques_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(techniques_label)

        # Return and Zoom row
        top_controls_row = QHBoxLayout()

        return_button = QPushButton("Return")
        return_button.setFont(QFont('Arial', 11, QFont.Bold))
        return_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #5a5a5a;
                border-radius: 8px;
                background-color: #3a3a3a;
                color: #e0e0e0;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #2f2f2f;
            }
        """)
        return_button.clicked.connect(self.back_clicked.emit)

        # Zoom control
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout()
        zoom_layout.setContentsMargins(8, 8, 8, 8)
        zoom_layout.setSpacing(6)

        zoom_row = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)
        self.zoom_slider.setMaximum(2000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSingleStep(10)
        self.zoom_slider.installEventFilter(self)

        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setMinimumWidth(50)
        self.zoom_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.zoom_slider.valueChanged.connect(
            lambda v: self.on_zoom_changed(v, self.zoom_value_label)
        )

        zoom_row.addWidget(self.zoom_slider)
        zoom_row.addWidget(self.zoom_value_label)
        zoom_layout.addLayout(zoom_row)

        zoom_group.setLayout(zoom_layout)
        top_controls_row.addWidget(return_button, 1)
        top_controls_row.addWidget(zoom_group, 4)
        right_layout.addLayout(top_controls_row)
        
        # Techniques list
        self.techniques_list = QListWidget()
        self.techniques_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        self.techniques_list.itemClicked.connect(self.on_technique_selected)
        
        # Load techniques
        techniques = self.load_techniques()
        for technique in techniques:
            self.techniques_list.addItem(technique)
        
        right_layout.addWidget(self.techniques_list, 1)
        
        # Parameters section (initially hidden)
        self.parameters_scroll = QScrollArea()
        self.parameters_scroll.setWidgetResizable(True)
        self.parameters_widget = QWidget()
        self.parameters_layout = QVBoxLayout()
        self.parameters_widget.setLayout(self.parameters_layout)
        self.parameters_scroll.setWidget(self.parameters_widget)
        self.parameters_scroll.setVisible(False)
        right_layout.addWidget(self.parameters_scroll, 1)
        
        # Revert + Apply row (initially hidden)
        action_row = QHBoxLayout()

        self.revert_button = QPushButton("Revert")
        self.revert_button.setMinimumHeight(35)
        self.revert_button.setFont(QFont('Arial', 11, QFont.Bold))
        self.revert_button.clicked.connect(self.on_revert_last_action)
        self.revert_button.setVisible(True)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setMinimumHeight(35)
        self.apply_button.setFont(QFont('Arial', 11, QFont.Bold))
        self.apply_button.clicked.connect(self.on_apply_technique)
        self.apply_button.setVisible(False)

        action_row.addWidget(self.revert_button, 1)
        action_row.addWidget(self.apply_button, 1)
        right_layout.addLayout(action_row)
        
        # Save button (initially hidden)
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumHeight(35)
        self.save_button.setFont(QFont('Arial', 11, QFont.Bold))
        self.save_button.clicked.connect(self.on_save_image)
        self.save_button.setVisible(False)
        right_layout.addWidget(self.save_button)
        
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, 1)  # 33% width
        
        self.setLayout(main_layout)
        QTimer.singleShot(0, lambda: self.on_zoom_changed(self.zoom_slider.value(), self.zoom_value_label))
        QTimer.singleShot(0, lambda: return_button.setFixedHeight(zoom_group.sizeHint().height()))
    
    def load_techniques(self):
        """Load available techniques from the techniques folder."""
        techniques = []
        if TECHNIQUES_PATH.exists():
            for file in TECHNIQUES_PATH.glob('*.py'):
                if file.name not in ('__init__.py', 'base.py'):
                    technique_name = file.stem.replace('_', ' ').title()
                    techniques.append(technique_name)
        return sorted(techniques)
    
    def on_technique_selected(self, item):
        """Handle technique selection."""
        technique_name = item.text()
        self.selected_technique = technique_name
        self.show_technique_parameters(technique_name)
        self.apply_button.setVisible(True)
    
    def show_technique_parameters(self, technique_name: str):
        """Show parameters for the selected technique."""
        # Clear previous parameters
        while self.parameters_layout.count():
            item = self.parameters_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        
        # Clear slider widgets dictionary
        self.slider_widgets = {}
        self.white_balance_method = None
        self.white_balance_toggle = None
        self.super_resolution_method = None
        self.super_resolution_toggle = None
        self.inpainting_enabled = False
        self.edited_image_display.set_painting_enabled(False)
        
        # Define parameters for each technique
        technique_params = {
            'Denoise': {
                'Simplified': {
                    'h': {'min': 1, 'max': 50, 'default': 10, 'step': 1}
                },
                'Advanced': {
                    'h_color': {'min': 1, 'max': 50, 'default': 10, 'step': 1},
                    'template_window_size': {'min': 3, 'max': 21, 'default': 7, 'step': 2},
                    'search_window_size': {'min': 3, 'max': 50, 'default': 21, 'step': 2}
                }
            },
            'Colour Mixing': {
                'Simplified': {
                    'reds': {'min': -100, 'max': 100, 'default': 0, 'step': 1},
                    'oranges': {'min': -100, 'max': 100, 'default': 0, 'step': 1},
                    'greens': {'min': -100, 'max': 100, 'default': 0, 'step': 1},
                    'aquas': {'min': -100, 'max': 100, 'default': 0, 'step': 1},
                    'blues': {'min': -100, 'max': 100, 'default': 0, 'step': 1},
                    'purples': {'min': -100, 'max': 100, 'default': 0, 'step': 1},
                    'magentas': {'min': -100, 'max': 100, 'default': 0, 'step': 1},
                    'yellows': {'min': -100, 'max': 100, 'default': 0, 'step': 1}
                }
            },
            'White Balancing': {},
            'Super Resolution': {},
            'Inpainting': {
                'Simplified': {
                    'brush_size': {'min': 5, 'max': 100, 'default': 25, 'step': 1}
                }
            }
        }
        
        params = technique_params.get(technique_name, {})

        if technique_name == 'White Balancing':
            method_group = QGroupBox("Method")
            method_layout = QVBoxLayout()

            gray_world_toggle = QCheckBox(" Via Gray World Assumption")
            gray_world_toggle.setTristate(False)

            gray_world_toggle.setChecked(False)
            self.white_balance_method = None
            self.white_balance_toggle = gray_world_toggle

            gray_world_toggle.toggled.connect(
                lambda checked: self.set_white_balance_method("gray_world" if checked else None)
            )
            method_layout.addWidget(gray_world_toggle)
            method_group.setLayout(method_layout)
            self.parameters_layout.addWidget(method_group)

            self.parameters_layout.addStretch()
            self.parameters_scroll.setVisible(True)
            return

        if technique_name == 'Super Resolution':
            method_group = QGroupBox("Method")
            method_layout = QVBoxLayout()

            method_button_group = QButtonGroup(self)
            method_button_group.setExclusive(True)

            upsampling_toggle = QCheckBox(" Via Upsampling")
            upsampling_toggle.setTristate(False)
            upsampling_toggle.setChecked(False)

            opencv_toggle = QCheckBox(" Via OpenCV Method")
            opencv_toggle.setTristate(False)
            opencv_toggle.setChecked(False)

            deep_learning_toggle = QCheckBox(" Via Deeplearning")
            deep_learning_toggle.setTristate(False)
            deep_learning_toggle.setChecked(False)

            upsampling_toggle.toggled.connect(
                lambda checked: self.set_super_resolution_method("upsampling" if checked else None)
            )
            opencv_toggle.toggled.connect(
                lambda checked: self.set_super_resolution_method("opencv" if checked else None)
            )
            deep_learning_toggle.toggled.connect(
                lambda checked: self.set_super_resolution_method("deep_learning" if checked else None)
            )

            method_button_group.addButton(upsampling_toggle)
            method_button_group.addButton(opencv_toggle)
            method_button_group.addButton(deep_learning_toggle)

            self.super_resolution_toggle = {
                "upsampling": upsampling_toggle,
                "opencv": opencv_toggle,
                "deep_learning": deep_learning_toggle
            }
            self.super_resolution_method = None

            method_layout.addWidget(upsampling_toggle)
            method_layout.addWidget(opencv_toggle)
            method_layout.addWidget(deep_learning_toggle)
            method_group.setLayout(method_layout)
            self.parameters_layout.addWidget(method_group)

            self.parameters_layout.addStretch()
            self.parameters_scroll.setVisible(True)
            return

        if technique_name == 'Inpainting':
            self.inpainting_enabled = True
            self.edited_image_display.set_painting_enabled(True)
            if self.inpainting_mask is None:
                self.initialize_inpainting_mask()

            inpaint_controls = QGroupBox("Inpainting Tools")
            inpaint_layout = QVBoxLayout()

            clear_mask_button = QPushButton("Clear Mask")
            clear_mask_button.setMinimumHeight(30)
            clear_mask_button.clicked.connect(self.clear_inpainting_mask)

            inpaint_layout.addWidget(clear_mask_button)
            inpaint_controls.setLayout(inpaint_layout)
            self.parameters_layout.addWidget(inpaint_controls)
        
        # Create Simplified section (expanded by default)
        if 'Simplified' in params:
            simplified_group = QGroupBox("Simplified")
            simplified_group.setCheckable(True)
            simplified_group.setChecked(True)
            simplified_layout = QVBoxLayout()
            
            for param_name, param_config in params['Simplified'].items():
                simplified_layout.addWidget(self.create_slider_widget(
                    param_name, param_config
                ))
            
            simplified_group.setLayout(simplified_layout)
            self.parameters_layout.addWidget(simplified_group)
        
        # Create Advanced section (collapsed by default)
        if 'Advanced' in params:
            advanced_group = QGroupBox("Advanced")
            advanced_group.setCheckable(True)
            advanced_group.setChecked(False)
            advanced_layout = QVBoxLayout()
            
            for param_name, param_config in params['Advanced'].items():
                advanced_layout.addWidget(self.create_slider_widget(
                    param_name, param_config
                ))
            
            advanced_group.setLayout(advanced_layout)
            self.parameters_layout.addWidget(advanced_group)
        
        # Add stretch at the end
        self.parameters_layout.addStretch()
        
        # Show parameters section
        self.parameters_scroll.setVisible(True)
        self.last_applied_params = self.capture_current_parameters()

    def set_white_balance_method(self, method: Optional[str]):
        """Set the selected white balance method."""
        self.white_balance_method = method

    def set_super_resolution_method(self, method: Optional[str]):
        """Set the selected super resolution method."""
        self.super_resolution_method = method

    def initialize_inpainting_mask(self):
        """Initialize the inpainting mask to image size."""
        height = self.original_pixmap.height()
        width = self.original_pixmap.width()
        self.inpainting_mask = np.zeros((height, width), dtype=np.uint8)
        self.edited_image_display.set_mask_size(width, height)

    def apply_inpainting_brush(self, x: int, y: int, pixmap_width: int, pixmap_height: int, brush_size: int):
        """Apply brush stroke to the inpainting mask."""
        if self.inpainting_mask is None:
            self.initialize_inpainting_mask()

        mask_height, mask_width = self.inpainting_mask.shape
        if pixmap_width == 0 or pixmap_height == 0:
            return

        scale_x = mask_width / pixmap_width
        scale_y = mask_height / pixmap_height

        img_x = int(x * scale_x)
        img_y = int(y * scale_y)

        radius = max(1, int(brush_size * (scale_x + scale_y) / 4))
        cv2.circle(self.inpainting_mask, (img_x, img_y), radius, 255, thickness=-1)
        self.update_images()

    def clear_inpainting_mask(self):
        """Clear the inpainting mask."""
        if self.inpainting_mask is None:
            self.initialize_inpainting_mask()
        else:
            self.inpainting_mask.fill(0)
        self.update_images()

    def pixmap_to_bgr(self, pixmap: QPixmap) -> np.ndarray:
        """Convert QPixmap to BGR numpy array."""
        q_image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width = q_image.width()
        height = q_image.height()
        bytes_per_line = q_image.bytesPerLine()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())

        arr = np.array(ptr).reshape(height, bytes_per_line)
        arr = arr[:, :width * 3].reshape(height, width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    def create_slider_widget(self, param_name: str, config: dict) -> QWidget:
        """Create a slider widget for a parameter."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)
        
        # Parameter label
        label = QLabel(param_name)
        label.setFont(QFont('Arial', 10))
        layout.addWidget(label)
        
        # Slider container
        slider_container = QHBoxLayout()
        slider_container.setContentsMargins(0, 0, 0, 0)
        slider_container.setSpacing(10)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(config['min'])
        slider.setMaximum(config['max'])
        slider.setValue(config['default'])
        slider.setProperty("default_value", config['default'])
        slider.setSingleStep(config.get('step', 1))
        slider_container.addWidget(slider)
        
        # Value label
        value_label = QLabel(str(config['default']))
        value_label.setMinimumWidth(40)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setFont(QFont('Arial', 9))
        slider_container.addWidget(value_label)
        
        # Connect slider to value label
        def on_value_changed(v):
            value_label.setText(str(v))
            if param_name == "brush_size":
                self.edited_image_display.set_brush_size(v)

        slider.valueChanged.connect(on_value_changed)
        if param_name == "brush_size":
            self.edited_image_display.set_brush_size(config['default'])
        
        slider_widget = QWidget()
        slider_widget.setLayout(slider_container)
        layout.addWidget(slider_widget)
        
        container.setLayout(layout)
        
        # Store slider reference using the parameter name
        self.slider_widgets[param_name] = slider
        
        return container

    def on_zoom_changed(self, value: int, label: QLabel):
        """Handle zoom slider changes."""
        adjusted_value = value - 1
        self.zoom_factor = max(0.5, min(20.0, adjusted_value / 100.0))
        label.setText(f"{value}%")
        self.update_images()

    def eventFilter(self, source, event):
        if source == self.zoom_slider and event.type() == event.MouseButtonDblClick:
            self.zoom_slider.setValue(100)
            return True
        return super().eventFilter(source, event)

    def sync_scrollbars(self, value: int, source: str, horizontal: bool = False):
        """Sync scroll positions between original and edited views."""
        if self._sync_scroll:
            return
        self._sync_scroll = True
        try:
            if horizontal:
                if source == "original":
                    self.edited_scroll_area.horizontalScrollBar().setValue(value)
                else:
                    self.original_scroll_area.horizontalScrollBar().setValue(value)
            else:
                if source == "original":
                    self.edited_scroll_area.verticalScrollBar().setValue(value)
                else:
                    self.original_scroll_area.verticalScrollBar().setValue(value)
        finally:
            self._sync_scroll = False
    
    def update_images(self):
        """Update the displayed images."""
        # Scale images to fit the available space with zoom
        original_base_height = max(250, self.original_scroll_area.viewport().height())
        edited_base_height = max(250, self.edited_scroll_area.viewport().height())

        original_target_height = int(original_base_height * self.zoom_factor)
        edited_target_height = int(edited_base_height * self.zoom_factor)

        original_scaled = self.original_pixmap.scaledToHeight(
            original_target_height,
            Qt.SmoothTransformation
        )
        edited_scaled = self.edited_pixmap.scaledToHeight(
            edited_target_height,
            Qt.SmoothTransformation
        )

        if self.inpainting_enabled and self.inpainting_mask is not None:
            mask_scaled = cv2.resize(
                self.inpainting_mask,
                (edited_scaled.width(), edited_scaled.height()),
                interpolation=cv2.INTER_NEAREST
            )
            overlay = np.zeros((mask_scaled.shape[0], mask_scaled.shape[1], 4), dtype=np.uint8)
            overlay[..., 0] = 255
            overlay[..., 3] = np.where(mask_scaled > 0, 120, 0)

            overlay_img = QImage(
                overlay.data,
                overlay.shape[1],
                overlay.shape[0],
                overlay.strides[0],
                QImage.Format_RGBA8888
            )
            composed = QPixmap(edited_scaled)
            painter = QPainter(composed)
            painter.drawImage(0, 0, overlay_img)
            painter.end()
            edited_scaled = composed
        
        self.original_image_display.setPixmap(original_scaled)
        self.edited_image_display.setPixmap(edited_scaled)
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_images()
    
    def on_apply_technique(self):
        """Apply the selected technique with current parameters."""
        if not self.selected_technique:
            QMessageBox.warning(self, "No Technique", "Please select a technique first.")
            return
        
        try:
            self.edit_history.append({
                "pixmap": self.edited_pixmap.copy(),
                "params": self.last_applied_params
            })

            # Default to original image for techniques unless overridden
            bgr_image = self.pixmap_to_bgr(self.original_pixmap)
            
            # Get current parameter values from sliders
            params = {}
            for param_name, slider in self.slider_widgets.items():
                params[param_name] = slider.value()
            
            # Apply the technique
            if self.selected_technique == 'Denoise':
                from techniques.denoise import Denoising
                denoiser = Denoising()
                denoiser.h = params.get('h', 10)
                denoiser.h_color = params.get('h_color', 10)
                denoiser.template_window_size = params.get('template_window_size', 7)
                denoiser.search_window_size = params.get('search_window_size', 21)
                
                result_image = denoiser.apply(bgr_image)
            elif self.selected_technique == 'Colour Mixing':
                from techniques.colour_mixing import ColourMixing
                colour_mixer = ColourMixing()
                colour_mixer.reds = params.get('reds', 0)
                colour_mixer.oranges = params.get('oranges', 0)
                colour_mixer.greens = params.get('greens', 0)
                colour_mixer.aquas = params.get('aquas', 0)
                colour_mixer.blues = params.get('blues', 0)
                colour_mixer.purples = params.get('purples', 0)
                colour_mixer.magentas = params.get('magentas', 0)
                colour_mixer.yellows = params.get('yellows', 0)
                
                result_image = colour_mixer.apply(bgr_image)
            elif self.selected_technique == 'White Balancing':
                if not self.white_balance_method:
                    QMessageBox.warning(
                        self,
                        "No Method Selected",
                        "Please select a white balance method first."
                    )
                    return

                from techniques.white_balancing import WhiteBalancing
                white_balancer = WhiteBalancing()
                if self.white_balance_method == "gray_world":
                    result_image = white_balancer.apply_gray_world(bgr_image)
                else:
                    QMessageBox.warning(self, "No Method Selected", "Please select a white balance method first.")
                    return
            elif self.selected_technique == 'Super Resolution':
                if not self.super_resolution_method:
                    QMessageBox.warning(
                        self,
                        "No Method Selected",
                        "Please select a super resolution method first."
                    )
                    return

                from techniques.super_resolution import SuperResolution
                super_res = SuperResolution()
                if self.super_resolution_method == "upsampling":
                    result_image = super_res.apply_upsampling(bgr_image)
                elif self.super_resolution_method == "opencv":
                    result_image = super_res.apply_opencv_method(bgr_image)
                elif self.super_resolution_method == "deep_learning":
                    result_image = super_res.apply_deep_learning(bgr_image)
                else:
                    QMessageBox.warning(self, "No Method Selected", "Please select a super resolution method first.")
                    return
            elif self.selected_technique == 'Inpainting':
                if self.inpainting_mask is None or not np.any(self.inpainting_mask):
                    QMessageBox.warning(self, "No Mask", "Please paint a mask on the edited image first.")
                    return

                from techniques.inpainting import Inpainting
                inpainting = Inpainting()
                bgr_image = self.pixmap_to_bgr(self.edited_pixmap)
                result_image = inpainting.apply_with_mask(bgr_image, self.inpainting_mask)
                self.clear_inpainting_mask()
            else:
                QMessageBox.warning(self, "Error", f"Technique {self.selected_technique} not implemented.")
                return
            
            # Convert result back to QPixmap
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            
            # Make a copy of the data to ensure it persists
            result_rgb_copy = result_rgb.copy()
            bytes_per_line = 3 * w
            q_img = QImage(result_rgb_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and update
            self.edited_pixmap = QPixmap.fromImage(q_img.copy())
            
            self.update_images()
            self.save_button.setVisible(True)
            self.last_applied_params = self.capture_current_parameters()
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to apply technique: {str(e)}\n\n{traceback.format_exc()}")

    def on_revert_last_action(self):
        """Revert the last applied edit and reset parameters to defaults."""
        if not self.edit_history:
            QMessageBox.information(self, "No Revert", "There is no previous edit to revert.")
            return

        last_state = self.edit_history.pop()
        self.edited_pixmap = last_state["pixmap"]
        self.update_images()
        self.restore_parameters(last_state.get("params", {}))
        self.last_applied_params = last_state.get("params", {})

    def restore_parameters(self, params: dict):
        """Restore parameter values from a saved state."""
        for name, value in params.get("sliders", {}).items():
            slider = self.slider_widgets.get(name)
            if slider:
                slider.setValue(value)
        
        if "white_balance_checked" in params and self.white_balance_toggle is not None:
            self.white_balance_toggle.setChecked(params["white_balance_checked"])
        
        if "super_resolution_method" in params and self.super_resolution_toggle is not None:
            self.super_resolution_method = params["super_resolution_method"]

    def on_reset_parameters(self):
        """Reset all parameters to their default values."""
        for name, slider in self.slider_widgets.items():
            default_value = slider.property("default_value")
            if default_value is not None:
                slider.setValue(default_value)

        if self.white_balance_toggle is not None:
            self.white_balance_toggle.setChecked(False)
            self.white_balance_method = None

        if self.inpainting_enabled:
            self.clear_inpainting_mask()

    def capture_current_parameters(self) -> dict:
        """Capture current parameter values for undo."""
        params = {
            "sliders": {},
            "white_balance_checked": None
        }

        for name, slider in self.slider_widgets.items():
            params["sliders"][name] = slider.value()

        if self.white_balance_toggle is not None:
            params["white_balance_checked"] = self.white_balance_toggle.isChecked()

        if self.super_resolution_toggle is not None:
            params["super_resolution_method"] = self.super_resolution_method

        return params

    def restore_parameters(self, params: dict):
        """Restore parameters from undo snapshot."""
        slider_values = params.get("sliders", {})
        for name, slider in self.slider_widgets.items():
            if name in slider_values:
                slider.setValue(slider_values[name])

        if self.white_balance_toggle is not None:
            checked = params.get("white_balance_checked")
            if checked is not None:
                self.white_balance_toggle.setChecked(checked)

        if self.super_resolution_toggle is not None:
            method = params.get("super_resolution_method")
            if method in self.super_resolution_toggle:
                self.super_resolution_toggle[method].setChecked(True)
    
    def on_save_image(self):
        """Save the edited image to the processed folder."""
        if not self.selected_technique:
            QMessageBox.warning(self, "No Technique", "Please select a technique first.")
            return
        
        try:
            # Create the processed folder for the technique
            technique_folder_name = self.selected_technique.lower()
            processed_folder = PROJECT_ROOT / "data" / "processed" / technique_folder_name
            processed_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename with timestamp (always save as PNG)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = Path(self.image_path).stem
            filename = f"{original_filename}_{timestamp}.png"
            
            # Save the image
            file_path = processed_folder / filename
            
            # Convert QPixmap to cv2 image and save (match preview conversion)
            q_image = self.edited_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            width = q_image.width()
            height = q_image.height()
            bytes_per_line = q_image.bytesPerLine()
            ptr = q_image.bits()
            ptr.setsize(q_image.byteCount())
            
            arr = np.array(ptr).reshape(height, bytes_per_line)
            arr = arr[:, :width * 3].reshape(height, width, 3)
            
            # Convert RGB to BGR for saving
            bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(file_path), bgr_image)
            
            # Extract and save metadata from original image
            metadata = extract_image_metadata(self.image_path)
            metadata["processed_info"] = {
                "technique_applied": self.selected_technique,
                "processed_date": datetime.now().isoformat(),
                "output_file": str(file_path),
                "output_size": f"{width} x {height}",
            }
            
            # Save metadata to data/metadata folder
            metadata_folder = PROJECT_ROOT / "data" / "metadata"
            metadata_folder.mkdir(parents=True, exist_ok=True)
            metadata_file_path = metadata_folder / f"{original_filename}_{timestamp}.json"
            
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            metadata_path = metadata_file_path
            
            QMessageBox.information(
                self,
                "Image Saved",
                f"Image saved to:\n{file_path}\n\nMetadata saved to:\n{metadata_path}"
            )
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}\n\n{traceback.format_exc()}")


class ImageEditorGUI(QMainWindow):
    """Main GUI window for the Photography Editor."""
    
    def __init__(self):
        super().__init__()
        self.selected_image = None
        self.image_cards = []
        self.stacked_widget = QStackedWidget()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the main UI."""
        self.setWindowTitle("Photography Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create gallery page
        self.gallery_page = self.create_gallery_page()
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.gallery_page)
        
        # Set stacked widget as central widget
        self.setCentralWidget(self.stacked_widget)
    
    def create_gallery_page(self):
        """Create the gallery/image selection page."""
        gallery_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create title
        title = QLabel("Raw Images")
        title_font = QFont('Arial', 16, QFont.Bold)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Create scroll area for image grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create grid widget and layout
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        
        # Load and display images
        image_files = sorted(RAW_IMAGES_PATH.glob('*'))
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = [f for f in image_files if f.suffix.lower() in valid_extensions]
        
        if not image_files:
            no_images_label = QLabel("No images found in /data/raw")
            grid_layout.addWidget(no_images_label, 0, 0)
        else:
            for idx, image_file in enumerate(image_files):
                card = ImageCard(str(image_file))
                card.clicked.connect(self.on_image_selected)
                self.image_cards.append(card)
                
                row = idx // 3
                col = idx % 3
                grid_layout.addWidget(card, row, col)
        
        # Add stretch at the end to push everything to the top
        spacer = QSpacerItem(
            20, 40,
            QSizePolicy.Minimum,
            QSizePolicy.Expanding
        )
        grid_layout.addItem(spacer, grid_layout.rowCount(), 0, 1, 3)
        
        grid_widget.setLayout(grid_layout)
        scroll_area.setWidget(grid_widget)
        main_layout.addWidget(scroll_area)
        
        # Create bottom button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.edit_button = QPushButton("Edit Image")
        self.edit_button.setMinimumWidth(150)
        self.edit_button.setMinimumHeight(40)
        self.edit_button.setFont(QFont('Arial', 11, QFont.Bold))
        self.edit_button.clicked.connect(self.on_edit_image)
        self.edit_button.setEnabled(False)
        button_layout.addWidget(self.edit_button)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # Set the gallery widget layout
        gallery_widget.setLayout(main_layout)
        return gallery_widget
    
    def on_image_selected(self, image_path: str):
        """Handle image selection."""
        # Deselect previous selection
        if self.selected_image:
            for card in self.image_cards:
                if card.image_path == self.selected_image:
                    card.set_selected(False)
        
        # Select new image
        self.selected_image = image_path
        for card in self.image_cards:
            if card.image_path == image_path:
                card.set_selected(True)
        
        self.edit_button.setEnabled(True)
    
    def on_edit_image(self):
        """Handle the Edit Image button click."""
        if self.selected_image:
            # Create and show the edit page
            edit_page = ImageEditPage(self.selected_image)
            edit_page.back_clicked.connect(self.on_back_to_gallery)
            
            # Add edit page to stacked widget
            self.stacked_widget.addWidget(edit_page)
            self.stacked_widget.setCurrentWidget(edit_page)
        else:
            QMessageBox.warning(
                self,
                "No Image Selected",
                "Please select an image first."
            )
    
    def on_back_to_gallery(self):
        """Handle navigation back to gallery."""
        # Remove the current edit page and switch back to gallery
        current_widget = self.stacked_widget.currentWidget()
        if isinstance(current_widget, ImageEditPage):
            self.stacked_widget.removeWidget(current_widget)
        self.stacked_widget.setCurrentWidget(self.gallery_page)


def main():
    """Entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = ImageEditorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
