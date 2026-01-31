#!/usr/bin/env python3
"""
Photography Editor GUI

Main GUI interface for the Photography Editor application.
Displays raw images in a grid layout and allows image selection.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QScrollArea, QMessageBox, QSpacerItem,
    QSizePolicy, QStackedWidget, QListWidget, QListWidgetItem, QGroupBox,
    QSlider, QCheckBox, QButtonGroup
)
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer

# Get the path to the data/raw folder
PROJECT_ROOT = Path(__file__).parent.parent
RAW_IMAGES_PATH = PROJECT_ROOT / "data" / "raw"
TECHNIQUES_PATH = PROJECT_ROOT / "techniques"


# This section handles the initial page where you can select an image to edit
class ImageCard(QWidget):
    """A clickable image card widget."""
    
    clicked = pyqtSignal(str)  # Emits the image path when clicked
    
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.selected = False
        self.original_pixmap = QPixmap(self.image_path)
        self.init_ui()
    
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
    
    def crop_to_square(self, pixmap: QPixmap) -> QPixmap:
        """Crop pixmap to 1:1 aspect ratio from center."""
        width = pixmap.width()
        height = pixmap.height()
        
        # Determine the size of the square (use the smaller dimension)
        square_size = min(width, height)
        
        # Calculate the crop position to center the crop
        x = (width - square_size) // 2
        y = (height - square_size) // 2
        
        # Crop the pixmap
        return pixmap.copy(x, y, square_size, square_size)
    
    def resizeEvent(self, event):
        """Handle resize events to update image preview."""
        super().resizeEvent(event)
        self.update_image_preview()
    
    def update_style(self):
        """Update the card style based on selection state."""
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
    
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        self.clicked.emit(self.image_path)
    
    def set_selected(self, selected: bool):
        """Set the selection state of the card."""
        self.selected = selected
        self.update_style()


# This section handles the image editing page 
class ImageEditPage(QWidget):
    """Page for editing a selected image."""
    
    back_clicked = pyqtSignal()
    
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.original_pixmap = QPixmap(image_path)
        self.edited_pixmap = self.original_pixmap.copy()
        self.selected_technique = None
        self.slider_widgets = {}  # Store slider references for value retrieval
        self.white_balance_method = None
        self.zoom_factor = 1.0
        self.init_ui()
    
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
        
        self.edited_image_display = QLabel()
        self.edited_image_display.setAlignment(Qt.AlignCenter)
        self.edited_image_display.setStyleSheet("border: 1px solid #cccccc;")

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

        # Zoom control
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout()
        zoom_layout.setContentsMargins(8, 8, 8, 8)
        zoom_layout.setSpacing(6)

        zoom_row = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)
        self.zoom_slider.setMaximum(700)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSingleStep(10)

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
        right_layout.addWidget(zoom_group)
        
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
        
        # Apply button (initially hidden)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setMinimumHeight(35)
        self.apply_button.setFont(QFont('Arial', 11, QFont.Bold))
        self.apply_button.clicked.connect(self.on_apply_technique)
        self.apply_button.setVisible(False)
        right_layout.addWidget(self.apply_button)
        
        # Save button (initially hidden)
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumHeight(35)
        self.save_button.setFont(QFont('Arial', 11, QFont.Bold))
        self.save_button.clicked.connect(self.on_save_image)
        self.save_button.setVisible(False)
        right_layout.addWidget(self.save_button)
        
        # Back button
        back_button = QPushButton("Back")
        back_button.setMinimumHeight(35)
        back_button.setFont(QFont('Arial', 10))
        back_button.clicked.connect(self.back_clicked.emit)
        right_layout.addWidget(back_button)
        
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, 1)  # 33% width
        
        self.setLayout(main_layout)
        QTimer.singleShot(0, lambda: self.on_zoom_changed(self.zoom_slider.value(), self.zoom_value_label))
    
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
            'Colourmixing': {
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
            'White Balancing': {}
        }
        
        params = technique_params.get(technique_name, {})

        if technique_name == 'White Balancing':
            method_group = QGroupBox("Method")
            method_layout = QVBoxLayout()

            gray_world_toggle = QCheckBox(" Via Gray World Assumption")
            gray_world_toggle.setTristate(False)
            gray_world_toggle.setStyleSheet("""
                QCheckBox::indicator {
                    width: 50px;
                    height: 28px;
                }
                QCheckBox::indicator:unchecked {
                    border-radius: 14px;
                    background: #e5e5ea;
                    border: 1px solid #d1d1d6;
                }
                QCheckBox::indicator:checked {
                    border-radius: 14px;
                    background: #34c759;
                    border: 1px solid #30b753;
                }
            """)

            gray_world_toggle.setChecked(False)
            self.white_balance_method = None

            gray_world_toggle.toggled.connect(
                lambda checked: self.set_white_balance_method("gray_world" if checked else None)
            )
            method_layout.addWidget(gray_world_toggle)
            method_group.setLayout(method_layout)
            self.parameters_layout.addWidget(method_group)

            self.parameters_layout.addStretch()
            self.parameters_scroll.setVisible(True)
            return
        
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

    def set_white_balance_method(self, method: Optional[str]):
        """Set the selected white balance method."""
        self.white_balance_method = method
    
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
        slider.setSingleStep(config.get('step', 1))
        slider_container.addWidget(slider)
        
        # Value label
        value_label = QLabel(str(config['default']))
        value_label.setMinimumWidth(40)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setFont(QFont('Arial', 9))
        slider_container.addWidget(value_label)
        
        # Connect slider to value label
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
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
        self.zoom_factor = max(0.5, min(7.0, adjusted_value / 100.0))
        label.setText(f"{value}%")
        self.update_images()

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
            # Convert original QPixmap to QImage to numpy array (reset to original)
            q_image = self.original_pixmap.toImage()
            
            # Convert to RGB888 format to ensure consistent 3-channel format
            q_image = q_image.convertToFormat(QImage.Format_RGB888)
            
            width = q_image.width()
            height = q_image.height()
            bytes_per_line = q_image.bytesPerLine()
            
            # Extract pixel data
            ptr = q_image.bits()
            ptr.setsize(q_image.byteCount())
            
            # Account for padding in bytes per line
            arr = np.array(ptr).reshape(height, bytes_per_line)
            
            # Extract only the actual image data (remove padding)
            arr = arr[:, :width * 3].reshape(height, width, 3)
            
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
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
            elif self.selected_technique == 'Colourmixing':
                from techniques.colourMixing import ColourMixing
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
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to apply technique: {str(e)}\n\n{traceback.format_exc()}")
    
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
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = Path(self.image_path).stem
            file_extension = Path(self.image_path).suffix
            filename = f"{original_filename}_{timestamp}{file_extension}"
            
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
            
            QMessageBox.information(
                self,
                "Image Saved",
                f"Image saved to:\n{file_path}"
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
