from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QFrame, QSpacerItem, QSizePolicy, QScrollArea)
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt
from Functions import ConfigFunctions
import os


class ModernCard(QFrame):
    """Modern card-style button with icon, title, and description"""
    
    def __init__(self, title, description, icon_path=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                padding: 20px;
            }
            QFrame:hover {
                border: 2px solid #2196F3;
                background-color: #f5f5f5;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        self.setLayout(layout)
        
        # Icon
        if icon_path and os.path.exists(icon_path):
            icon_label = QLabel()
            pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(scaled_pixmap)
            icon_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1976D2;")
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(description)
        desc_font = QFont()
        desc_font.setPointSize(11)
        desc_label.setFont(desc_font)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: #666;")
        layout.addWidget(desc_label)
        
        # Action button
        self.action_btn = QPushButton("Open →")
        self.action_btn.setMinimumHeight(40)
        self.action_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(self.action_btn)
        
        self.setMinimumHeight(320)
        self.setMaximumWidth(400)
        self.setCursor(Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        """Make the whole card clickable"""
        self.action_btn.click()
        super().mousePressEvent(event)


class StartPanel(QWidget):

    def __init__(self, parent):
        """Constructor"""
        super().__init__(parent)
        
        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width = configs['window_width']
        self.window_height = configs['window_height']
        
        # Set uniform background for all widgets
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
            }
            QLabel {
                background-color: transparent;
                color: inherit;
            }
            QVBoxLayout, QHBoxLayout {
                background-color: transparent;
            }
            QSpacerItem {
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: #f5f7fa;
            }
        """)
        
        # Main layout (wrapper for scroll area)
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(wrapper_layout)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        wrapper_layout.addWidget(scroll_area)
        
        # Content widget inside scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        # Main layout for content
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(30)
        content_widget.setLayout(main_layout)
        
        # Header section
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)
        
        # Main title
        title = QLabel("ALMA")
        title_font = QFont()
        title_font.setPointSize(48)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1976D2;")
        header_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Automated Limb Motion Analysis")
        subtitle_font = QFont()
        subtitle_font.setPointSize(16)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666;")
        header_layout.addWidget(subtitle)
        
        main_layout.addLayout(header_layout)
        main_layout.addSpacing(20)
        
        # Intro text
        intro = QLabel("Choose your workflow:")
        intro_font = QFont()
        intro_font.setPointSize(14)
        intro.setFont(intro_font)
        intro.setAlignment(Qt.AlignCenter)
        intro.setStyleSheet("color: #333;")
        main_layout.addWidget(intro)
        
        # Cards layout (main analysis)
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(30)
        
        # Footfall Detection Card
        self.footfall_card = ModernCard(
            "Footfall Detection",
            "Ladder Rung Analysis - Detect and validate footfalls, slips, and falls with interactive timeline visualization",
            "./Resources/image_ladder_rung.png"
        )
        self.footfall_card.action_btn.clicked.connect(parent.on_analyze_footfall)
        cards_layout.addWidget(self.footfall_card)
        
        # Kinematics Card
        self.kinematics_card = ModernCard(
            "Gait Kinematics",
            "Extract comprehensive gait parameters including stride metrics, joint angles, and motion variability",
            "./Resources/image_kinematics.png"
        )
        self.kinematics_card.action_btn.clicked.connect(parent.on_analyze_stride)
        cards_layout.addWidget(self.kinematics_card)
        
        # Center the cards
        cards_container = QHBoxLayout()
        cards_container.addStretch()
        cards_container.addLayout(cards_layout)
        cards_container.addStretch()
        
        main_layout.addLayout(cards_container)
        
        # Data Analysis section (more prominent)
        main_layout.addSpacing(30)
        
        analysis_header = QLabel("Data Analysis Tools:")
        analysis_header_font = QFont()
        analysis_header_font.setPointSize(12)
        analysis_header_font.setBold(True)
        analysis_header.setFont(analysis_header_font)
        analysis_header.setAlignment(Qt.AlignCenter)
        analysis_header.setStyleSheet("color: #555;")
        main_layout.addWidget(analysis_header)
        
        main_layout.addSpacing(10)
        
        # Analysis buttons layout
        analysis_layout = QHBoxLayout()
        analysis_layout.setSpacing(20)
        
        rf_btn = QPushButton("Random Forest Classification")
        rf_btn.setMinimumHeight(45)
        rf_btn.setStyleSheet(self._get_analysis_button_style('#4CAF50'))
        rf_btn.clicked.connect(parent.on_random_forest)
        
        pca_btn = QPushButton("Principal Component Analysis")
        pca_btn.setMinimumHeight(45)
        pca_btn.setStyleSheet(self._get_analysis_button_style('#9C27B0'))
        pca_btn.clicked.connect(parent.on_PCA)
        
        analysis_layout.addStretch()
        analysis_layout.addWidget(rf_btn)
        analysis_layout.addWidget(pca_btn)
        analysis_layout.addStretch()
        
        main_layout.addLayout(analysis_layout)
        main_layout.addStretch()
        
        # Footer with version
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        
        version_label = QLabel("v2.0")
        version_label.setStyleSheet("color: #999; font-size: 10px;")
        footer_layout.addWidget(version_label)
        
        main_layout.addLayout(footer_layout)
    
    def _get_analysis_button_style(self, color):
        """Get stylesheet for analysis buttons"""
        hover_color = color + 'E0'  # Add transparency for hover
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color};
                opacity: 0.9;
            }}
            QPushButton:pressed {{
                background-color: {color};
                opacity: 0.8;
            }}
        """
    
    

