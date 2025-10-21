from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QFileDialog, 
                               QMessageBox, QLineEdit, QGroupBox, QFrame,
                               QVBoxLayout, QHBoxLayout, QScrollArea)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from Functions import ConfigFunctions, DataAnalysisFunctions
import os


class StepCard(QFrame):
    """Modern step card for data analysis workflow"""
    
    def __init__(self, step_num, title, description, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            StepCard {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        self.setLayout(layout)
        
        # Step header
        header_layout = QHBoxLayout()
        step_badge = QLabel(str(step_num))
        step_badge.setStyleSheet("""
            background-color: #2196F3;
            color: white;
            border-radius: 15px;
            padding: 5px;
            font-weight: bold;
            min-width: 30px;
            max-width: 30px;
            min-height: 30px;
            max-height: 30px;
        """)
        step_badge.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(step_badge)
        
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #333;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(desc_label)
        
        # Content area (to be filled by parent)
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(8)
        layout.addLayout(self.content_layout)


class RandomForestPanel(QWidget):

    def __init__(self, parent):
        """Constructor"""
        super().__init__(parent)

        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width = configs['window_width']
        self.window_height = configs['window_height']
        self.dirname = os.getcwd()
        self.folder1 = ''
        self.folder2 = ''
        self.group_name_1 = ''
        self.group_name_2 = ''
        self.output_path = ''

        # Set background
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
            }
            QScrollArea {
                border: none;
                background-color: #f5f7fa;
            }
        """)
        
        # Wrapper layout for scroll area
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(wrapper_layout)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        wrapper_layout.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)
        content_widget.setLayout(main_layout)

        # Top bar with return button
        top_bar = QHBoxLayout()
        return_btn = QPushButton("← Return to Start Page")
        return_btn.setMinimumHeight(35)
        return_btn.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)
        return_btn.clicked.connect(lambda: parent.on_start())
        top_bar.addWidget(return_btn)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)
        
        title = QLabel("Random Forest Classification")
        title_font = QFont()
        title_font.setPointSize(28)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1976D2;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Classify animals and identify important gait parameters")
        subtitle_font = QFont()
        subtitle_font.setPointSize(13)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666;")
        header_layout.addWidget(subtitle)
        
        main_layout.addLayout(header_layout)
        main_layout.addSpacing(10)

        # Step 1: Group 1
        self.step1_card = StepCard(1, "Select Group 1", "Choose the folder containing kinematic data for the first group")
        
        btn_layout1 = QHBoxLayout()
        self.select_group1_button = QPushButton("Select Group 1 Folder")
        self.select_group1_button.setMinimumHeight(40)
        self.select_group1_button.setStyleSheet(self._get_button_style())
        self.select_group1_button.clicked.connect(lambda: self.SelectGroupFolder(0))
        btn_layout1.addWidget(self.select_group1_button)
        btn_layout1.addStretch()
        self.step1_card.content_layout.addLayout(btn_layout1)
        
        self.select_group1_text = QLabel("No folder selected")
        self.select_group1_text.setStyleSheet("color: #999; font-style: italic;")
        self.step1_card.content_layout.addWidget(self.select_group1_text)
        
        main_layout.addWidget(self.step1_card)

        # Step 2: Group 2
        self.step2_card = StepCard(2, "Select Group 2", "Choose the folder containing kinematic data for the second group")
        
        btn_layout2 = QHBoxLayout()
        self.select_group2_button = QPushButton("Select Group 2 Folder")
        self.select_group2_button.setMinimumHeight(40)
        self.select_group2_button.setStyleSheet(self._get_button_style())
        self.select_group2_button.clicked.connect(lambda: self.SelectGroupFolder(1))
        btn_layout2.addWidget(self.select_group2_button)
        btn_layout2.addStretch()
        self.step2_card.content_layout.addLayout(btn_layout2)
        
        self.select_group2_text = QLabel("No folder selected")
        self.select_group2_text.setStyleSheet("color: #999; font-style: italic;")
        self.step2_card.content_layout.addWidget(self.select_group2_text)
        
        main_layout.addWidget(self.step2_card)

        # Step 3: Output
        self.step3_card = StepCard(3, "Select Output Location", "Choose where to save analysis results")
        
        btn_layout3 = QHBoxLayout()
        self.select_output_folder_button = QPushButton("Select Output Folder")
        self.select_output_folder_button.setMinimumHeight(40)
        self.select_output_folder_button.setStyleSheet(self._get_button_style())
        self.select_output_folder_button.clicked.connect(self.SelectOutputFolder)
        btn_layout3.addWidget(self.select_output_folder_button)
        btn_layout3.addStretch()
        self.step3_card.content_layout.addLayout(btn_layout3)
        
        self.output_path_text = QLabel("No output folder selected")
        self.output_path_text.setStyleSheet("color: #999; font-style: italic;")
        self.step3_card.content_layout.addWidget(self.output_path_text)
        
        main_layout.addWidget(self.step3_card)

        # Run button
        self.random_forest_button = QPushButton("Run Random Forest Classification")
        self.random_forest_button.setMinimumHeight(50)
        self.random_forest_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.random_forest_button.clicked.connect(self.RandomForest)
        self.random_forest_button.setEnabled(False)
        main_layout.addWidget(self.random_forest_button)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 14px; padding: 10px;")
        main_layout.addWidget(self.status_label)

        main_layout.addStretch()
    
    def _get_button_style(self):
        return """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """


    def SelectGroupFolder(self, group_no):
        path = QFileDialog.getExistingDirectory(
            self, f'Select folder with gait kinematic parameters for Group {group_no+1}', self.dirname)
        
        if path:
            from PySide6.QtWidgets import QInputDialog
            # Ask for group name
            group_name, ok = QInputDialog.getText(
                self, f'Group {group_no+1} Name', 
                f'Enter a name for Group {group_no+1}:',
                text=f'Group{group_no+1}'
            )
            
            if ok and group_name:
                if group_no == 0:
                    self.folder1 = path
                    self.group_name_1 = group_name
                    self.select_group1_text.setText(f"{group_name}\n{path}")
                    self.select_group1_text.setStyleSheet("color: #2e7d32; font-weight: bold;")
                elif group_no == 1:
                    self.folder2 = path
                    self.group_name_2 = group_name
                    self.select_group2_text.setText(f"{group_name}\n{path}")
                    self.select_group2_text.setStyleSheet("color: #2e7d32; font-weight: bold;")
                
                self.check_ready_to_run()


    def SelectOutputFolder(self):
        path = QFileDialog.getExistingDirectory(
            self, 'Select folder to save analysis results', self.dirname)
        if path:
            self.output_path = path
            self.output_path_text.setText(f"{path}")
            self.output_path_text.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self.check_ready_to_run()
    
    
    def check_ready_to_run(self):
        """Enable run button when all inputs are ready"""
        if self.folder1 and self.folder2 and self.output_path:
            self.random_forest_button.setEnabled(True)
            self.status_label.setText("Ready to run classification")
            self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")


    def RandomForest(self):
        try:
            self.status_label.setText("Running Random Forest classification...")
            self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
            self.random_forest_button.setEnabled(False)
            
            group_1_file_list = DataAnalysisFunctions.find_files(self.folder1)
            group_2_file_list = DataAnalysisFunctions.find_files(self.folder2)
            file_lists = [group_1_file_list, group_2_file_list]
            group_names = [self.group_name_1, self.group_name_2]
            combined_df = DataAnalysisFunctions.combine_files(file_lists, group_names, self.output_path, 'strides')

            DataAnalysisFunctions.random_forest(combined_df, self.output_path)

            self.status_label.setText(f"Analysis completed! Results saved to:\n{self.output_path}")
            self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold; font-size: 16px;")
            
            QMessageBox.information(self, "Success", 
                f"Random Forest classification completed!\n\nResults saved to:\n{self.output_path}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self.random_forest_button.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{str(e)}")