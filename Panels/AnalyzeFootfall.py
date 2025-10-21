from PySide6.QtWidgets import (QWidget, QGridLayout, QLabel, QPushButton, QFileDialog, 
                               QMessageBox, QRadioButton, QButtonGroup, QComboBox, 
                               QLineEdit, QCheckBox, QProgressBar, QTextEdit, QGroupBox,
                               QVBoxLayout, QHBoxLayout, QSpinBox, QDoubleSpinBox, QSlider,
                               QDialog, QDialogButtonBox, QListWidget, QStackedWidget,
                               QListWidgetItem, QFrame, QScrollArea)
from PySide6.QtGui import QFont, QShortcut, QKeySequence
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from Functions import FootfallFunctions, ConfigFunctions
from Functions.InteractiveTimelineWidget import InteractiveTimelineWidget
import os
import numpy as np
import pandas as pd

class AnalyzeFootfallPanel(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width, self.window_height, self.frame_rate = configs['window_width'], configs['window_height'], configs['frame_rate']
        
        # Wizard state
        self.wizard_step = 0  # 0-4 for 5 steps
        self.detection_complete = False
        self.video_imported = False
        self.validation_complete = False
        
        # Data state
        self.dirname = os.getcwd()
        self.csv_dirname = None
        self.video_dirname = None
        self.filename = None
        self.video = None
        self.video_name = None
        self.df = None
        self.bodyparts = []
        self.selected_bodyparts = []
        self.method_selection = 'Deviation'
        
        # Detection settings from config
        self.likelihood_threshold = configs['likelihood_threshold']
        self.depth_threshold = configs['depth_threshold']
        self.threshold = configs['threshold']
        
        # Detection results
        self.n_pred, self.depth_pred, self.t_pred, self.start_pred, self.end_pred, self.bodypart_list_pred = 0, [], [], [], [], []
        self.confirmed = []
        self.slip_fall_pred = []
        
        # Validation state
        self.n_val, self.depth_val, self.t_val, self.start_val, self.end_val, self.bodypart_list_val, self.slip_fall_val = 0, [], [], [], [], [], []
        self.check_slip_fall = True
        
        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("Ladder Rung (Footfall) Analysis Wizard")
        font = QFont()
        font.setPointSize(20)
        self.header.setFont(font)
        self.layout.addWidget(self.header)

        # Wizard stack
        self.wizard_stack = QStackedWidget()
        self.layout.addWidget(self.wizard_stack)

        # Setup wizard steps
        self.setup_step1_import()
        self.setup_step2_settings()
        self.setup_step3_video()
        self.setup_step4_validation()
        self.setup_step5_export()

        # Navigation buttons
        self.nav_layout = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.back_button.clicked.connect(self.go_back)
        self.next_button.clicked.connect(self.go_next)
        self.back_button.setEnabled(False)
        self.next_button.setEnabled(False)
        
        self.nav_layout.addWidget(self.back_button)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.next_button)
        self.layout.addLayout(self.nav_layout)

        self.wizard_stack.setCurrentIndex(0)


    def setup_step1_import(self):
        """Step 1: CSV Import, Method Selection, and Bodypart Selection"""
        step1 = QWidget()
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        step1.setLayout(wrapper_layout)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        wrapper_layout.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout()
        content_widget.setLayout(layout)

        # CSV Import Section
        csv_group = QGroupBox("CSV File Import")
        csv_layout = QVBoxLayout()
        
        csv_btn_layout = QHBoxLayout()
        self.import_csv_button = QPushButton("Import CSV File")
        self.import_csv_button.clicked.connect(self.import_csv)
        csv_btn_layout.addWidget(self.import_csv_button)
        csv_btn_layout.addStretch()
        csv_layout.addLayout(csv_btn_layout)
        
        self.csv_status = QLabel("No file imported yet.\nImport a CSV file from DeepLabCut or similar format.")
        self.csv_status.setWordWrap(True)
        csv_layout.addWidget(self.csv_status)
        
        csv_group.setLayout(csv_layout)
        layout.addWidget(csv_group)

        # Detection Method Section
        method_group = QGroupBox("Detection Method")
        method_layout = QVBoxLayout()
        
        self.method_button_group = QButtonGroup()
        self.method_deviation = QRadioButton("Deviation")
        self.method_threshold = QRadioButton("Threshold")
        self.method_baseline = QRadioButton("Baseline")
        
        self.method_button_group.addButton(self.method_deviation, 0)
        self.method_button_group.addButton(self.method_threshold, 1)
        self.method_button_group.addButton(self.method_baseline, 2)
        self.method_deviation.setChecked(True)
        
        method_layout.addWidget(self.method_deviation)
        method_layout.addWidget(QLabel("  Detects footfalls based on deviation from mean position"))
        method_layout.addWidget(self.method_threshold)
        method_layout.addWidget(QLabel("  Detects footfalls using a fixed or auto-calculated threshold"))
        method_layout.addWidget(self.method_baseline)
        method_layout.addWidget(QLabel("  Detects footfalls relative to a baseline position"))
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Bodypart Selection Section
        bodypart_group = QGroupBox("Bodypart Selection")
        bodypart_layout = QVBoxLayout()
        
        bodypart_layout.addWidget(QLabel("Select bodyparts to detect footfalls for:"))
        self.bodypart_list = QListWidget()
        self.bodypart_list.itemChanged.connect(self.on_bodypart_selection_changed)
        bodypart_layout.addWidget(self.bodypart_list)
        
        bodypart_group.setLayout(bodypart_layout)
        layout.addWidget(bodypart_group)

        layout.addStretch()
        self.wizard_stack.addWidget(step1)


    def setup_step2_settings(self):
        """Step 2: Detection Settings and Run Detection"""
        step2 = QWidget()
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        step2.setLayout(wrapper_layout)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        wrapper_layout.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout()
        content_widget.setLayout(layout)

        # Detection Parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QGridLayout()
        
        # Likelihood Threshold
        params_layout.addWidget(QLabel("Likelihood Threshold:"), 0, 0)
        self.likelihood_spin = QDoubleSpinBox()
        self.likelihood_spin.setRange(0.0, 1.0)
        self.likelihood_spin.setSingleStep(0.1)
        self.likelihood_spin.setValue(self.likelihood_threshold)
        params_layout.addWidget(self.likelihood_spin, 0, 1)
        help_text = QLabel("Minimum confidence for bodypart tracking. Higher values = stricter detection.")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: gray; font-size: 10pt;")
        params_layout.addWidget(help_text, 1, 0, 1, 2)
        
        # Depth Threshold
        params_layout.addWidget(QLabel("Depth Threshold:"), 2, 0)
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.0, 1.0)
        self.depth_spin.setSingleStep(0.1)
        self.depth_spin.setValue(self.depth_threshold)
        params_layout.addWidget(self.depth_spin, 2, 1)
        help_text2 = QLabel("Percentage of recovery relative to depth of previous footfall/slip. Used for footfall detection.")
        help_text2.setWordWrap(True)
        help_text2.setStyleSheet("color: gray; font-size: 10pt;")
        params_layout.addWidget(help_text2, 3, 0, 1, 2)
        
        # Threshold Value (only for Threshold method)
        params_layout.addWidget(QLabel("Threshold Value (optional):"), 4, 0)
        self.threshold_input = QLineEdit()
        self.threshold_input.setText(str(self.threshold) if self.threshold != '' else '')
        self.threshold_input.setPlaceholderText("Auto-calculated if empty")
        params_layout.addWidget(self.threshold_input, 4, 1)
        help_text3 = QLabel("Specific threshold value for threshold-based detection. Leave empty for automatic calculation.")
        help_text3.setWordWrap(True)
        help_text3.setStyleSheet("color: gray; font-size: 10pt;")
        params_layout.addWidget(help_text3, 5, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Run Detection
        detect_layout = QVBoxLayout()
        self.run_detect_button = QPushButton("Run Detection")
        self.run_detect_button.clicked.connect(self.run_detection)
        self.run_detect_button.setStyleSheet("QPushButton { font-size: 14pt; padding: 10px; }")
        detect_layout.addWidget(self.run_detect_button)
        
        self.detection_results = QLabel("Detection not run yet.")
        self.detection_results.setWordWrap(True)
        detect_layout.addWidget(self.detection_results)
        layout.addLayout(detect_layout)

        # Action Buttons
        action_layout = QHBoxLayout()
        self.save_pred_button = QPushButton("Save Detected Footfalls")
        self.save_pred_button.clicked.connect(self.save_prediction)
        self.save_pred_button.setEnabled(False)
        action_layout.addWidget(self.save_pred_button)
        
        self.proceed_validation_button = QPushButton("Proceed to Validation")
        self.proceed_validation_button.clicked.connect(self.go_to_video_import)
        self.proceed_validation_button.setEnabled(False)
        action_layout.addWidget(self.proceed_validation_button)
        layout.addLayout(action_layout)

        layout.addStretch()
        self.wizard_stack.addWidget(step2)


    def setup_step3_video(self):
        """Step 3: Video Import"""
        step3 = QWidget()
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        step3.setLayout(wrapper_layout)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        wrapper_layout.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout()
        content_widget.setLayout(layout)

        # Video Import Section
        video_group = QGroupBox("Video Import")
        video_layout = QVBoxLayout()
        
        video_btn_layout = QHBoxLayout()
        self.import_video_button = QPushButton("Import Video File")
        self.import_video_button.clicked.connect(self.import_video)
        video_btn_layout.addWidget(self.import_video_button)
        video_btn_layout.addStretch()
        video_layout.addLayout(video_btn_layout)
        
        self.video_status = QLabel("No video imported yet.\nImport the corresponding video for visual validation of detected footfalls.")
        self.video_status.setWordWrap(True)
        video_layout.addWidget(self.video_status)
        
        help_label = QLabel("Note: Ensure the video corresponds to the CSV file for accurate validation.\n"
                           "You'll be able to mark start/end of footfalls and distinguish slips from falls.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 10pt;")
        video_layout.addWidget(help_label)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        layout.addStretch()
        self.wizard_stack.addWidget(step3)


    def setup_step4_validation(self):
        """Step 4: Visual Validation Interface"""
        from PySide6.QtWidgets import QScrollArea
        
        step4 = QWidget()
        
        # Create scroll area for validation UI
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        scroll_widget = QWidget()
        self.validation_layout = QGridLayout()
        scroll_widget.setLayout(self.validation_layout)
        scroll_area.setWidget(scroll_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        step4.setLayout(main_layout)
        
        # This will be populated when entering validation mode
        self.validation_widget = step4
        
        self.wizard_stack.addWidget(step4)


    def setup_step5_export(self):
        """Step 5: Save & Complete"""
        step5 = QWidget()
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        step5.setLayout(wrapper_layout)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        wrapper_layout.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout()
        content_widget.setLayout(layout)

        # Validation Summary
        summary_group = QGroupBox("Validation Summary")
        summary_layout = QVBoxLayout()
        self.summary_label = QLabel("Summary will appear here after validation.")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Export Options
        export_group = QGroupBox("Export Results")
        export_layout = QVBoxLayout()
        
        save_btn_layout = QHBoxLayout()
        self.save_results_button = QPushButton("Save Validated Results")
        self.save_results_button.clicked.connect(self.save_validation)
        save_btn_layout.addWidget(self.save_results_button)
        save_btn_layout.addStretch()
        export_layout.addLayout(save_btn_layout)
        
        self.export_status = QLabel("Results not saved yet.")
        export_layout.addWidget(self.export_status)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Next Actions
        actions_layout = QHBoxLayout()
        self.analyze_another_button = QPushButton("Analyze Another File")
        self.analyze_another_button.clicked.connect(self.reset_wizard)
        actions_layout.addWidget(self.analyze_another_button)
        
        self.return_start_button = QPushButton("Return to Start")
        self.return_start_button.clicked.connect(self.return_to_start)
        actions_layout.addWidget(self.return_start_button)
        layout.addLayout(actions_layout)

        layout.addStretch()
        self.wizard_stack.addWidget(step5)


    def import_csv(self):
        """Import CSV file and detect bodyparts"""
        filename, _ = QFileDialog.getOpenFileName(self, 'Choose a file', self.dirname, 
                                                   'CSV files (*.csv);;All files (*.*)')
        if filename:
            self.csv_dirname = os.path.dirname(filename)
            self.filename = filename
            self.df, self.filename = FootfallFunctions.read_file(self.filename)
            self.df, self.bodyparts = FootfallFunctions.fix_column_names(self.df)
            
            if self.df is not None:
                self.csv_status.setText(f"File imported successfully!\n\n{self.filename}\n\nDetected {len(self.bodyparts)} bodyparts.")
                
                # Populate bodypart list with checkboxes
                self.bodypart_list.clear()
                for bodypart in self.bodyparts:
                    item = QListWidgetItem(bodypart)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.bodypart_list.addItem(item)
                
                self.validate_step1()


    def on_bodypart_selection_changed(self):
        """Handle bodypart selection changes"""
        self.validate_step1()


    def validate_step1(self):
        """Validate Step 1 and enable Next button if ready"""
        if self.df is not None:
            # Check if at least one bodypart is selected
            selected_count = 0
            for i in range(self.bodypart_list.count()):
                if self.bodypart_list.item(i).checkState() == Qt.Checked:
                    selected_count += 1
            
            if selected_count > 0:
                self.next_button.setEnabled(True)
                return True
        
        self.next_button.setEnabled(False)
        return False


    def run_detection(self):
        """Run footfall detection"""
        # Update settings from UI
        self.likelihood_threshold = self.likelihood_spin.value()
        self.depth_threshold = self.depth_spin.value()
        threshold_text = self.threshold_input.text().strip()
        self.threshold = threshold_text if threshold_text else ''
        
        # Get selected method
        method_id = self.method_button_group.checkedId()
        if method_id == 0:
            self.method_selection = 'Deviation'
        elif method_id == 1:
            self.method_selection = 'Threshold'
        else:
            self.method_selection = 'Baseline'
        
        # Get selected bodyparts
        self.selected_bodyparts = []
        for i in range(self.bodypart_list.count()):
            if self.bodypart_list.item(i).checkState() == Qt.Checked:
                self.selected_bodyparts.append(self.bodypart_list.item(i).text())
        
        # Run detection
        n_pred, depth_pred, t_pred, start_pred, end_pred, bodypart_list_pred = 0, [], [], [], [], []

        for bodypart in self.selected_bodyparts:
            n_pred_temp, depth_pred_temp, t_pred_temp, start_pred_temp, end_pred_temp = \
                FootfallFunctions.find_footfalls(self.df, bodypart, 'y', panel=self, 
                                                method=self.method_selection, 
                                                likelihood_threshold=self.likelihood_threshold, 
                                                depth_threshold=self.depth_threshold, 
                                                threshold=self.threshold)
            n_pred += n_pred_temp
            depth_pred.extend(depth_pred_temp)
            t_pred.extend(t_pred_temp)
            start_pred.extend(start_pred_temp)
            end_pred.extend(end_pred_temp)
            bodypart_list_pred.extend([bodypart for _ in range(n_pred_temp)])

        # Sort and remove duplicates
        depth_pred = FootfallFunctions.sort_list(t_pred, depth_pred)
        start_pred = FootfallFunctions.sort_list(t_pred, start_pred)
        end_pred = FootfallFunctions.sort_list(t_pred, end_pred)
        bodypart_list_pred = FootfallFunctions.sort_list(t_pred, bodypart_list_pred)
        t_pred = sorted(t_pred)

        to_remove = FootfallFunctions.find_duplicates(t_pred)
        for ind in range(len(to_remove)-1, -1, -1):
            i = to_remove[ind]
            t_pred.pop(i)
            depth_pred.pop(i)
            end_pred.pop(i)
            start_pred.pop(i)
            bodypart_list_pred.pop(i)
            n_pred -= 1

        self.n_pred = n_pred
        self.depth_pred = depth_pred
        self.t_pred = t_pred
        self.start_pred = start_pred
        self.end_pred = end_pred
        self.bodypart_list_pred = bodypart_list_pred
        self.confirmed = [0] * self.n_pred
        self.slip_fall_pred = [''] * self.n_pred

        # Display results
        if n_pred > 0:
            avg_depth = np.mean(self.depth_pred)
            self.detection_results.setText(f"Detected {self.n_pred} footfalls, {avg_depth:.2f} pixels deep on average.")
            self.detection_complete = True
            self.save_pred_button.setEnabled(True)
            self.proceed_validation_button.setEnabled(True)
        else:
            self.detection_results.setText("No footfalls detected. Try different bodyparts or method.")
            self.detection_complete = False
            self.save_pred_button.setEnabled(False)
            self.proceed_validation_button.setEnabled(False)


    def save_prediction(self):
        """Save detected footfalls to CSV"""
        save_pred_dialog = QFileDialog.getSaveFileName(self, 'Save detected footfalls as...', 
                                                       self.dirname, 'CSV files (*.csv);;All files (*.*)')
        if save_pred_dialog[0]:
            pathname = save_pred_dialog[0]
            try:
                FootfallFunctions.make_output(pathname, self.df, self.t_pred, self.depth_pred, 
                                             self.start_pred, self.end_pred, self.bodypart_list_pred, 
                                             self.slip_fall_pred, self.frame_rate)
                QMessageBox.information(self, "Success", f"Predictions saved to:\n{pathname}")
            except IOError:
                QMessageBox.critical(self, "Error", f"Cannot save data to file {pathname}.\nTry another location or filename.")


    def go_to_video_import(self):
        """Navigate to video import step"""
        self.wizard_step = 2
        self.wizard_stack.setCurrentIndex(2)
        self.back_button.setEnabled(True)
        self.next_button.setEnabled(False)


    def import_video(self):
        """Import video file"""
        import_dialog = QFileDialog.getOpenFileName(self, 'Choose a video file', self.dirname, 
                                                    'Video files (*.avi *.mp4 *.webm *.mov);;All files (*.*)')
        if import_dialog[0]:
            self.video = import_dialog[0]
            self.video_dirname = os.path.dirname(self.video)
            if '/' in self.video:
                self.video_name = self.video.split('/')[-1]
            elif '\\' in self.video:
                self.video_name = self.video.split('\\')[-1]
            
            self.video_imported = True
            self.video_status.setText(f"Video imported successfully!\n\n{self.video_name}\n\nReady for validation.")
            self.next_button.setEnabled(True)


    def initialize_validation_ui(self):
        """Initialize the validation UI when entering Step 4"""
        # Clear existing widgets
        while self.validation_layout.count():
            item = self.validation_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Initialize validation state
        self.n_val = self.n_pred
        self.depth_val = self.depth_pred[:]
        self.t_val = self.t_pred[:]
        self.start_val = self.start_pred[:]
        self.end_val = self.end_pred[:]
        self.bodypart_list_val = self.bodypart_list_pred[:]
        self.slip_fall_val = self.slip_fall_pred[:]
        
        if self.n_pred > 0:
            self.n_frame = self.t_pred[0]
            self.t_pred_id = 0
            self.t_val_id = 0
        else:
            self.n_frame = 0
            self.t_pred_id = 0
            self.t_val_id = 0
        
        self.t_pred_max = len(self.t_pred) - 1 if len(self.t_pred) > 0 else 0
        self.t_val_max = len(self.t_val) - 1 if len(self.t_val) > 0 else 0
        
        if self.n_frame in self.t_pred:
            self.bodypart = self.bodypart_list_pred[self.t_pred.index(self.n_frame)]
        else:
            self.bodypart = self.selected_bodyparts[0] if self.selected_bodyparts else ''
        
        self.zoom = False
        self.zoom_image = False
        
        # File info button
        self.file_info_button = QPushButton("Show File Information")
        self.file_info_button.clicked.connect(self.display_info)
        self.validation_layout.addWidget(self.file_info_button, 0, 0)
        
        # Zoom frame button
        self.zoom_frame_button = QPushButton("Zoom In")
        self.zoom_frame_button.clicked.connect(self.zoom_frame)
        self.validation_layout.addWidget(self.zoom_frame_button, 0, 1)
        
        # Frame display
        frame = FootfallFunctions.plot_frame(self.video, self.n_frame, 8, 4, 
                                            int(self.frame_rate), self.df, self.bodypart, self.zoom_image)
        self.frame_canvas = FigureCanvas(frame)
        self.validation_layout.addWidget(self.frame_canvas, 1, 0, 6, 2)
        
        # Validate and Reject buttons
        self.validate_button = QPushButton("Confirm (Space)")
        self.validate_button.clicked.connect(self.on_validate)
        self.validation_layout.addWidget(self.validate_button, 1, 2)
        
        self.reject_button = QPushButton("Reject")
        self.reject_button.clicked.connect(self.on_reject)
        self.validation_layout.addWidget(self.reject_button, 1, 3)
        
        # Navigation buttons - footfall
        self.prev_pred_button = QPushButton("← Previous Detected Footfall (A)")
        self.prev_pred_button.clicked.connect(lambda: self.switch_frame(None, 'prev_pred'))
        self.validation_layout.addWidget(self.prev_pred_button, 2, 2, 1, 2)
        
        self.frame_label = QLabel("Frame")
        self.validation_layout.addWidget(self.frame_label, 3, 2, 1, 2, Qt.AlignCenter)
        
        self.next_pred_button = QPushButton("Next Detected Footfall (D) →")
        self.next_pred_button.clicked.connect(lambda: self.switch_frame(None, 'next_pred'))
        self.validation_layout.addWidget(self.next_pred_button, 4, 2, 1, 2)
        
        self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val)
        
        # Frame navigation
        nav_frame_layout = QHBoxLayout()
        self.prev10_button = QPushButton("<<")
        self.prev10_button.clicked.connect(lambda: self.switch_frame(None, -10))
        nav_frame_layout.addWidget(self.prev10_button)
        
        self.prev_button = QPushButton("<")
        self.prev_button.clicked.connect(lambda: self.switch_frame(None, -1))
        nav_frame_layout.addWidget(self.prev_button)
        
        self.slider_label = QLabel(str(self.n_frame + 1))
        nav_frame_layout.addWidget(self.slider_label)
        
        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(lambda: self.switch_frame(None, 1))
        nav_frame_layout.addWidget(self.next_button)
        
        self.next10_button = QPushButton(">>")
        self.next10_button.clicked.connect(lambda: self.switch_frame(None, 10))
        nav_frame_layout.addWidget(self.next10_button)
        
        self.validation_layout.addLayout(nav_frame_layout, 5, 2, 1, 2)
        
        # Checkboxes
        checkbox_layout = QHBoxLayout()
        self.start_check_box = QCheckBox("Start of footfall")
        self.start_check_box.stateChanged.connect(lambda: self.mark_frame(None, 'start'))
        checkbox_layout.addWidget(self.start_check_box)
        
        self.val_check_box = QCheckBox("Confirmed")
        self.val_check_box.stateChanged.connect(lambda: self.mark_frame(None, 'confirmed'))
        if self.n_pred > 0:
            self.val_check_box.setChecked(self.confirmed[0])
        checkbox_layout.addWidget(self.val_check_box)
        
        self.end_check_box = QCheckBox("End of footfall")
        self.end_check_box.stateChanged.connect(lambda: self.mark_frame(None, 'end'))
        checkbox_layout.addWidget(self.end_check_box)
        
        self.validation_layout.addLayout(checkbox_layout, 6, 2, 1, 2)
        
        # Marking controls
        marking_group = QGroupBox("Mark Slip/Fall")
        marking_layout = QVBoxLayout()
        
        mark_btn_layout = QHBoxLayout()
        self.mark_slip_button = QPushButton("Mark as Slip")
        self.mark_slip_button.clicked.connect(lambda: self.mark_slip_fall('slip'))
        mark_btn_layout.addWidget(self.mark_slip_button)
        
        self.mark_fall_button = QPushButton("Mark as Fall")
        self.mark_fall_button.clicked.connect(lambda: self.mark_slip_fall('fall'))
        mark_btn_layout.addWidget(self.mark_fall_button)
        marking_layout.addLayout(mark_btn_layout)
        
        self.slip_fall_label = QLabel("Current: None")
        marking_layout.addWidget(self.slip_fall_label)
        
        marking_group.setLayout(marking_layout)
        self.validation_layout.addWidget(marking_group, 7, 2, 1, 2)
        
        # Progress indicator
        self.progress_label = QLabel(f"Validated 0 / {self.n_pred} footfalls")
        self.validation_layout.addWidget(self.progress_label, 8, 2, 1, 2)
        
        # Interactive PyQtGraph timeline (replaces matplotlib graph)
        self.interactive_timeline = InteractiveTimelineWidget()
        self.interactive_timeline.frame_clicked.connect(self.on_timeline_frame_clicked)
        
        # Set data
        self.interactive_timeline.set_data(self.df, self.bodyparts, self.selected_bodyparts, 'y')
        self.interactive_timeline.set_current_bodypart(self.bodypart)
        self.interactive_timeline.set_detection_data(self.t_val, self.start_val, self.end_val, 
                                                     self.bodypart_list_val, self.confirmed, self.slip_fall_val)
        self.interactive_timeline.set_current_frame(self.n_frame, self.likelihood_threshold)
        self.interactive_timeline.update_plot()
        
        self.validation_layout.addWidget(self.interactive_timeline, 9, 0, 1, 4)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(len(self.df))
        self.slider.setValue(self.n_frame + 1)
        self.slider.valueChanged.connect(self.on_slider_scroll)
        self.validation_layout.addWidget(self.slider, 10, 0, 1, 4)
        
        # Likelihood threshold controls
        likelihood_layout = QHBoxLayout()
        likelihood_layout.addWidget(QLabel("Likelihood threshold:"))
        self.likelihood_input = QLineEdit(str(self.likelihood_threshold))
        likelihood_layout.addWidget(self.likelihood_input)
        self.likelihood_button = QPushButton("Update")
        self.likelihood_button.clicked.connect(self.on_likelihood_update)
        likelihood_layout.addWidget(self.likelihood_button)
        likelihood_layout.addStretch()
        self.validation_layout.addLayout(likelihood_layout, 11, 0, 1, 4)
        
        # Bodypart selection and zoom
        controls_layout = QHBoxLayout()
        self.to_start_button = QPushButton("< Start of footfall")
        self.to_start_button.clicked.connect(lambda: self.switch_frame(None, 'start'))
        controls_layout.addWidget(self.to_start_button)
        
        self.bodypart_to_plot = QComboBox()
        self.bodypart_to_plot.addItems(self.selected_bodyparts)
        self.bodypart_to_plot.currentTextChanged.connect(self.on_bodypart_plot)
        controls_layout.addWidget(self.bodypart_to_plot)
        
        self.zoom_button = QPushButton("Zoom In Plot")
        self.zoom_button.clicked.connect(self.zoom_plot)
        controls_layout.addWidget(self.zoom_button)
        
        self.to_end_button = QPushButton("End of footfall >")
        self.to_end_button.clicked.connect(lambda: self.switch_frame(None, 'end'))
        controls_layout.addWidget(self.to_end_button)
        
        self.validation_layout.addLayout(controls_layout, 12, 0, 1, 4)
        
        # Save and restart buttons
        final_layout = QHBoxLayout()
        self.save_val_button = QPushButton("Save (Ctrl-S)")
        self.save_val_button.clicked.connect(self.save_validation_temp)
        final_layout.addWidget(self.save_val_button)
        
        self.restart_button = QPushButton("Finish Validation")
        self.restart_button.clicked.connect(self.finish_validation)
        final_layout.addWidget(self.restart_button)
        final_layout.addStretch()
        
        self.validation_layout.addLayout(final_layout, 13, 0, 1, 4)
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
        # Initial display
        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)


    def setup_shortcuts(self):
        """Setup keyboard shortcuts for validation"""
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self.validation_widget)
        self.shortcut_space.activated.connect(self.on_validate)
        
        self.shortcut_a = QShortcut(QKeySequence(Qt.Key_A), self.validation_widget)
        self.shortcut_a.activated.connect(lambda: self.switch_frame(None, 'prev_pred'))
        
        self.shortcut_d = QShortcut(QKeySequence(Qt.Key_D), self.validation_widget)
        self.shortcut_d.activated.connect(lambda: self.switch_frame(None, 'next_pred'))
        
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self.validation_widget)
        self.shortcut_save.activated.connect(self.save_validation_temp)


    def mark_frame(self, e, mark_type):
        """Mark frame as start, confirmed, or end"""
        if mark_type == "confirmed":
            is_checked = self.val_check_box.isChecked()
            if is_checked:
                if self.n_frame not in self.t_val:
                    self.n_val += 1
                    self.t_val.append(self.n_frame)
                    self.depth_val.append(np.nan)
                    self.start_val.append(np.nan)
                    self.end_val.append(np.nan)
                    self.bodypart_list_val.append(self.bodypart_to_plot.currentText() if hasattr(self, 'bodypart_to_plot') else self.bodypart)
                    self.confirmed.append(1)

                    self.depth_val = FootfallFunctions.sort_list(self.t_val, self.depth_val)
                    self.start_val = FootfallFunctions.sort_list(self.t_val, self.start_val)
                    self.end_val = FootfallFunctions.sort_list(self.t_val, self.end_val)
                    self.bodypart_list_val = FootfallFunctions.sort_list(self.t_val, self.bodypart_list_val)
                    self.confirmed = FootfallFunctions.sort_list(self.t_val, self.confirmed)
                    self.t_val = sorted(self.t_val)
                else:
                    index = self.t_val.index(self.n_frame)
                    self.confirmed[index] = 1
                
                FootfallFunctions.ControlButton(self)
                FootfallFunctions.DisplayPlots(self)
            else:
                if self.n_frame in self.t_val:
                    self.n_val -= 1
                    index = self.t_val.index(self.n_frame)
                    self.confirmed[index] = 0

        elif mark_type == "start":
            _, self.next_val_confirmed = FootfallFunctions.find_confirmed_neighbors(self.n_frame, self.t_val, self.confirmed, end=self.t_val_id)
            
            if self.n_frame in self.t_val:
                index = self.t_val.index(self.n_frame)
            elif self.next_val_confirmed != 0:
                index = self.t_val.index(self.next_val_confirmed)
            else:
                index = self.t_val.index(self.next_val) if self.next_val != 0 and self.next_val in self.t_val else 0
            
            is_checked = self.start_check_box.isChecked()
            if is_checked:
                if self.n_frame not in self.start_val:
                    self.start_val.pop(index)
                    self.start_val.insert(index, self.n_frame)
            else:
                if self.n_frame in self.start_val:
                    self.start_val.pop(index)
                    self.start_val.insert(index, np.nan)
            
        elif mark_type == "end":
            self.prev_val_confirmed, _ = FootfallFunctions.find_confirmed_neighbors(self.n_frame, self.t_val, self.confirmed, start=self.t_val_id)
            
            if self.n_frame in self.t_val:
                index = self.t_val.index(self.n_frame)
            elif self.prev_val_confirmed != 0:
                index = self.t_val.index(self.prev_val_confirmed)
            else:
                index = self.t_val.index(self.prev_val) if self.prev_val != 0 and self.prev_val in self.t_val else 0
            
            is_checked = self.end_check_box.isChecked()
            if is_checked:
                if self.n_frame not in self.end_val:
                    self.end_val.pop(index)
                    self.end_val.insert(index, self.n_frame)
            else:
                if self.n_frame in self.end_val:
                    self.end_val.pop(index)
                    self.end_val.insert(index, np.nan)
            
            FootfallFunctions.ControlButton(self)

        FootfallFunctions.DisplayPlots(self)
        self.update_progress()


    def on_validate(self):
        """Confirm current footfall"""
        self.val_check_box.setChecked(True)
        
        if self.check_slip_fall:
            slip_fall = self.slip_fall_popup()
        else:
            slip_fall = 'NaN'

        if self.n_frame not in self.t_val:
            self.n_val += 1
            self.t_val.append(self.n_frame)
            self.depth_val.append(np.nan)
            self.start_val.append(np.nan)
            self.end_val.append(np.nan)
            self.bodypart_list_val.append(self.bodypart)
            self.confirmed.append(1)
            self.slip_fall_val.append(slip_fall)

            self.depth_val = FootfallFunctions.sort_list(self.t_val, self.depth_val)
            self.start_val = FootfallFunctions.sort_list(self.t_val, self.start_val)
            self.end_val = FootfallFunctions.sort_list(self.t_val, self.end_val)
            self.bodypart_list_val = FootfallFunctions.sort_list(self.t_val, self.bodypart_list_val)
            self.confirmed = FootfallFunctions.sort_list(self.t_val, self.confirmed)
            self.slip_fall_val = FootfallFunctions.sort_list(self.t_val, self.slip_fall_val)
            self.t_val = sorted(self.t_val)

            self.t_val_id = self.t_val.index(self.n_frame)
            self.t_val_max += 1

            if self.t_val_id + 1 >= self.t_val_max:
                self.prev_val = self.t_val[self.t_val_max - 1]
                self.next_val = 0
                self.t_val_id += 1
            else:
                self.prev_val = self.t_val[self.t_val_id]
                self.next_val = self.t_val[self.t_val_id + 2]
                self.t_val_id += 1
        else:
            index = self.t_val.index(self.n_frame)
            self.confirmed[index] = 1
            self.slip_fall_val[index] = slip_fall

            if self.next_pred != 0:
                self.n_frame = self.next_pred
                self.slider.setValue(self.n_frame)
                self.slider_label.setText(str(self.n_frame + 1))

            if self.t_pred_id + 1 >= self.t_pred_max:
                self.prev_pred = self.t_pred[self.t_pred_max - 1] if self.t_pred_max >= 0 else 0
                self.next_pred = 0
                self.t_pred_id += 1
            else:
                self.prev_pred = self.t_pred[self.t_pred_id]
                self.next_pred = self.t_pred[self.t_pred_id + 2] if self.t_pred_id + 2 < len(self.t_pred) else 0
                self.t_pred_id += 1
            
            if self.t_val_id + 1 >= self.t_val_max:
                self.prev_val = self.t_val[self.t_val_max - 1] if self.t_val_max >= 0 else 0
                self.next_val = 0
                self.t_val_id += 1
            else:
                self.prev_val = self.t_val[self.t_val_id]
                self.next_val = self.t_val[self.t_val_id + 2] if self.t_val_id + 2 < len(self.t_val) else 0
                self.t_val_id += 1

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)
        self.update_progress()


    def on_reject(self):
        """Reject current footfall"""
        self.val_check_box.setChecked(False)

        if self.n_frame in self.t_val:
            self.n_val -= 1
            index = self.t_val.index(self.n_frame)
            self.confirmed[index] = 0

        if self.next_pred != 0:
            self.n_frame = self.next_pred
            self.slider.setValue(self.n_frame)
            self.slider_label.setText(str(self.n_frame + 1))

        if self.t_pred_id + 1 >= self.t_pred_max:
            self.prev_pred = self.t_pred[-1] if self.t_pred else 0
            self.next_pred = 0
            self.t_pred_id += 1
        else:
            self.prev_pred = self.t_pred[self.t_pred_id]
            self.next_pred = self.t_pred[self.t_pred_id + 2] if self.t_pred_id + 2 < len(self.t_pred) else 0
            self.t_pred_id += 1
        
        if self.t_val_id + 1 >= self.t_val_max:
            self.prev_val = self.t_val[-1] if self.t_val else 0
            self.next_val = 0
            self.t_val_id += 1
        else:
            self.prev_val = self.t_val[self.t_val_id]
            self.next_val = self.t_val[self.t_val_id + 2] if self.t_val_id + 2 < len(self.t_val) else 0
            self.t_val_id += 1

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)
        self.update_progress()


    def on_slider_scroll(self):
        """Handle slider scrolling"""
        self.n_frame = self.slider.value() - 1

        if self.n_frame > self.prev_pred and self.n_frame < self.t_pred[self.t_pred_id]:
            self.next_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame < self.next_pred and self.n_frame > self.t_pred[self.t_pred_id]:
            self.prev_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame <= self.prev_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, end=self.t_pred_id)
            if self.prev_pred == 0:
                self.t_pred_id = 0
            elif self.next_pred == 0:
                self.t_pred_id = self.t_pred_max
            else:
                index = self.t_pred.index(self.prev_pred)
                self.t_pred_id = index + 1
        elif self.n_frame >= self.next_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, start=self.t_pred_id)
            if self.prev_pred == 0:
                self.t_pred_id = 0
            elif self.next_pred == 0:
                self.t_pred_id = self.t_pred_max
            else:
                index = self.t_pred.index(self.prev_pred)
                self.t_pred_id = index + 1
                
        if self.n_frame > self.prev_val and self.n_frame < self.t_val[self.t_val_id]:
            self.next_val = self.t_val[self.t_val_id]
        elif self.n_frame < self.next_val and self.n_frame > self.t_val[self.t_val_id]:
            self.prev_val = self.t_val[self.t_val_id]
        elif self.n_frame <= self.prev_val:
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, end=self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1
        elif self.n_frame >= self.next_val:
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, start=self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1

        self.slider_label.setText(str(self.n_frame + 1))
        
        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)


    def switch_frame(self, e, new_frame):
        """Switch to a different frame"""
        if type(new_frame) is int:
            self.n_frame = self.n_frame + new_frame
        elif new_frame == 'next_pred':
            self.n_frame = self.next_val if self.next_val != 0 else self.n_frame
        elif new_frame == 'prev_pred':
            self.n_frame = self.prev_val if self.prev_val != 0 else self.n_frame
        elif self.n_frame in self.t_val:
            index = self.t_val.index(self.n_frame)
            if self.start_val[index] is not np.nan and new_frame == 'start': 
                self.n_frame = self.start_val[index]
                self.next_val = self.t_val[index]
            elif self.end_val[index] is not np.nan and new_frame == 'end':
                self.n_frame = self.end_val[index]
                self.prev_val = self.t_val[index]
            
        self.slider.setValue(self.n_frame)
        self.slider_label.setText(str(self.n_frame + 1))

        if self.n_frame > self.prev_pred and self.n_frame < self.t_pred[self.t_pred_id]:
            self.next_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame < self.next_pred and self.n_frame > self.t_pred[self.t_pred_id]:
            self.prev_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame <= self.prev_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, end=self.t_pred_id)
            if self.prev_pred == 0:
                self.t_pred_id = 0
            elif self.next_pred == 0:
                self.t_pred_id = self.t_pred_max
            else:
                index = self.t_pred.index(self.prev_pred)
                self.t_pred_id = index + 1
        elif self.n_frame >= self.next_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, start=self.t_pred_id)
            if self.prev_pred == 0:
                self.t_pred_id = 0
            elif self.next_pred == 0:
                self.t_pred_id = self.t_pred_max
            else:
                index = self.t_pred.index(self.prev_pred)
                self.t_pred_id = index + 1
                
        if self.n_frame > self.prev_val and self.n_frame < self.t_val[self.t_val_id]:
            self.next_val = self.t_val[self.t_val_id]
        elif self.n_frame < self.next_val and self.n_frame > self.t_val[self.t_val_id]:
            self.prev_val = self.t_val[self.t_val_id]
        elif self.n_frame <= self.prev_val:
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, end=self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1
        elif self.n_frame >= self.next_val:
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, start=self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)


    def on_bodypart_plot(self):
        """Handle bodypart selection change in validation"""
        self.bodypart = self.bodypart_to_plot.currentText()
        if self.n_frame in self.t_val:
            index = self.t_val.index(self.n_frame)
            self.bodypart_list_val[index] = self.bodypart
        
        # Update interactive timeline if available
        if hasattr(self, 'interactive_timeline'):
            self.interactive_timeline.set_current_bodypart(self.bodypart)

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self, set_bodypart=False)


    def on_likelihood_update(self):
        """Update likelihood threshold"""
        self.likelihood_threshold = float(self.likelihood_input.text())
        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)


    def zoom_plot(self):
        """Toggle plot zoom"""
        if not self.zoom:
            self.zoom = True
            self.zoom_button.setText('Zoom Out Plot')
            if hasattr(self, 'interactive_timeline'):
                self.interactive_timeline.zoom_check.setChecked(True)
        else:
            self.zoom = False
            self.zoom_button.setText('Zoom In Plot')
            if hasattr(self, 'interactive_timeline'):
                self.interactive_timeline.zoom_check.setChecked(False)

        FootfallFunctions.DisplayPlots(self)
    
    def on_timeline_frame_clicked(self, frame):
        """Handle frame click from interactive timeline"""
        self.n_frame = frame
        self.slider.setValue(frame + 1)
        self.slider_label.setText(str(frame + 1))
        
        # Update neighbors
        if self.n_frame > self.prev_pred and self.n_frame < self.t_pred[self.t_pred_id] if self.t_pred_id < len(self.t_pred) else False:
            self.next_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame < self.next_pred and self.n_frame > self.t_pred[self.t_pred_id] if self.t_pred_id < len(self.t_pred) else False:
            self.prev_pred = self.t_pred[self.t_pred_id]
        else:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred)
        
        if self.n_frame > self.prev_val and self.n_frame < self.t_val[self.t_val_id] if self.t_val_id < len(self.t_val) else False:
            self.next_val = self.t_val[self.t_val_id]
        elif self.n_frame < self.next_val and self.n_frame > self.t_val[self.t_val_id] if self.t_val_id < len(self.t_val) else False:
            self.prev_val = self.t_val[self.t_val_id]
        else:
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val)
        
        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)


    def zoom_frame(self):
        """Toggle frame zoom"""
        if not self.zoom_image:
            self.zoom_image = True
            self.zoom_frame_button.setText('Zoom Out')
        else:
            self.zoom_image = False
            self.zoom_frame_button.setText('Zoom In')

        FootfallFunctions.DisplayPlots(self)


    def display_info(self):
        """Display file information"""
        QMessageBox.information(self, "File Information", 
                               f"Currently validating detected footfalls for:\n\n{self.filename}\n\nand\n\n{self.video_name}")


    def mark_slip_fall(self, mark_type):
        """Mark current footfall as slip or fall"""
        if self.n_frame in self.t_val:
            index = self.t_val.index(self.n_frame)
            self.slip_fall_val[index] = mark_type
            self.slip_fall_label.setText(f"Current: {mark_type.capitalize()}")
            FootfallFunctions.DisplayPlots(self)


    def slip_fall_popup(self):
        """Show slip/fall selection dialog"""
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Slip or Fall?")
        dlg.setText("Is it a slip (short, shallow) or fall (long, deep)?")
        
        slip_btn = dlg.addButton("Slip", QMessageBox.YesRole)
        fall_btn = dlg.addButton("Fall", QMessageBox.NoRole)
        skip_btn = dlg.addButton("Skip/Don't Ask Again", QMessageBox.RejectRole)
        
        dlg.exec()
        
        clicked = dlg.clickedButton()
        if clicked == slip_btn:
            return 'slip'
        elif clicked == fall_btn:
            return 'fall'
        elif clicked == skip_btn:
            self.check_slip_fall = False
            return ''
        return ''


    def update_progress(self):
        """Update validation progress"""
        validated = sum(self.confirmed)
        self.progress_label.setText(f"Validated {validated} / {self.n_pred} footfalls")


    def save_validation_temp(self):
        """Save validation results during validation"""
        save_pred_dialog = QFileDialog.getSaveFileName(self, 'Save validated results as...', 
                                                       self.dirname, 'CSV files (*.csv);;All files (*.*)')
        if save_pred_dialog[0]:
            pathname = save_pred_dialog[0]
            try:
                FootfallFunctions.make_output(pathname, self.df, self.t_val, self.depth_val, 
                                             self.start_val, self.end_val, self.bodypart_list_val, 
                                             self.slip_fall_val, self.frame_rate, self.confirmed, True)
                QMessageBox.information(self, "Success", f"Validation results saved to:\n{pathname}")
            except IOError:
                QMessageBox.critical(self, "Error", f"Cannot save data to file {pathname}.\nTry another location or filename.")


    def finish_validation(self):
        """Finish validation and move to export step"""
        self.validation_complete = True
        self.wizard_step = 4
        self.wizard_stack.setCurrentIndex(4)
        self.back_button.setEnabled(False)
        self.next_button.setEnabled(False)
        
        # Update summary
        validated = sum(self.confirmed)
        avg_depth = np.mean([d for d in self.depth_val if not np.isnan(d)]) if self.depth_val else 0
        self.summary_label.setText(f"Total footfalls detected: {self.n_pred}\n"
                                   f"Total confirmed: {validated}\n"
                                   f"Total rejected: {self.n_pred - validated}\n"
                                   f"Average depth: {avg_depth:.2f} pixels")


    def save_validation(self):
        """Save final validation results"""
        save_pred_dialog = QFileDialog.getSaveFileName(self, 'Save validated results as...', 
                                                       self.dirname, 'CSV files (*.csv);;All files (*.*)')
        if save_pred_dialog[0]:
            pathname = save_pred_dialog[0]
            try:
                FootfallFunctions.make_output(pathname, self.df, self.t_val, self.depth_val, 
                                             self.start_val, self.end_val, self.bodypart_list_val, 
                                             self.slip_fall_val, self.frame_rate, self.confirmed, True)
                self.export_status.setText(f"Results saved successfully to:\n{pathname}")
                QMessageBox.information(self, "Success", "Validation results saved successfully!")
            except IOError:
                QMessageBox.critical(self, "Error", f"Cannot save data to file {pathname}.\nTry another location or filename.")


    def reset_wizard(self):
        """Reset wizard to analyze another file"""
        # Reset all state
        self.wizard_step = 0
        self.detection_complete = False
        self.video_imported = False
        self.validation_complete = False
        
        self.filename = None
        self.video = None
        self.video_name = None
        self.df = None
        self.bodyparts = []
        self.selected_bodyparts = []
        
        self.n_pred, self.depth_pred, self.t_pred = 0, [], []
        self.start_pred, self.end_pred, self.bodypart_list_pred = [], [], []
        self.confirmed = []
        self.slip_fall_pred = []
        
        self.n_val, self.depth_val, self.t_val = 0, [], []
        self.start_val, self.end_val, self.bodypart_list_val, self.slip_fall_val = [], [], [], []
        
        # Reset UI
        self.csv_status.setText("No file imported yet.\nImport a CSV file from DeepLabCut or similar format.")
        self.bodypart_list.clear()
        self.detection_results.setText("Detection not run yet.")
        self.save_pred_button.setEnabled(False)
        self.proceed_validation_button.setEnabled(False)
        self.video_status.setText("No video imported yet.\nImport the corresponding video for visual validation of detected footfalls.")
        self.export_status.setText("Results not saved yet.")
        
        # Go to step 1
        self.wizard_stack.setCurrentIndex(0)
        self.back_button.setEnabled(False)
        self.next_button.setEnabled(False)


    def return_to_start(self):
        """Return to main start panel"""
        self.parent().parent().stacked_widget.setCurrentWidget(self.parent().parent().StartPanel)
        self.parent().parent().statusBar().showMessage('Welcome!')


    def go_next(self):
        """Navigate to next step"""
        if self.wizard_step == 0:
            # Step 1 to Step 2
            if self.validate_step1():
                # Collect selected bodyparts
                self.selected_bodyparts = []
                for i in range(self.bodypart_list.count()):
                    if self.bodypart_list.item(i).checkState() == Qt.Checked:
                        self.selected_bodyparts.append(self.bodypart_list.item(i).text())
                
                # Get selected method
                method_id = self.method_button_group.checkedId()
                if method_id == 0:
                    self.method_selection = 'Deviation'
                elif method_id == 1:
                    self.method_selection = 'Threshold'
                else:
                    self.method_selection = 'Baseline'
                
                self.wizard_step = 1
                self.wizard_stack.setCurrentIndex(1)
                self.back_button.setEnabled(True)
                self.next_button.setEnabled(False)
        
        elif self.wizard_step == 2:
            # Step 3 (video) to Step 4 (validation)
            if self.video_imported:
                self.wizard_step = 3
                self.wizard_stack.setCurrentIndex(3)
                self.initialize_validation_ui()
                self.back_button.setEnabled(False)
                self.next_button.setEnabled(False)


    def go_back(self):
        """Navigate to previous step"""
        if self.wizard_step == 1:
            self.wizard_step = 0
            self.wizard_stack.setCurrentIndex(0)
            self.back_button.setEnabled(False)
            self.next_button.setEnabled(True)
        elif self.wizard_step == 2:
            self.wizard_step = 1
            self.wizard_stack.setCurrentIndex(1)
            self.back_button.setEnabled(True)
            self.next_button.setEnabled(False)
