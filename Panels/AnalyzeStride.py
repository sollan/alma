from PySide6.QtWidgets import (QWidget, QGridLayout, QLabel, QPushButton, QFileDialog, 
                               QMessageBox, QRadioButton, QButtonGroup, QComboBox, 
                               QLineEdit, QCheckBox, QProgressBar, QTextEdit, QGroupBox,
                               QVBoxLayout, QHBoxLayout, QSpinBox, QDoubleSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QScrollArea)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, Signal
from Functions import ConfigFunctions, KinematicsFunctions
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading
import matplotlib.pyplot as plt
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=ResourceWarning)


class AnalyzeStridePanel(QWidget):

    def __init__(self, parent):
        """Constructor"""
        super().__init__(parent)

        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width = configs['window_width']
        self.window_height = configs['window_height']
        self.frame_rate = configs['frame_rate']
        self.cutoff_f = configs['lowpass_filter_cutoff']
        self.analysis_type = "Treadmill"
        self.pixels_per_cm = configs['pixels_per_cm']
        if configs['cm_speed'] == '':
            self.cm_speed = None
        else:   
            self.cm_speed = configs['cm_speed']
        self.dragging_filter = bool(configs['dragging_filter'])
        self.drag_clearance_cm = float(configs.get('drag_clearance_cm', 0.3))
        self.drag_min_consecutive_frames = int(configs.get('drag_min_consecutive_frames', 4))
        self.no_outlier_filter = bool(configs['no_outlier_filter'])
        rtl_config = configs['right_to_left']
        if rtl_config == "auto" or rtl_config == "Auto" or rtl_config == "AUTO":
            self.right_to_left = "auto"
        else:
            self.right_to_left = bool(rtl_config)
        self.auto_calibrate_spatial = bool(configs.get('auto_calibrate_spatial', True))
        self.reference_segment = configs.get('reference_segment', 'hip_knee')
        self.reference_length_cm = configs.get('reference_length_cm', '')
        if self.reference_length_cm == '':
            self.reference_length_cm = None
        else:
            self.reference_length_cm = float(self.reference_length_cm)

        # Filtering thresholds
        self.step_height_min_cm = float(configs.get('step_height_min_cm', 0.0))
        self.step_height_max_cm = float(configs.get('step_height_max_cm', 1.5))
        self.stride_length_min_cm = float(configs.get('stride_length_min_cm', 0.0))
        self.stride_length_max_cm = float(configs.get('stride_length_max_cm', 8.0))
    
        self.stride_widgets = []
        self.has_input_path = False
        self.has_imported_file = False
        self.dirname = os.getcwd()
        self.custom_bodypart_mapping = None
        self.bodyparts_raw = []
        self.mapping_widgets = []

        # Wizard state variables
        self.wizard_step = 1
        self.mapping_confirmed = False
        self.analysis_settings = {}
        self.calibration_method = 'reference'  # 'reference' or 'manual'
        self.selected_files = []
        self.is_bulk_analysis = False
        self.output_folder = None

        # Create main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("Kinematic Analysis")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.header.setFont(font)
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)

        # Create wizard stack
        self.wizard_stack = QStackedWidget()
        self.layout.addWidget(self.wizard_stack)

        # Create wizard steps
        self.setup_step1_file_selection()
        self.setup_step2_mapping()
        self.setup_step3_settings()
        self.setup_step4_output()

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

        # Set initial step
        self.wizard_stack.setCurrentIndex(0)

    def setup_step1_file_selection(self):
        """Step 1: File Selection"""
        step1_widget = QWidget()
        step1_wrapper = QVBoxLayout()
        step1_wrapper.setContentsMargins(0, 0, 0, 0)
        step1_widget.setLayout(step1_wrapper)
        
        # Create scroll area for step 1
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        step1_wrapper.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        step1_layout = QVBoxLayout()
        content_widget.setLayout(step1_layout)

        # Title
        title = QLabel("Step 1: Select Files for Analysis")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        step1_layout.addWidget(title)

        # Instructions
        instructions = QLabel("Choose how you want to analyze your data:")
        instructions.setAlignment(Qt.AlignCenter)
        step1_layout.addWidget(instructions)

        # File selection buttons
        button_layout = QHBoxLayout()
        
        # Single file button (recommended)
        self.single_file_button = QPushButton("Analyze Single File")
        self.single_file_button.setMinimumHeight(60)
        self.single_file_button.clicked.connect(self.select_single_file)
        
        # Bulk analysis button
        self.bulk_analysis_button = QPushButton("Bulk Analysis")
        self.bulk_analysis_button.setMinimumHeight(60)
        self.bulk_analysis_button.clicked.connect(self.select_bulk_files)
        
        button_layout.addWidget(self.single_file_button)
        button_layout.addWidget(self.bulk_analysis_button)
        step1_layout.addLayout(button_layout)

        # Recommendation text
        recommendation = QLabel("💡 Recommended: Start with 'Analyze Single File' for first-time users")
        recommendation.setStyleSheet("color: #666; font-style: italic;")
        recommendation.setAlignment(Qt.AlignCenter)
        step1_layout.addWidget(recommendation)

        # File status display
        self.file_status_label = QLabel("No files selected")
        self.file_status_label.setAlignment(Qt.AlignCenter)
        self.file_status_label.setStyleSheet("color: #888; padding: 10px;")
        step1_layout.addWidget(self.file_status_label)

        step1_layout.addStretch()
        self.wizard_stack.addWidget(step1_widget)

    def select_single_file(self):
        """Select a single CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select CSV file', self.dirname, 
            'CSV files (*.csv);;All files (*.*)')
        
        if file_path:
            self.selected_files = [file_path]
            self.is_bulk_analysis = False
            filename = os.path.basename(file_path)
            self.file_status_label.setText(f"Selected: {filename}")
            self.file_status_label.setStyleSheet("color: #2e7d32; padding: 10px;")
            self.validate_step1()

    def select_bulk_files(self):
        """Select folder for bulk analysis"""
        folder_path = QFileDialog.getExistingDirectory(
            self, 'Select folder containing CSV files', self.dirname)
        
        if folder_path:
            # Find all CSV files in the folder
            csv_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(folder_path, file))
            
            if csv_files:
                self.selected_files = csv_files
                self.is_bulk_analysis = True
                self.file_status_label.setText(f"Selected folder: {len(csv_files)} CSV files found")
                self.file_status_label.setStyleSheet("color: #2e7d32; padding: 10px;")
                self.validate_step1()
            else:
                QMessageBox.warning(self, "No CSV Files", 
                    "No CSV files found in the selected folder.")
                self.file_status_label.setText("No CSV files found in selected folder")
                self.file_status_label.setStyleSheet("color: #d32f2f; padding: 10px;")

    def validate_step1(self):
        """Validate Step 1 and enable Next button"""
        if self.selected_files:
            self.next_button.setEnabled(True)
            self.next_button.setText("Next")
        else:
            self.next_button.setEnabled(False)

    def setup_step2_mapping(self):
        """Step 2: Label Mapping Confirmation"""
        step2_widget = QWidget()
        step2_wrapper = QVBoxLayout()
        step2_wrapper.setContentsMargins(0, 0, 0, 0)
        step2_widget.setLayout(step2_wrapper)
        
        # Create scroll area for step 2
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        step2_wrapper.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        step2_layout = QVBoxLayout()
        content_widget.setLayout(step2_layout)

        # Title
        title = QLabel("Step 2: Confirm Bodypart Mapping")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        step2_layout.addWidget(title)

        # Instructions
        instructions = QLabel("Review and confirm the detected bodypart mappings. You can modify them if needed.")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setWordWrap(True)
        step2_layout.addWidget(instructions)

        # Detected bodyparts display
        self.detected_bodyparts_label = QLabel("Detected bodyparts: ")
        self.detected_bodyparts_label.setAlignment(Qt.AlignCenter)
        step2_layout.addWidget(self.detected_bodyparts_label)

        # Mapping table
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(2)
        self.mapping_table.setHorizontalHeaderLabels(["Standard Bodypart", "Detected in CSV"])
        self.mapping_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        step2_layout.addWidget(self.mapping_table)

        # Confirm mapping button
        self.confirm_mapping_button = QPushButton("Confirm Mapping")
        self.confirm_mapping_button.clicked.connect(self.confirm_mapping)
        self.confirm_mapping_button.setMinimumHeight(40)
        step2_layout.addWidget(self.confirm_mapping_button)

        # Status label
        self.mapping_status_label = QLabel("Please confirm the mapping above to continue")
        self.mapping_status_label.setAlignment(Qt.AlignCenter)
        self.mapping_status_label.setStyleSheet("color: #d32f2f; padding: 10px;")
        step2_layout.addWidget(self.mapping_status_label)

        step2_layout.addStretch()
        self.wizard_stack.addWidget(step2_widget)

    def load_mapping_data(self):
        """Load and display mapping data when entering Step 2"""
        if not self.selected_files:
            return

        # Load the first file to detect bodyparts
        try:
            import pandas as pd
            # Use the same method as KinematicsFunctions.read_file
            df = pd.read_csv(self.selected_files[0], header=[1,2])
            self.df = df
            
            # Extract bodypart names from MultiIndex columns
            # Columns are (bodypart, coordinate) where coordinate is 'x', 'y', or 'likelihood'
            self.bodyparts_raw = [col[0] for col in df.columns if col[1] == 'x']
            
            print("CSV columns:", df.columns.tolist())
            print("Detected bodyparts:", self.bodyparts_raw)
            
            # Display detected bodyparts
            self.detected_bodyparts_label.setText(f"Detected bodyparts: {', '.join(self.bodyparts_raw)}")
            
            # Auto-detect mapping
            standard_bodyparts = ['toe', 'mtp', 'ankle', 'knee', 'hip', 'iliac crest']
            auto_mapping, found_bodyparts = KinematicsFunctions.detect_bodypart_mapping(self.bodyparts_raw)
            
            # Setup mapping table
            self.mapping_table.setRowCount(len(standard_bodyparts))
            self.bodypart_combos = {}
            
            for i, standard_bp in enumerate(standard_bodyparts):
                # Standard bodypart label
                standard_item = QTableWidgetItem(standard_bp)
                standard_item.setFlags(standard_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(i, 0, standard_item)
                
                # Dropdown for detected bodypart
                combo = QComboBox()
                combo.addItem("(none)")
                combo.addItems(self.bodyparts_raw)
                
                # Set auto-detected mapping
                for raw_bp in self.bodyparts_raw:
                    if auto_mapping.get(raw_bp) == standard_bp:
                        combo.setCurrentText(raw_bp)
                        break
                
                self.mapping_table.setCellWidget(i, 1, combo)
                self.bodypart_combos[standard_bp] = combo
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load CSV file: {str(e)}")
            self.mapping_status_label.setText("Error loading file")
            self.mapping_status_label.setStyleSheet("color: #d32f2f; padding: 10px;")

    def confirm_mapping(self):
        """Confirm the bodypart mapping"""
        new_mapping = {}
        for standard_bp, combo in self.bodypart_combos.items():
            selected = combo.currentText()
            if selected != "(none)":
                new_mapping[selected] = standard_bp
        
        if not new_mapping:
            QMessageBox.warning(self, "Warning", "Please select at least one bodypart mapping!")
            return
        
        self.custom_bodypart_mapping = new_mapping
        
        # Apply mapping to dataframe
        try:
            self.df, self.bodyparts, self.bodyparts_raw = KinematicsFunctions.fix_column_names(
                self.df, self.custom_bodypart_mapping)
            
            self.mapping_confirmed = True
            self.mapping_status_label.setText("✓ Mapping confirmed successfully!")
            self.mapping_status_label.setStyleSheet("color: #2e7d32; padding: 10px;")
            self.validate_step2()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying mapping: {str(e)}")

    def validate_step2(self):
        """Validate Step 2 and enable Next button"""
        if self.mapping_confirmed:
            self.next_button.setEnabled(True)
            self.next_button.setText("Next")
        else:
            self.next_button.setEnabled(False)

    def setup_step3_settings(self):
        """Step 3: Analysis Settings"""
        step3_widget = QWidget()
        step3_layout = QVBoxLayout()
        step3_widget.setLayout(step3_layout)

        # Title
        title = QLabel("Step 3: Configure Analysis Settings")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        step3_layout.addWidget(title)

        # Create scroll area for settings
        from PySide6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        step3_layout.addWidget(scroll_area)

        # Experimental Setup Section
        exp_group = QGroupBox("Experimental Setup")
        exp_layout = QVBoxLayout()
        exp_group.setLayout(exp_layout)
        
        self.analysis_type_group = QButtonGroup()
        self.treadmill_radio = QRadioButton("Treadmill")
        self.spontaneous_radio = QRadioButton("Spontaneous walking")
        self.treadmill_radio.setChecked(True)
        self.analysis_type_group.addButton(self.treadmill_radio, 0)
        self.analysis_type_group.addButton(self.spontaneous_radio, 1)
        self.analysis_type_group.buttonClicked.connect(self.on_analysis_type_changed)
        
        exp_layout.addWidget(self.treadmill_radio)
        exp_layout.addWidget(self.spontaneous_radio)
        
        exp_help = QLabel("Treadmill: constant speed walking. Spontaneous: free walking with variable speed.")
        exp_help.setStyleSheet("color: #666; font-size: 11px;")
        exp_help.setWordWrap(True)
        exp_layout.addWidget(exp_help)
        
        scroll_layout.addWidget(exp_group)

        # Speed & Calibration Section (for Treadmill)
        self.speed_group = QGroupBox("Speed & Calibration (Treadmill)")
        speed_layout = QVBoxLayout()
        self.speed_group.setLayout(speed_layout)
        
        # Treadmill Speed
        speed_input_layout = QHBoxLayout()
        speed_input_layout.addWidget(QLabel("Treadmill Speed (cm/s):"))
        self.treadmill_speed_input = QDoubleSpinBox()
        self.treadmill_speed_input.setRange(0.1, 100.0)
        self.treadmill_speed_input.setValue(self.cm_speed if self.cm_speed else 30.0)
        self.treadmill_speed_input.setDecimals(1)
        speed_input_layout.addWidget(self.treadmill_speed_input)
        speed_layout.addLayout(speed_input_layout)
        
        speed_help = QLabel("The belt speed of your treadmill in cm/s. Used to calculate stride parameters.")
        speed_help.setStyleSheet("color: #666; font-size: 11px;")
        speed_help.setWordWrap(True)
        speed_layout.addWidget(speed_help)
        
        # Frame Rate
        fps_input_layout = QHBoxLayout()
        fps_input_layout.addWidget(QLabel("Frame Rate (fps):"))
        self.frame_rate_input = QDoubleSpinBox()
        self.frame_rate_input.setRange(1.0, 1000.0)
        self.frame_rate_input.setValue(self.frame_rate)
        self.frame_rate_input.setDecimals(1)
        fps_input_layout.addWidget(self.frame_rate_input)
        
        # Load FPS from video button
        self.load_fps_button = QPushButton("Load from Video")
        self.load_fps_button.clicked.connect(self.on_load_fps_from_video)
        fps_input_layout.addWidget(self.load_fps_button)
        
        speed_layout.addLayout(fps_input_layout)
        
        fps_help = QLabel("Video frame rate in frames per second. Critical for accurate temporal measurements. Click 'Load from Video' to detect FPS automatically.")
        fps_help.setStyleSheet("color: #666; font-size: 11px;")
        fps_help.setWordWrap(True)
        speed_layout.addWidget(fps_help)
        
        scroll_layout.addWidget(self.speed_group)

        # Spatial Calibration Method Section
        calib_group = QGroupBox("Spatial Calibration Method")
        calib_layout = QVBoxLayout()
        calib_group.setLayout(calib_layout)
        
        self.calib_method_group = QButtonGroup()
        self.reference_radio = QRadioButton("Reference Body Segment (Recommended)")
        self.manual_radio = QRadioButton("Manual Pixel-to-CM Ratio")
        self.reference_radio.setChecked(True)
        self.calib_method_group.addButton(self.reference_radio, 0)
        self.calib_method_group.addButton(self.manual_radio, 1)
        self.calib_method_group.buttonClicked.connect(self.on_calib_method_changed)
        
        calib_layout.addWidget(self.reference_radio)
        calib_layout.addWidget(self.manual_radio)
        
        # Reference segment settings
        self.reference_settings = QWidget()
        ref_layout = QVBoxLayout()
        self.reference_settings.setLayout(ref_layout)
        
        ref_segment_layout = QHBoxLayout()
        ref_segment_layout.addWidget(QLabel("Reference Segment:"))
        self.reference_segment_combo = QComboBox()
        self.reference_segment_combo.addItems([
            "ankle_toe (1.5cm)", "hip_knee (2.5cm)", 
            "knee_ankle (2.0cm)", "ankle_mtp (0.8cm)"
        ])
        self.reference_segment_combo.setCurrentText("ankle_toe (1.5cm)")
        ref_segment_layout.addWidget(self.reference_segment_combo)
        ref_layout.addLayout(ref_segment_layout)
        
        ref_length_layout = QHBoxLayout()
        ref_length_layout.addWidget(QLabel("Segment Length (cm):"))
        self.reference_length_input = QDoubleSpinBox()
        self.reference_length_input.setRange(0.1, 10.0)
        self.reference_length_input.setValue(1.5)
        self.reference_length_input.setDecimals(2)
        ref_length_layout.addWidget(self.reference_length_input)
        ref_layout.addLayout(ref_length_layout)
        
        ref_help = QLabel("Uses known body segment length for camera-independent calibration. ankle_toe = distance between ankle and toe joint (~1.5cm in mice). Recommended for accuracy across different camera distances.")
        ref_help.setStyleSheet("color: #666; font-size: 11px;")
        ref_help.setWordWrap(True)
        ref_layout.addWidget(ref_help)
        
        calib_layout.addWidget(self.reference_settings)
        
        # Manual calibration settings
        self.manual_settings = QWidget()
        manual_layout = QVBoxLayout()
        self.manual_settings.setLayout(manual_layout)
        
        manual_input_layout = QHBoxLayout()
        manual_input_layout.addWidget(QLabel("Pixels per CM:"))
        self.pixels_per_cm_input = QDoubleSpinBox()
        self.pixels_per_cm_input.setRange(1.0, 1000.0)
        self.pixels_per_cm_input.setValue(self.pixels_per_cm if self.pixels_per_cm else 50.0)
        self.pixels_per_cm_input.setDecimals(2)
        manual_input_layout.addWidget(self.pixels_per_cm_input)
        manual_layout.addLayout(manual_input_layout)
        
        manual_help = QLabel("If you've pre-calculated pixel-to-cm ratio, enter it here. Only use if you have an accurate measurement from your setup.")
        manual_help.setStyleSheet("color: #666; font-size: 11px;")
        manual_help.setWordWrap(True)
        manual_layout.addWidget(manual_help)
        
        calib_layout.addWidget(self.manual_settings)
        self.manual_settings.hide()  # Hide by default
        
        scroll_layout.addWidget(calib_group)

        # Movement Analysis Settings Section
        movement_group = QGroupBox("Movement Analysis Settings")
        movement_layout = QVBoxLayout()
        movement_group.setLayout(movement_layout)
        
        # Walking Direction
        direction_layout = QHBoxLayout()
        direction_layout.addWidget(QLabel("Walking Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Auto-detect", "Left-to-Right", "Right-to-Left"])
        if self.right_to_left == "auto":
            self.direction_combo.setCurrentText("Auto-detect")
        elif self.right_to_left:
            self.direction_combo.setCurrentText("Right-to-Left")
        else:
            self.direction_combo.setCurrentText("Left-to-Right")
        direction_layout.addWidget(self.direction_combo)
        movement_layout.addLayout(direction_layout)
        
        direction_help = QLabel("Direction of movement. Auto-detect recommended. Manual override if detection fails.")
        direction_help.setStyleSheet("color: #666; font-size: 11px;")
        direction_help.setWordWrap(True)
        movement_layout.addWidget(direction_help)
        
        # Drag Clearance Threshold
        drag_layout = QHBoxLayout()
        drag_layout.addWidget(QLabel("Drag Clearance Threshold (cm):"))
        self.drag_clearance_input = QDoubleSpinBox()
        self.drag_clearance_input.setRange(0.01, 2.0)
        self.drag_clearance_input.setValue(self.drag_clearance_cm)
        self.drag_clearance_input.setDecimals(2)
        drag_layout.addWidget(self.drag_clearance_input)
        movement_layout.addLayout(drag_layout)
        
        drag_help = QLabel("Minimum toe height above ground (in cm) to NOT count as dragging. Values below this are marked as dragging. Typical: 0.1cm for mice.")
        drag_help.setStyleSheet("color: #666; font-size: 11px;")
        drag_help.setWordWrap(True)
        movement_layout.addWidget(drag_help)
        
        # Drag Consecutive Frames Threshold
        drag_frames_layout = QHBoxLayout()
        drag_frames_layout.addWidget(QLabel("Drag Detection Sensitivity (frames):"))
        self.drag_consecutive_frames_input = QSpinBox()
        self.drag_consecutive_frames_input.setRange(1, 10)
        self.drag_consecutive_frames_input.setValue(self.drag_min_consecutive_frames)
        drag_frames_layout.addWidget(self.drag_consecutive_frames_input)
        movement_layout.addLayout(drag_frames_layout)
        
        drag_frames_help = QLabel("Minimum consecutive frames of ground contact to count as dragging. Lower = more sensitive (detects brief drags). Higher = less noise. Default: 4 frames.")
        drag_frames_help.setStyleSheet("color: #666; font-size: 11px;")
        drag_frames_help.setWordWrap(True)
        movement_layout.addWidget(drag_frames_help)
        
        # Lowpass Filter Cutoff
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Lowpass Filter Cutoff (Hz):"))
        self.filter_cutoff_input = QDoubleSpinBox()
        self.filter_cutoff_input.setRange(0.1, 50.0)
        self.filter_cutoff_input.setValue(self.cutoff_f)
        self.filter_cutoff_input.setDecimals(1)
        filter_layout.addWidget(self.filter_cutoff_input)
        movement_layout.addLayout(filter_layout)
        
        filter_help = QLabel("Butterworth filter frequency to smooth coordinate data. Lower = more smoothing. Typical: 6Hz for mouse gait.")
        filter_help.setStyleSheet("color: #666; font-size: 11px;")
        filter_help.setWordWrap(True)
        movement_layout.addWidget(filter_help)
        
        scroll_layout.addWidget(movement_group)

        # Stride Filtering Section (Collapsible)
        self.filtering_group = QGroupBox("Stride Filtering (Optional)")
        filtering_layout = QVBoxLayout()
        self.filtering_group.setLayout(filtering_layout)
        
        # Step height range
        step_height_layout = QHBoxLayout()
        step_height_layout.addWidget(QLabel("Step Height Range (cm):"))
        self.step_height_min_input = QDoubleSpinBox()
        self.step_height_min_input.setRange(0.0, 5.0)
        self.step_height_min_input.setValue(self.step_height_min_cm)
        self.step_height_min_input.setDecimals(2)
        step_height_layout.addWidget(QLabel("Min:"))
        step_height_layout.addWidget(self.step_height_min_input)
        step_height_layout.addWidget(QLabel("Max:"))
        self.step_height_max_input = QDoubleSpinBox()
        self.step_height_max_input.setRange(0.0, 5.0)
        self.step_height_max_input.setValue(self.step_height_max_cm)
        self.step_height_max_input.setDecimals(2)
        step_height_layout.addWidget(self.step_height_max_input)
        filtering_layout.addLayout(step_height_layout)
        
        # Stride length range
        stride_length_layout = QHBoxLayout()
        stride_length_layout.addWidget(QLabel("Stride Length Range (cm):"))
        self.stride_length_min_input = QDoubleSpinBox()
        self.stride_length_min_input.setRange(0.0, 20.0)
        self.stride_length_min_input.setValue(self.stride_length_min_cm)
        self.stride_length_min_input.setDecimals(2)
        stride_length_layout.addWidget(QLabel("Min:"))
        stride_length_layout.addWidget(self.stride_length_min_input)
        stride_length_layout.addWidget(QLabel("Max:"))
        self.stride_length_max_input = QDoubleSpinBox()
        self.stride_length_max_input.setRange(0.0, 20.0)
        self.stride_length_max_input.setValue(self.stride_length_max_cm)
        self.stride_length_max_input.setDecimals(2)
        stride_length_layout.addWidget(self.stride_length_max_input)
        filtering_layout.addLayout(stride_length_layout)
        
        filtering_help = QLabel("Filter out strides outside these ranges. Use to exclude abnormal movements.")
        filtering_help.setStyleSheet("color: #666; font-size: 11px;")
        filtering_help.setWordWrap(True)
        filtering_layout.addWidget(filtering_help)
        
        scroll_layout.addWidget(self.filtering_group)

        step3_layout.addWidget(scroll_area)
        self.wizard_stack.addWidget(step3_widget)

    def on_analysis_type_changed(self):
        """Handle analysis type change"""
        if self.treadmill_radio.isChecked():
            self.analysis_type = "Treadmill"
            self.speed_group.setVisible(True)
        else:
            self.analysis_type = "Spontaneous walking"
            self.speed_group.setVisible(False)

    def on_calib_method_changed(self):
        """Handle calibration method change"""
        if self.reference_radio.isChecked():
            self.calibration_method = 'reference'
            self.reference_settings.show()
            self.manual_settings.hide()
        else:
            self.calibration_method = 'manual'
            self.reference_settings.hide()
            self.manual_settings.show()

    def on_load_fps_from_video(self):
        """Load FPS from a video file"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, 
            'Select a video file to read FPS', 
            '', 
            'Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)'
        )
        
        if video_path:
            try:
                fps = ConfigFunctions.update_fps_from_video('./config.yaml', video_path)
                if fps is not None:
                    self.frame_rate_input.setValue(fps)
                    self.frame_rate = fps
                    QMessageBox.information(
                        self, 
                        "FPS Detected", 
                        f"Frame rate detected: {fps} fps\n\nThe config.yaml file has been updated."
                    )
                else:
                    QMessageBox.critical(
                        self, 
                        "Error", 
                        "Could not read FPS from the selected video.\n\nPlease check that:\n- The video file is not corrupted\n- The format is supported (mp4, avi, mov, mkv)\n- You have read permissions"
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error reading video: {str(e)}")

    def validate_step3(self):
        """Validate Step 3 and enable Next button"""
        # All settings are valid by default since we use spinboxes with ranges
        self.next_button.setEnabled(True)
        self.next_button.setText("Next")

    def setup_step4_output(self):
        """Step 4: Output & Run"""
        step4_widget = QWidget()
        step4_wrapper = QVBoxLayout()
        step4_wrapper.setContentsMargins(0, 0, 0, 0)
        step4_widget.setLayout(step4_wrapper)
        
        # Create scroll area for step 4
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        step4_wrapper.addWidget(scroll_area)
        
        # Content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        step4_layout = QVBoxLayout()
        content_widget.setLayout(step4_layout)

        # Title
        title = QLabel("Step 4: Select Output & Run Analysis")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        step4_layout.addWidget(title)

        # Output folder selection
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        
        self.select_output_button = QPushButton("Select Output Folder")
        self.select_output_button.clicked.connect(self.select_output_folder)
        self.select_output_button.setMinimumHeight(40)
        output_layout.addWidget(self.select_output_button)
        
        self.output_path_label = QLabel("No output folder selected")
        self.output_path_label.setAlignment(Qt.AlignCenter)
        self.output_path_label.setStyleSheet("color: #888; padding: 10px;")
        output_layout.addWidget(self.output_path_label)
        
        step4_layout.addWidget(output_group)

        # Settings summary
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        step4_layout.addWidget(summary_group)

        # Progress section
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setVisible(False)
        progress_layout.addWidget(self.progress_label)
        
        step4_layout.addWidget(progress_group)

        # Action buttons
        button_layout = QHBoxLayout()
        
        self.start_analysis_button = QPushButton("Start Analysis")
        self.start_analysis_button.clicked.connect(self.start_analysis)
        self.start_analysis_button.setMinimumHeight(50)
        self.start_analysis_button.setEnabled(False)
        
        self.analyze_another_button = QPushButton("Analyze Another File")
        self.analyze_another_button.clicked.connect(self.analyze_another_file)
        self.analyze_another_button.setMinimumHeight(50)
        self.analyze_another_button.setVisible(False)
        
        button_layout.addWidget(self.start_analysis_button)
        button_layout.addWidget(self.analyze_another_button)
        step4_layout.addLayout(button_layout)

        step4_layout.addStretch()
        self.wizard_stack.addWidget(step4_widget)

    def select_output_folder(self):
        """Select output folder for analysis results"""
        folder_path = QFileDialog.getExistingDirectory(
            self, 'Select folder to save analysis results', self.dirname)
        
        if folder_path:
            self.output_folder = folder_path
            self.output_path_label.setText(f"Output folder: {folder_path}")
            self.output_path_label.setStyleSheet("color: #2e7d32; padding: 10px;")
            self.update_analysis_summary()
            self.validate_step4()

    def update_analysis_summary(self):
        """Update the analysis summary text"""
        if not self.output_folder:
            return
            
        summary = "Analysis Configuration:\n\n"
        summary += f"Files: {len(self.selected_files)} file(s)\n"
        if self.is_bulk_analysis:
            summary += f"Mode: Bulk analysis\n"
        else:
            summary += f"Mode: Single file analysis\n"
            summary += f"File: {os.path.basename(self.selected_files[0])}\n"
        
        summary += f"Analysis Type: {self.analysis_type}\n"
        
        if self.analysis_type == "Treadmill":
            summary += f"Treadmill Speed: {self.treadmill_speed_input.value()} cm/s\n"
        
        summary += f"Frame Rate: {self.frame_rate_input.value()} fps\n"
        summary += f"Calibration: {self.calibration_method}\n"
        
        if self.calibration_method == 'reference':
            segment = self.reference_segment_combo.currentText()
            summary += f"Reference Segment: {segment}\n"
            summary += f"Segment Length: {self.reference_length_input.value()} cm\n"
        else:
            summary += f"Pixels per CM: {self.pixels_per_cm_input.value()}\n"
        
        direction = self.direction_combo.currentText()
        summary += f"Walking Direction: {direction}\n"
        summary += f"Drag Clearance: {self.drag_clearance_input.value()} cm\n"
        summary += f"Filter Cutoff: {self.filter_cutoff_input.value()} Hz\n"
        
        self.summary_text.setText(summary)

    def validate_step4(self):
        """Validate Step 4 and enable Start Analysis button"""
        if self.output_folder:
            self.start_analysis_button.setEnabled(True)
        else:
            self.start_analysis_button.setEnabled(False)

    def start_analysis(self):
        """Start the analysis process"""
        if not self.output_folder or not self.selected_files:
            return
        
        # Update UI for analysis
        self.start_analysis_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Preparing analysis...")
        
        # Collect settings
        self.collect_analysis_settings()
        
        # Start analysis in a separate thread
        self.analysis_thread = AnalysisThread(self)
        self.analysis_thread.progress_updated.connect(self.update_progress)
        self.analysis_thread.analysis_completed.connect(self.analysis_completed)
        self.analysis_thread.start()

    def collect_analysis_settings(self):
        """Collect all settings from the wizard"""
        self.analysis_settings = {
            'analysis_type': self.analysis_type,
            'frame_rate': self.frame_rate_input.value(),
            'calibration_method': self.calibration_method,
            'drag_clearance_cm': self.drag_clearance_input.value(),
            'drag_min_consecutive_frames': self.drag_consecutive_frames_input.value(),
            'filter_cutoff': self.filter_cutoff_input.value(),
            'step_height_min_cm': self.step_height_min_input.value(),
            'step_height_max_cm': self.step_height_max_input.value(),
            'stride_length_min_cm': self.stride_length_min_input.value(),
            'stride_length_max_cm': self.stride_length_max_input.value(),
        }
        
        # Direction setting
        direction = self.direction_combo.currentText()
        if direction == "Auto-detect":
            self.analysis_settings['right_to_left'] = "auto"
        elif direction == "Right-to-Left":
            self.analysis_settings['right_to_left'] = True
        else:
            self.analysis_settings['right_to_left'] = False
        
        # Calibration settings
        if self.calibration_method == 'reference':
            segment_text = self.reference_segment_combo.currentText()
            segment = segment_text.split(' ')[0]  # Extract "ankle_toe" from "ankle_toe (1.5cm)"
            self.analysis_settings['reference_segment'] = segment
            self.analysis_settings['reference_length_cm'] = self.reference_length_input.value()
            self.analysis_settings['auto_calibrate_spatial'] = True
            # For reference method, we don't set pixels_per_cm directly
            self.analysis_settings['pixels_per_cm'] = None
        else:
            self.analysis_settings['pixels_per_cm'] = self.pixels_per_cm_input.value()
            self.analysis_settings['auto_calibrate_spatial'] = False
            self.analysis_settings['reference_segment'] = None
            self.analysis_settings['reference_length_cm'] = None
        
        
        # Treadmill settings
        if self.analysis_type == "Treadmill":
            self.analysis_settings['cm_speed'] = self.treadmill_speed_input.value()

    def update_progress(self, value, text):
        """Update progress bar and label"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def analysis_completed(self, success, message):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        if success:
            self.progress_label.setText("✓ Analysis completed successfully!")
            self.progress_label.setStyleSheet("color: #2e7d32; padding: 10px;")
            self.progress_label.setVisible(True)
            self.analyze_another_button.setVisible(True)
            QMessageBox.information(self, "Analysis Complete", 
                f"Analysis completed successfully!\n\nResults saved to: {self.output_folder}")
        else:
            self.progress_label.setText(f"✗ Analysis failed: {message}")
            self.progress_label.setStyleSheet("color: #d32f2f; padding: 10px;")
            self.progress_label.setVisible(True)
            QMessageBox.critical(self, "Analysis Failed", f"Analysis failed: {message}")
        
        self.start_analysis_button.setEnabled(True)

    def analyze_another_file(self):
        """Reset wizard to analyze another file"""
        # Reset wizard state
        self.wizard_step = 1
        self.mapping_confirmed = False
        self.selected_files = []
        self.output_folder = None
        self.is_bulk_analysis = False
        
        # Reset UI
        self.file_status_label.setText("No files selected")
        self.file_status_label.setStyleSheet("color: #888; padding: 10px;")
        self.mapping_status_label.setText("Please confirm the mapping above to continue")
        self.mapping_status_label.setStyleSheet("color: #d32f2f; padding: 10px;")
        self.output_path_label.setText("No output folder selected")
        self.output_path_label.setStyleSheet("color: #888; padding: 10px;")
        self.summary_text.clear()
        self.progress_label.setVisible(False)
        self.analyze_another_button.setVisible(False)
        
        # Go back to step 1
        self.wizard_stack.setCurrentIndex(0)
        self.back_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.next_button.setText("Next")

    def go_next(self):
        """Navigate to next step"""
        if self.wizard_step < 4:
            self.wizard_step += 1
            self.wizard_stack.setCurrentIndex(self.wizard_step - 1)
            
            # Load data for step 2 when entering it
            if self.wizard_step == 2:
                self.load_mapping_data()
            
            # Update navigation buttons
            self.back_button.setEnabled(True)
            if self.wizard_step == 4:
                self.next_button.setText("Start Analysis")
                self.next_button.setEnabled(False)  # Will be enabled when output folder is selected
            else:
                self.next_button.setText("Next")
                self.next_button.setEnabled(False)
            
            # Validate current step
            if self.wizard_step == 1:
                self.validate_step1()
            elif self.wizard_step == 2:
                self.validate_step2()
            elif self.wizard_step == 3:
                self.validate_step3()

    def go_back(self):
        """Navigate to previous step"""
        if self.wizard_step > 1:
            self.wizard_step -= 1
            self.wizard_stack.setCurrentIndex(self.wizard_step - 1)
            
            # Update navigation buttons
            if self.wizard_step == 1:
                self.back_button.setEnabled(False)
            else:
                self.back_button.setEnabled(True)
            
            self.next_button.setText("Next")
            self.next_button.setEnabled(False)
            
            # Validate current step
            if self.wizard_step == 1:
                self.validate_step1()
            elif self.wizard_step == 2:
                self.validate_step2()
            elif self.wizard_step == 3:
                self.validate_step3()


class AnalysisThread(QThread):
    """Thread for running analysis in background"""
    progress_updated = Signal(int, str)
    analysis_completed = Signal(bool, str)
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
    
    def run(self):
        """Run the analysis"""
        try:
            self.progress_updated.emit(10, "Loading data...")
            
            # Process each file
            total_files = len(self.parent.selected_files)
            for i, file_path in enumerate(self.parent.selected_files):
                self.progress_updated.emit(10 + (i * 80 // total_files), 
                    f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
                
                # Load and process file using the same method as KinematicsFunctions
                import pandas as pd
                df = pd.read_csv(file_path, header=[1,2])
                
                # Apply bodypart mapping if available
                if self.parent.custom_bodypart_mapping:
                    df, bodyparts, bodyparts_raw = KinematicsFunctions.fix_column_names(
                        df, self.parent.custom_bodypart_mapping)
                else:
                    bodyparts_raw = [col[0] for col in df.columns if col[1] == 'x']
                    bodyparts = bodyparts_raw
                
                # Extract parameters
                if self.parent.analysis_type == "Treadmill":
                    # For treadmill analysis, we need to determine which bodypart to use
                    # Use the first available bodypart from the mapping
                    bodypart = 'toe'  # Default, but we should use the mapped bodypart
                    if self.parent.custom_bodypart_mapping:
                        # Find the first mapped bodypart
                        for mapped_bp in self.parent.custom_bodypart_mapping.values():
                            if mapped_bp in bodyparts:
                                bodypart = mapped_bp
                                break
                    
                    parameters, pd_dataframe_coords, is_stance, bodyparts, drag_masks, starts = KinematicsFunctions.extract_parameters(
                        self.parent.analysis_settings['frame_rate'],
                        df,
                        self.parent.analysis_settings['filter_cutoff'],
                        bodypart,
                        cm_speed=self.parent.analysis_settings.get('cm_speed'),
                        right_to_left=self.parent.analysis_settings['right_to_left'],
                        step_height_min_cm=self.parent.analysis_settings['step_height_min_cm'],
                        step_height_max_cm=self.parent.analysis_settings['step_height_max_cm'],
                        stride_length_min_cm=self.parent.analysis_settings['stride_length_min_cm'],
                        stride_length_max_cm=self.parent.analysis_settings['stride_length_max_cm'],
                        drag_clearance_cm=self.parent.analysis_settings['drag_clearance_cm'],
                        drag_min_consecutive_frames=self.parent.analysis_settings['drag_min_consecutive_frames']
                    )
                else:
                    # Handle pixels_per_cm - use default if None or empty
                    pixels_per_cm = self.parent.analysis_settings.get('pixels_per_cm')
                    if pixels_per_cm is None or pixels_per_cm == '':
                        pixels_per_cm = 49.143
                    
                    parameters, pd_dataframe_coords, is_stance, bodyparts, drag_masks, starts = KinematicsFunctions.extract_spontaneous_parameters(
                        self.parent.analysis_settings['frame_rate'],
                        df,
                        self.parent.analysis_settings['filter_cutoff'],
                        pixels_per_cm=pixels_per_cm,
                        no_outlier_filter=False,
                        dragging_filter=False,
                        step_height_min_cm=self.parent.analysis_settings['step_height_min_cm'],
                        step_height_max_cm=self.parent.analysis_settings['step_height_max_cm'],
                        stride_length_min_cm=self.parent.analysis_settings['stride_length_min_cm'],
                        stride_length_max_cm=self.parent.analysis_settings['stride_length_max_cm'],
                        drag_clearance_cm=self.parent.analysis_settings['drag_clearance_cm'],
                        drag_min_consecutive_frames=self.parent.analysis_settings['drag_min_consecutive_frames']
                    )
                
                # Save results
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(self.parent.output_folder, f"{base_name}_parameters.csv")
                parameters.to_csv(output_file, index=False)
                
                # Save coordinate data if available
                if pd_dataframe_coords is not None:
                    coords_file = os.path.join(self.parent.output_folder, f"{base_name}_coordinates.csv")
                    pd_dataframe_coords.to_csv(coords_file, index=False)
                
                # Generate stickplot visualization
                try:
                    self.progress_updated.emit(90 + (i * 5 // total_files), 
                        f"Generating stickplot for {os.path.basename(file_path)}...")
                    
                    # Prepare SVG filename for stickplot
                    stickplot_file = os.path.join(self.parent.output_folder, f"{base_name}_stickplot.svg")
                    
                    # Generate continuous stickplot with drag visualization
                    # Pass the full SVG path so the function saves it directly as SVG
                    stickplot_result = KinematicsFunctions.return_continuous(
                        parameters, 
                        n_continuous=10,  # 10 consecutive strides as requested
                        plot=True, 
                        pd_dataframe_coords=pd_dataframe_coords, 
                        bodyparts=bodyparts, 
                        is_stance=is_stance, 
                        filename=stickplot_file,  # Pass full SVG path
                        drag_masks=drag_masks,
                        starts=starts,
                        drag_min_consecutive_frames=self.parent.analysis_settings['drag_min_consecutive_frames']
                    )
                    
                    # The function saves the file internally, just close any remaining figures
                    if plt.get_fignums():
                        plt.close('all')  # Close all figures to free memory
                    
                    if os.path.exists(stickplot_file):
                        print(f"Stickplot saved to: {stickplot_file}")
                    else:
                        print(f"No figure was created for {filename} - skipping stickplot")
                        
                except Exception as e:
                    print(f"Warning: Could not generate stickplot for {filename}: {e}")
                    # Continue with analysis even if stickplot fails
            
            self.progress_updated.emit(100, "Analysis completed!")
            self.analysis_completed.emit(True, "Analysis completed successfully")
            
        except Exception as e:
            self.analysis_completed.emit(False, str(e))

