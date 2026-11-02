from PySide6.QtWidgets import (QWidget, QGridLayout, QLabel, QPushButton, QFileDialog, 
                               QMessageBox, QRadioButton, QButtonGroup, QComboBox, 
                               QLineEdit, QCheckBox, QProgressBar, QTextEdit, QGroupBox,
                               QVBoxLayout, QHBoxLayout, QSpinBox, QDoubleSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QScrollArea)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, Signal
from Functions import ConfigFunctions, KinematicsFunctions
import os
import json
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
        self.runway_defaults = {
            'confidence': configs.get('runway_confidence_threshold',.6),
            'max_gap_s': configs.get('runway_max_gap_s',.02),
            'min_body_speed': configs.get('runway_min_body_speed',.25),
            'min_contact_s': configs.get('runway_min_contact_s',.03),
            'joint_interpolation': configs.get('runway_joint_interpolation','none'),
            'joint_max_gap_s': configs.get('runway_joint_max_gap_s',.02),
            'joint_max_interpolated_fraction': configs.get('runway_joint_max_interpolated_fraction',.1),
            'joint_max_segment_change': configs.get('runway_joint_max_segment_change',.35),
            'joint_cutoff': configs.get('runway_joint_filter_cutoff'),
            'stickplot_source': configs.get('runway_stickplot_source','tracked'),
        }
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
        self.calibration_method = 'belt_motion'
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
        
        speed_help = QLabel("With belt-motion calibration, this speed sets the inferred spatial scale. With a measured calibration, it is a recorded cross-check: correction still follows tracked belt motion, and the report gives its implied speed in cm/s.")
        speed_help.setStyleSheet("color: #666; font-size: 11px;")
        speed_help.setWordWrap(True)
        speed_layout.addWidget(speed_help)
        
        # Frame rate applies to both treadmill and spontaneous recordings.
        fps_group = QGroupBox("Frame Rate")
        fps_layout = QVBoxLayout(fps_group)
        fps_input_layout = QHBoxLayout()
        fps_input_layout.addWidget(QLabel("Frame Rate (fps):"))
        self.frame_rate_input = QDoubleSpinBox()
        self.frame_rate_input.setRange(1.0, 1000.0)
        self.frame_rate_input.setValue(self.frame_rate)
        self.frame_rate_input.setDecimals(2)
        fps_input_layout.addWidget(self.frame_rate_input)
        
        # Load FPS from video button
        self.load_fps_button = QPushButton("Load from Video")
        self.load_fps_button.clicked.connect(self.on_load_fps_from_video)
        fps_input_layout.addWidget(self.load_fps_button)
        
        fps_layout.addLayout(fps_input_layout)
        
        fps_help = QLabel("Video frame rate in frames per second. Critical for accurate temporal measurements. Click 'Load from Video' to detect FPS automatically.")
        fps_help.setStyleSheet("color: #666; font-size: 11px;")
        fps_help.setWordWrap(True)
        fps_layout.addWidget(fps_help)
        
        scroll_layout.addWidget(self.speed_group)
        scroll_layout.addWidget(fps_group)

        # Spatial Calibration Method Section
        calib_group = QGroupBox("Spatial Calibration Method")
        calib_layout = QVBoxLayout()
        calib_group.setLayout(calib_layout)
        
        self.calib_method_group = QButtonGroup()
        self.belt_radio = QRadioButton("Belt-motion estimate (existing treadmill method)")
        self.reference_radio = QRadioButton("Measured Reference Body Segment")
        self.manual_radio = QRadioButton("Measured Pixels per CM")
        self.belt_radio.setChecked(True)
        self.calib_method_group.addButton(self.belt_radio, 2)
        self.calib_method_group.addButton(self.reference_radio, 0)
        self.calib_method_group.addButton(self.manual_radio, 1)
        self.calib_method_group.buttonClicked.connect(self.on_calib_method_changed)
        
        calib_layout.addWidget(self.belt_radio)
        calib_layout.addWidget(self.reference_radio)
        calib_layout.addWidget(self.manual_radio)
        self.calibration_help = QLabel("Belt-motion estimation uses tracked belt movement, FPS and belt speed to infer the spatial scale. A measured scale changes unit conversion; pixel motion correction still uses tracked belt movement. For spontaneous walking, enter a measured pixels-per-cm ratio.")
        self.calibration_help.setWordWrap(True)
        self.calibration_help.setStyleSheet("color: #666; font-size: 11px;")
        calib_layout.addWidget(self.calibration_help)
        
        # Reference segment settings
        self.reference_settings = QWidget()
        ref_layout = QVBoxLayout()
        self.reference_settings.setLayout(ref_layout)
        
        ref_segment_layout = QHBoxLayout()
        ref_segment_layout.addWidget(QLabel("Reference Segment:"))
        self.reference_segment_combo = QComboBox()
        self.reference_segment_combo.addItems([
            "ankle_toe (enter measured length)", "hip_knee (enter measured length)",
            "knee_ankle (enter measured length)", "ankle_mtp (enter measured length)"
        ])
        self.reference_segment_combo.setCurrentIndex(0)
        ref_segment_layout.addWidget(self.reference_segment_combo)
        ref_layout.addLayout(ref_segment_layout)
        
        ref_length_layout = QHBoxLayout()
        ref_length_layout.addWidget(QLabel("Measured Segment Length (cm):"))
        self.reference_length_input = QDoubleSpinBox()
        self.reference_length_input.setRange(0.0, 10.0)
        self.reference_length_input.setSpecialValueText("Enter measured length")
        self.reference_length_input.setValue(0.)
        self.reference_length_input.setDecimals(2)
        ref_length_layout.addWidget(self.reference_length_input)
        ref_layout.addLayout(ref_length_layout)
        
        ref_help = QLabel("Enter a length measured for this animal/setup; no anatomical length is assumed. At least 10 jointly confident frames and a stable segment are required; unreliable calibration produces an error without a fallback. Reference calibration is currently available for treadmill analysis.")
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
        self.pixels_per_cm_input.setRange(0.0, 1000.0)
        self.pixels_per_cm_input.setSpecialValueText("Enter measured scale")
        self.pixels_per_cm_input.setValue(self.pixels_per_cm if self.pixels_per_cm else 0.)
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

        self.runway_group = QGroupBox("Runway Detection and Kinematics")
        runway_layout = QVBoxLayout(self.runway_group)
        self.runway_confidence_input = QDoubleSpinBox()
        self.runway_confidence_input.setRange(.01,.99)
        self.runway_confidence_input.setDecimals(2)
        self.runway_confidence_input.setSingleStep(.05)
        self.runway_confidence_input.setValue(self.runway_defaults['confidence'])
        self.runway_gap_input = QDoubleSpinBox()
        self.runway_gap_input.setRange(0.,100.)
        self.runway_gap_input.setValue(self.runway_defaults['max_gap_s']*1000)
        self.runway_body_speed_input = QDoubleSpinBox()
        self.runway_body_speed_input.setRange(0.,10.)
        self.runway_body_speed_input.setDecimals(2)
        self.runway_body_speed_input.setValue(self.runway_defaults['min_body_speed'])
        self.runway_contact_input = QDoubleSpinBox()
        self.runway_contact_input.setRange(1.,500.)
        self.runway_contact_input.setValue(self.runway_defaults['min_contact_s']*1000)
        for label,control in [
                ('Tracking confidence threshold:',self.runway_confidence_input),
                ('Maximum gap bridged for event timing (ms):',self.runway_gap_input),
                ('Minimum body speed (reference lengths/s):',self.runway_body_speed_input),
                ('Minimum supported contact duration (ms):',self.runway_contact_input)]:
            line=QHBoxLayout()
            line.addWidget(QLabel(label))
            line.addWidget(control)
            runway_layout.addLayout(line)
        self.runway_body_speed_input.setToolTip('One reference length is the median valid toe-to-body-reference distance in this recording. This controls the moving-bout gate; it is not an assumed anatomical body length.')
        self.joint_interpolation_input = QComboBox()
        self.joint_interpolation_input.addItem('Observed only (no joint reconstruction)', 'none')
        self.joint_interpolation_input.addItem('Short-gap linear interpolation', 'linear')
        self.joint_interpolation_input.addItem('Short-gap PCHIP interpolation', 'pchip')
        self.joint_interpolation_input.setCurrentIndex(max(0,self.joint_interpolation_input.findData(self.runway_defaults['joint_interpolation'])))
        self.joint_gap_input = QDoubleSpinBox()
        self.joint_gap_input.setRange(0.,100.)
        self.joint_gap_input.setValue(self.runway_defaults['joint_max_gap_s']*1000)
        self.joint_fraction_input = QDoubleSpinBox()
        self.joint_fraction_input.setRange(0.,100.)
        self.joint_fraction_input.setValue(self.runway_defaults['joint_max_interpolated_fraction']*100)
        self.joint_segment_input = QDoubleSpinBox()
        self.joint_segment_input.setRange(0.,100.)
        self.joint_segment_input.setValue(self.runway_defaults['joint_max_segment_change']*100)
        joint_controls = [self.joint_gap_input,self.joint_fraction_input,self.joint_segment_input]
        for label,control in [
                ('Joint coordinate reconstruction:',self.joint_interpolation_input),
                ('Maximum joint gap to reconstruct (ms):',self.joint_gap_input),
                ('Maximum reconstructed frames per metric (%):',self.joint_fraction_input),
                ('Maximum local segment-length deviation (%):',self.joint_segment_input)]:
            line=QHBoxLayout()
            line.addWidget(QLabel(label))
            line.addWidget(control)
            runway_layout.addLayout(line)
        def update_joint_controls():
            enabled=self.joint_interpolation_input.currentData() != 'none'
            for control in joint_controls: control.setEnabled(enabled)
        self.joint_interpolation_input.currentIndexChanged.connect(update_joint_controls)
        update_joint_controls()
        self.joint_fraction_input.setToolTip('For each angle or distance, the union of reconstructed frames across its required joints must stay within this percentage of the cycle. Other metrics and timing remain available.')
        self.joint_segment_input.setToolTip('Reconstructed bone lengths are compared with local observed endpoint lengths. No anatomical length is assumed. A nearby observed joint is required for a geometry check.')
        self.joint_cutoff_checkbox = QCheckBox('Separate joint filter cutoff (Hz):')
        self.joint_cutoff_input = QDoubleSpinBox()
        self.joint_cutoff_input.setRange(.1,200.)
        self.joint_cutoff_input.setValue(self.runway_defaults['joint_cutoff'] or self.cutoff_f)
        self.joint_cutoff_checkbox.setChecked(self.runway_defaults['joint_cutoff'] is not None)
        self.joint_cutoff_input.setEnabled(self.joint_cutoff_checkbox.isChecked())
        self.joint_cutoff_checkbox.toggled.connect(self.joint_cutoff_input.setEnabled)
        line=QHBoxLayout()
        line.addWidget(self.joint_cutoff_checkbox)
        line.addWidget(self.joint_cutoff_input)
        runway_layout.addLayout(line)
        self.joint_cutoff_input.setToolTip('Optional filter for joint angles and toe-to-crest distance. Higher cutoffs preserve faster changes and more tracking noise. Stride detection and foot metrics retain the main filter cutoff. Unchecked inherits the main cutoff.')
        self.stickplot_source_input = QComboBox()
        self.stickplot_source_input.addItem('Tracked keypoints (unfiltered review)', 'tracked')
        self.stickplot_source_input.addItem('Analysis coordinates (filtered)', 'analysis')
        self.stickplot_source_input.setCurrentIndex(max(0,self.stickplot_source_input.findData(self.runway_defaults['stickplot_source'])))
        line=QHBoxLayout()
        line.addWidget(QLabel('Stick plot coordinates:'))
        line.addWidget(self.stickplot_source_input)
        runway_layout.addLayout(line)
        runway_help=QLabel('Only complete cycles inside moving bouts are scored. Event gap filling and optional joint reconstruction have separate limits. Joint reconstruction occurs before filtering, uses local geometry/motion checks, and flags estimated metrics and peaks. Dashed bones identify reconstructed endpoints. Long gaps and recording edges remain unavailable. Review estimates against video; likelihood scores are never replaced.')
        runway_help.setWordWrap(True)
        runway_help.setStyleSheet('color: #666; font-size: 11px;')
        runway_layout.addWidget(runway_help)
        review_help=QLabel('Tracked review draws finite input positions directly; low-confidence bones are gray and dotted. Filter padding never hides a visible tracked point in this view. Analysis plots show the filtered coordinates used for calculations. Changing the plot source does not change measurements.')
        review_help.setWordWrap(True)
        review_help.setStyleSheet('color: #666; font-size: 11px;')
        runway_layout.addWidget(review_help)
        self.runway_group.hide()
        scroll_layout.addWidget(self.runway_group)

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
        self._direction_by_mode = {'Treadmill': self.direction_combo.currentText(),
                                   'Spontaneous walking': 'Auto-detect'}
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
        
        drag_help = QLabel("Swing frames with toe clearance below this value count as contact. A larger value includes more contact; this control does not define stride boundaries.")
        drag_help.setStyleSheet("color: #666; font-size: 11px;")
        drag_help.setWordWrap(True)
        movement_layout.addWidget(drag_help)
        
        # Drag Consecutive Frames Threshold
        drag_frames_layout = QHBoxLayout()
        drag_frames_layout.addWidget(QLabel("Minimum Drag Contact Run (frames):"))
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
        
        filter_help = QLabel("Lower cutoff means more smoothing. Changing cutoff can move stride/phase boundaries and alter angle peaks; inspect these events against the video.")
        filter_help.setStyleSheet("color: #666; font-size: 11px;")
        filter_help.setWordWrap(True)
        movement_layout.addWidget(filter_help)
        
        scroll_layout.addWidget(movement_group)

        # Stride Filtering Section (Collapsible)
        self.filtering_group = QGroupBox("Stride Acceptance Limits")
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
        
        filtering_help = QLabel("These limits select intervals after detection and can change summary averages. Minimum step height 0 retains zero-height dragging steps; a positive minimum excludes them.")
        filtering_help.setStyleSheet("color: #666; font-size: 11px;")
        filtering_help.setWordWrap(True)
        filtering_layout.addWidget(filtering_help)
        
        scroll_layout.addWidget(self.filtering_group)
        self.settings_status_label = QLabel()
        self.settings_status_label.setWordWrap(True)
        self.settings_status_label.setStyleSheet("color: #b45309; padding: 8px;")
        for control in (self.reference_length_input, self.pixels_per_cm_input,
                        self.frame_rate_input, self.filter_cutoff_input,
                        self.step_height_min_input, self.step_height_max_input,
                        self.stride_length_min_input, self.stride_length_max_input):
            control.valueChanged.connect(lambda _: self.validate_step3())
        self.on_calib_method_changed()
        self.joint_cutoff_input.valueChanged.connect(lambda _: self.validate_step3())
        self.joint_cutoff_checkbox.toggled.connect(lambda _: self.validate_step3())

        step3_layout.addWidget(scroll_area)
        step3_layout.addWidget(self.settings_status_label)
        self.wizard_stack.addWidget(step3_widget)

    def on_analysis_type_changed(self):
        """Handle analysis type change"""
        previous = self.analysis_type
        self._direction_by_mode[previous] = self.direction_combo.currentText()
        if self.treadmill_radio.isChecked():
            self.analysis_type = "Treadmill"
            self.speed_group.setVisible(True)
            self.belt_radio.setEnabled(True)
            self.reference_radio.setEnabled(True)
            self.runway_group.hide()
        else:
            self.analysis_type = "Spontaneous walking"
            self.speed_group.setVisible(False)
            self.belt_radio.setEnabled(False)
            self.reference_radio.setEnabled(False)
            self.runway_group.show()
            if self.belt_radio.isChecked() or self.reference_radio.isChecked():
                self.manual_radio.setChecked(True)
        self.direction_combo.setCurrentText(self._direction_by_mode[self.analysis_type])
        self.on_calib_method_changed()

    def on_calib_method_changed(self):
        """Handle calibration method change"""
        if self.belt_radio.isChecked():
            self.calibration_method = 'belt_motion'
            self.reference_settings.hide()
            self.manual_settings.hide()
        elif self.reference_radio.isChecked():
            self.calibration_method = 'reference'
            self.reference_settings.show()
            self.manual_settings.hide()
        else:
            self.calibration_method = 'manual'
            self.reference_settings.hide()
            self.manual_settings.show()
        if hasattr(self, 'next_button'):
            self.validate_step3()

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
        message = ''
        if self.calibration_method == 'manual' and self.pixels_per_cm_input.value() <= 0:
            message = 'Enter a measured pixels-per-cm scale to continue.'
        elif self.calibration_method == 'reference' and self.reference_length_input.value() <= 0:
            message = 'Enter the measured reference-segment length to continue.'
        elif self.filter_cutoff_input.value() >= self.frame_rate_input.value() / 2:
            message = 'Filter cutoff must be less than half the frame rate.'
        elif self.analysis_type != 'Treadmill' and self.joint_cutoff_checkbox.isChecked() and self.joint_cutoff_input.value() >= self.frame_rate_input.value()/2:
            message = 'Joint filter cutoff must be less than half the frame rate.'
        elif self.step_height_min_input.value() > self.step_height_max_input.value():
            message = 'Minimum step height must not exceed the maximum.'
        elif self.stride_length_min_input.value() > self.stride_length_max_input.value():
            message = 'Minimum stride length must not exceed the maximum.'
        self.settings_status_label.setText(message)
        self.settings_status_label.setVisible(bool(message))
        if self.wizard_step == 3:
            self.next_button.setEnabled(not message)
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
        elif self.calibration_method == 'manual':
            summary += f"Pixels per CM: {self.pixels_per_cm_input.value()}\n"
        else:
            summary += "Spatial scale inferred from tracked belt motion, FPS and treadmill speed.\n"
        
        direction = self.direction_combo.currentText()
        summary += f"Walking Direction: {direction}\n"
        summary += f"Drag Clearance: {self.drag_clearance_input.value()} cm\n"
        summary += f"Minimum Drag Run: {self.drag_consecutive_frames_input.value()} frames\n"
        summary += f"Filter Cutoff: {self.filter_cutoff_input.value()} Hz\n"
        summary += f"Step Height Limits: {self.step_height_min_input.value()}–{self.step_height_max_input.value()} cm\n"
        summary += f"Stride Length Limits: {self.stride_length_min_input.value()}–{self.stride_length_max_input.value()} cm\n"
        if self.analysis_type != 'Treadmill':
            summary += f"Tracking Confidence: {self.runway_confidence_input.value()}\n"
            summary += f"Maximum Event Gap: {self.runway_gap_input.value()} ms\n"
            summary += f"Minimum Contact: {self.runway_contact_input.value()} ms\n"
            summary += f"Minimum Body Speed: {self.runway_body_speed_input.value()} reference lengths/s\n"
            summary += f"Joint Reconstruction: {self.joint_interpolation_input.currentText()}\n"
            summary += f"Joint Filter: {self.joint_cutoff_input.value() if self.joint_cutoff_checkbox.isChecked() else self.filter_cutoff_input.value()} Hz\n"
            summary += f"Stick Plot: {self.stickplot_source_input.currentText()}\n"
            if self.joint_interpolation_input.currentData() != 'none':
                summary += f"Joint Gap Limit: {self.joint_gap_input.value()} ms; Reconstructed Frame Limit: {self.joint_fraction_input.value()}%; Local Segment Deviation Limit: {self.joint_segment_input.value()}%\n"
            if self.no_outlier_filter:
                summary += "Distance/height range filters bypassed by configuration; tracking quality checks remain active.\n"
            if self.dragging_filter:
                summary += "Near-continuous dragging intervals excluded by configuration.\n"
        
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
        elif self.calibration_method == 'manual':
            self.analysis_settings['pixels_per_cm'] = self.pixels_per_cm_input.value()
            self.analysis_settings['auto_calibrate_spatial'] = False
            self.analysis_settings['reference_segment'] = None
            self.analysis_settings['reference_length_cm'] = None
        else:
            self.analysis_settings['pixels_per_cm'] = None
            self.analysis_settings['auto_calibrate_spatial'] = False
            self.analysis_settings['reference_segment'] = None
            self.analysis_settings['reference_length_cm'] = None
        
        
        # Treadmill settings
        if self.analysis_type == "Treadmill":
            self.analysis_settings['cm_speed'] = self.treadmill_speed_input.value()
        else:
            self.analysis_settings.update(confidence_threshold=self.runway_confidence_input.value(),
                max_gap_s=self.runway_gap_input.value()/1000,
                min_body_speed=self.runway_body_speed_input.value(),
                min_contact_s=self.runway_contact_input.value()/1000,
                joint_interpolation=self.joint_interpolation_input.currentData(),
                joint_max_gap_s=self.joint_gap_input.value()/1000,
                joint_max_interpolated_fraction=self.joint_fraction_input.value()/100,
                joint_max_segment_change=self.joint_segment_input.value()/100,
                joint_cutoff_f=self.joint_cutoff_input.value() if self.joint_cutoff_checkbox.isChecked() else None,
                stickplot_coordinate_source=self.stickplot_source_input.currentData(),
                no_outlier_filter=self.no_outlier_filter,dragging_filter=self.dragging_filter)

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
                f"{message}\n\nResults saved to: {self.output_folder}")
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
            analysis_reports = []
            for i, file_path in enumerate(self.parent.selected_files):
                self.progress_updated.emit(10 + (i * 80 // total_files), 
                    f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
                
                # Load and process file using the same method as KinematicsFunctions
                import pandas as pd
                df = pd.read_csv(file_path, header=[1,2])
                
                # Apply bodypart mapping if available
                df, bodyparts, bodyparts_raw = KinematicsFunctions.fix_column_names(
                    df, self.parent.custom_bodypart_mapping)
                
                # Extract parameters
                if self.parent.analysis_type == "Treadmill":
                    # Mapping dictionary order must not change calibration.
                    bodypart = 'toe'
                    
                    parameters, pd_dataframe_coords, is_stance, bodyparts, drag_masks, starts = KinematicsFunctions.extract_parameters(
                        self.parent.analysis_settings['frame_rate'],
                        df,
                        self.parent.analysis_settings['filter_cutoff'],
                        bodypart,
                        cm_speed=self.parent.analysis_settings.get('cm_speed'),
                        calibration_method=self.parent.analysis_settings.get('calibration_method', 'belt_motion'),
                        pixels_per_cm=self.parent.analysis_settings.get('pixels_per_cm'),
                        reference_segment=self.parent.analysis_settings.get('reference_segment'),
                        reference_length_cm=self.parent.analysis_settings.get('reference_length_cm'),
                        right_to_left=self.parent.analysis_settings['right_to_left'],
                        step_height_min_cm=self.parent.analysis_settings['step_height_min_cm'],
                        step_height_max_cm=self.parent.analysis_settings['step_height_max_cm'],
                        stride_length_min_cm=self.parent.analysis_settings['stride_length_min_cm'],
                        stride_length_max_cm=self.parent.analysis_settings['stride_length_max_cm'],
                        drag_clearance_cm=self.parent.analysis_settings['drag_clearance_cm'],
                        drag_min_consecutive_frames=self.parent.analysis_settings['drag_min_consecutive_frames']
                    )
                else:
                    # A physical scale must be supplied; never assume one.
                    pixels_per_cm = self.parent.analysis_settings.get('pixels_per_cm')
                    
                    parameters, pd_dataframe_coords, is_stance, bodyparts, drag_masks, starts = KinematicsFunctions.extract_spontaneous_parameters(
                        self.parent.analysis_settings['frame_rate'],
                        df,
                        self.parent.analysis_settings['filter_cutoff'],
                        pixels_per_cm=pixels_per_cm,
                        no_outlier_filter=self.parent.analysis_settings.get('no_outlier_filter',False),
                        dragging_filter=self.parent.analysis_settings.get('dragging_filter',False),
                        step_height_min_cm=self.parent.analysis_settings['step_height_min_cm'],
                        step_height_max_cm=self.parent.analysis_settings['step_height_max_cm'],
                        stride_length_min_cm=self.parent.analysis_settings['stride_length_min_cm'],
                        stride_length_max_cm=self.parent.analysis_settings['stride_length_max_cm'],
                        drag_clearance_cm=self.parent.analysis_settings['drag_clearance_cm'],
                        drag_min_consecutive_frames=self.parent.analysis_settings['drag_min_consecutive_frames'],
                        confidence_threshold=self.parent.analysis_settings.get('confidence_threshold',.6),
                        max_gap_s=self.parent.analysis_settings.get('max_gap_s',.02),
                        min_body_speed=self.parent.analysis_settings.get('min_body_speed',.25),
                        min_contact_s=self.parent.analysis_settings.get('min_contact_s',.03),
                        right_to_left=self.parent.analysis_settings.get('right_to_left','auto'),
                        joint_interpolation=self.parent.analysis_settings.get('joint_interpolation','none'),
                        joint_max_gap_s=self.parent.analysis_settings.get('joint_max_gap_s',.02),
                        joint_max_interpolated_fraction=self.parent.analysis_settings.get('joint_max_interpolated_fraction',.1),
                        joint_max_segment_change=self.parent.analysis_settings.get('joint_max_segment_change',.35),
                        joint_cutoff_f=self.parent.analysis_settings.get('joint_cutoff_f')
                    )
                
                # Save results
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(self.parent.output_folder, f"{base_name}_parameters.csv")
                parameters.to_csv(output_file, index=False)

                report = parameters.attrs.get('analysis_report')
                if report is not None:
                    report['source_csv'] = os.path.basename(file_path)
                    if self.parent.analysis_type != 'Treadmill':
                        report['stickplot_coordinate_source'] = self.parent.analysis_settings.get('stickplot_coordinate_source','tracked')
                    report_file = os.path.join(self.parent.output_folder, f"{base_name}_analysis_report.json")
                    with open(report_file, 'w', encoding='utf-8') as handle:
                        json.dump(report, handle, indent=2, allow_nan=False)
                    analysis_reports.append(report)
                
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
                        drag_min_consecutive_frames=self.parent.analysis_settings['drag_min_consecutive_frames'],
                        coordinate_source=self.parent.analysis_settings.get('stickplot_coordinate_source','tracked') if self.parent.analysis_type != 'Treadmill' else 'analysis'
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
            if analysis_reports:
                retained = sum(report['retained_intervals'] for report in analysis_reports)
                candidates = sum(report['candidate_intervals'] for report in analysis_reports)
                if retained == 0:
                    self.analysis_completed.emit(False,
                        "No usable stride measurements were retained. Diagnostic reports were saved with the results; check tracking, direction, calibration, and stride limits.")
                    return
                message = f"Analysis completed: {retained} of {candidates} candidate intervals retained."
                if any(report['warnings'] for report in analysis_reports):
                    message += "\nQuality warnings were found. Review the analysis_report.json files before interpreting measurements."
            else:
                message = "Analysis completed successfully"
            self.analysis_completed.emit(True, message)
            
        except Exception as e:
            self.analysis_completed.emit(False, str(e))

