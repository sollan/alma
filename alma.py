import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMenuBar, QStatusBar
from PySide6.QtCore import Qt
from Panels import Start, AnalyzeStride, AnalyzeFootfall, RandomForest, PCA


class HomeFrame(QMainWindow):

    def __init__(self, *args, **kw):
        
        super(HomeFrame, self).__init__(*args, **kw)

        # Create stacked widget for panel management
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create panels
        self.StartPanel = Start.StartPanel(self)
        self.AnalyzeFootfallPanel = AnalyzeFootfall.AnalyzeFootfallPanel(self)
        self.AnalyzeStridePanel = AnalyzeStride.AnalyzeStridePanel(self)
        self.RandomForestPanel = RandomForest.RandomForestPanel(self)
        self.PCAPanel = PCA.PCAPanel(self)

        # Add panels to stacked widget
        self.stacked_widget.addWidget(self.StartPanel)
        self.stacked_widget.addWidget(self.AnalyzeFootfallPanel)
        self.stacked_widget.addWidget(self.AnalyzeStridePanel)
        self.stacked_widget.addWidget(self.RandomForestPanel)
        self.stacked_widget.addWidget(self.PCAPanel)

        # Set current panel to Start
        self.stacked_widget.setCurrentWidget(self.StartPanel)

        # Set uniform background for main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QStackedWidget {
                background-color: #f5f7fa;
            }
        """)

        self.makeMenuBar()
        self.statusBar().showMessage("Welcome!")


    def makeMenuBar(self):
        """
        more functions / menu items can be added
        for additional features
        """
        from PySide6.QtWidgets import QMenuBar, QMenu
        from PySide6.QtGui import QAction

        menu_bar = self.menuBar()

        # Limb motion analysis menu
        limb_motion_menu = menu_bar.addMenu("&Limb motion analysis")
        
        start_action = QAction("&Start", self)
        start_action.setShortcut("Ctrl+S")
        start_action.triggered.connect(self.on_start)
        limb_motion_menu.addAction(start_action)

        footfall_action = QAction("&Footfall detection", self)
        footfall_action.setShortcut("Ctrl+L")
        footfall_action.setStatusTip("Detect footfalls and validate results")
        footfall_action.triggered.connect(self.on_analyze_footfall)
        limb_motion_menu.addAction(footfall_action)

        stride_action = QAction("&Gait kinematic data extraction", self)
        stride_action.setShortcut("Ctrl+K")
        stride_action.setStatusTip("Extract gait kinematic parameters with bodypart coordinate data")
        stride_action.triggered.connect(self.on_analyze_stride)
        limb_motion_menu.addAction(stride_action)

        # Data analysis menu
        analysis_menu = menu_bar.addMenu("&Data analysis")
        
        rf_action = QAction("&Random forest classification", self)
        rf_action.setShortcut("Ctrl+R")
        rf_action.setStatusTip("Random forest classification using extracted gait kinematic parameters")
        rf_action.triggered.connect(self.on_random_forest)
        analysis_menu.addAction(rf_action)

        pca_action = QAction("&PCA", self)
        pca_action.setShortcut("Ctrl+P")
        pca_action.setStatusTip("PCA using extracted gait kinematic parameters")
        pca_action.triggered.connect(self.on_PCA)
        analysis_menu.addAction(pca_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

        help_action = QAction("&Help", self)
        help_action.triggered.connect(self.on_help)
        help_menu.addAction(help_action)
    

    def on_start(self):
        self.stacked_widget.setCurrentWidget(self.StartPanel)
        self.statusBar().showMessage('Welcome!')
    

    def on_quit(self, e):

        self.Close(True)

    
    def on_analyze_footfall(self):
        self.stacked_widget.setCurrentWidget(self.AnalyzeFootfallPanel)
        self.statusBar().showMessage('Ready for automated footfall analysis')


    def on_analyze_stride(self):
        self.stacked_widget.setCurrentWidget(self.AnalyzeStridePanel)
        self.statusBar().showMessage('Ready for automated kinematic analysis')


    def on_random_forest(self):
        self.stacked_widget.setCurrentWidget(self.RandomForestPanel)
        self.statusBar().showMessage('Run random forest classification on extracted gait kinematic parameters')


    def on_PCA(self):
        self.stacked_widget.setCurrentWidget(self.PCAPanel)
        self.statusBar().showMessage('Run PCA on extracted gait kinematic parameters')


    def on_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "About ALMA", 
                               "This is a toolbox for fully automated (rodent) "
                               "limb motion analysis with markerless bodypart tracking data. "
                               "Compatible with csv output from DeepLabCut.")

    def on_help(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Support", 
                               "Please go to our GitHub Wiki for help or email us :)")


if __name__ == '__main__':
    
    from Functions import ConfigFunctions

    configs = ConfigFunctions.load_config('./config.yaml')
    window_width, window_height = configs['window_width'], configs['window_height']

    app = QApplication(sys.argv)
    home_frame = HomeFrame()
    home_frame.setWindowTitle('ALMA - Automated Limb Motion Analysis')
    # Set both the size and minimum size to respect config values
    home_frame.resize(window_width, window_height)
    home_frame.setMinimumSize(window_width, window_height)
    # Uncomment the next line to make the window non-resizable (fixed size):
    # home_frame.setFixedSize(window_width, window_height)
    home_frame.show()
    sys.exit(app.exec())