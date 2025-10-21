"""
Interactive Timeline Widget for Footfall Detection using PyQtGraph
Replaces rigid matplotlib visualization with dynamic, zoomable timeline
"""

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QGroupBox
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
import numpy as np


class InteractiveTimelineWidget(QWidget):
    """Interactive timeline for footfall/slip detection with PyQtGraph"""
    
    # Signals for UI integration
    frame_clicked = Signal(int)  # Emitted when user clicks on timeline to jump to frame
    marker_moved = Signal(int, str, int)  # (footfall_index, marker_type ['start'/'end'/'peak'], new_frame)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Data storage
        self.pd_dataframe = None
        self.bodyparts = []
        self.selected_bodyparts = []
        self.current_bodypart = None
        self.axis = 'y'
        
        # Detection data
        self.t_pred = []
        self.start_pred = []
        self.end_pred = []
        self.bodypart_list = []
        self.confirmed = []
        self.slip_fall_val = []
        
        # Display state
        self.n_current_frame = 0
        self.likelihood_threshold = 0.1
        self.zoom_enabled = True
        
        # Plot items storage
        self.bodypart_curves = {}
        self.marker_items = {}
        
    def setup_ui(self):
        """Setup the UI layout"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)
        
        # Compact control panel
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(5, 2, 5, 2)
        
        self.zoom_check = QCheckBox("Auto-Zoom")
        self.zoom_check.setChecked(True)
        self.zoom_check.stateChanged.connect(self.on_zoom_toggle)
        control_layout.addWidget(self.zoom_check)
        
        self.reset_zoom_btn = QPushButton("Reset")
        self.reset_zoom_btn.setMaximumWidth(60)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        control_layout.addWidget(self.reset_zoom_btn)
        
        control_layout.addWidget(QLabel("|"))
        
        # Bodypart visibility inline
        control_layout.addWidget(QLabel("Show:"))
        self.legend_layout = QHBoxLayout()
        self.legend_layout.setSpacing(8)
        control_layout.addLayout(self.legend_layout)
        
        control_layout.addStretch()
        
        self.info_label = QLabel("Click timeline to jump to frame")
        self.info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        control_layout.addWidget(self.info_label)
        
        layout.addLayout(control_layout)
        
        # Create PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'Frame', **{'font-size': '10pt'})
        self.plot_widget.setLabel('left', 'Y Position (px)', **{'font-size': '10pt'})
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMinimumHeight(200)
        self.plot_widget.setMaximumHeight(300)
        self.plot_widget.invertY(True)
        
        # Enable mouse interaction
        self.plot_widget.scene().sigMouseClicked.connect(self.on_timeline_click)
        
        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        
        # Mouse moved event for crosshair
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, 
                                    rateLimit=60, slot=self.on_mouse_moved)
        
        layout.addWidget(self.plot_widget)
        
        self.bodypart_checkboxes = {}
    
    def on_zoom_toggle(self, state):
        """Toggle auto-zoom on/off"""
        self.zoom_enabled = (state == Qt.Checked)
        if self.zoom_enabled and self.pd_dataframe is not None:
            self.update_zoom()
    
    def reset_zoom(self):
        """Reset zoom to show entire timeline"""
        if self.pd_dataframe is not None:
            self.plot_widget.setXRange(0, len(self.pd_dataframe), padding=0.02)
            y_data = self.pd_dataframe[f'{self.current_bodypart} {self.axis}']
            y_min, y_max = y_data.min(), y_data.max()
            self.plot_widget.setYRange(y_min, y_max, padding=0.1)  # Y axis already inverted
    
    def on_mouse_moved(self, evt):
        """Update crosshair position on mouse move"""
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
    
    def on_timeline_click(self, event):
        """Handle click on timeline to jump to frame"""
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                frame = int(mousePoint.x())
                if 0 <= frame < len(self.pd_dataframe):
                    self.frame_clicked.emit(frame)
    
    def set_data(self, pd_dataframe, bodyparts, selected_bodyparts, axis='y'):
        """Set the data for the timeline"""
        self.pd_dataframe = pd_dataframe
        self.bodyparts = bodyparts
        self.selected_bodyparts = selected_bodyparts
        self.axis = axis
        
        # Clear existing bodypart checkboxes
        for checkbox in self.bodypart_checkboxes.values():
            checkbox.deleteLater()
        self.bodypart_checkboxes.clear()
        
        # Create compact checkboxes for each bodypart
        colors = self._generate_colors(len(selected_bodyparts))
        for i, bp in enumerate(selected_bodyparts):
            checkbox = QCheckBox(bp)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, b=bp: self.toggle_bodypart(b, state))
            
            # Color indicator - compact style
            color = colors[i]
            checkbox.setStyleSheet(f"QCheckBox {{ color: rgb({color[0]}, {color[1]}, {color[2]}); font-weight: bold; font-size: 10px; }}")
            
            self.bodypart_checkboxes[bp] = checkbox
            self.legend_layout.addWidget(checkbox)
    
    def set_current_bodypart(self, bodypart):
        """Set the primary bodypart to display"""
        self.current_bodypart = bodypart
    
    def set_detection_data(self, t_pred, start_pred, end_pred, bodypart_list, confirmed, slip_fall_val=None):
        """Set footfall detection data"""
        self.t_pred = t_pred
        self.start_pred = start_pred
        self.end_pred = end_pred
        self.bodypart_list = bodypart_list
        self.confirmed = confirmed
        self.slip_fall_val = slip_fall_val if slip_fall_val else [''] * len(t_pred)
    
    def set_current_frame(self, frame, likelihood_threshold=0.1):
        """Update the current frame marker"""
        self.n_current_frame = frame
        self.likelihood_threshold = likelihood_threshold
        
        # Update crosshair position
        if self.pd_dataframe is not None and 0 <= frame < len(self.pd_dataframe):
            y_val = self.pd_dataframe[f'{self.current_bodypart} {self.axis}'].iloc[frame]
            self.vLine.setPos(frame)
            self.hLine.setPos(y_val)
            
            if self.zoom_enabled:
                self.update_zoom()
    
    def update_zoom(self):
        """Update zoom to focus on current frame"""
        if self.pd_dataframe is None or self.current_bodypart is None:
            return
        
        x_max = len(self.pd_dataframe)
        frame = self.n_current_frame
        
        # X-axis zoom
        if frame <= 300:
            self.plot_widget.setXRange(0, 600, padding=0)
        elif frame >= x_max - 300:
            self.plot_widget.setXRange(x_max - 600, x_max, padding=0)
        else:
            self.plot_widget.setXRange(frame - 300, frame + 300, padding=0)
        
        # Y-axis zoom around current point
        y_current = self.pd_dataframe[f'{self.current_bodypart} {self.axis}'].iloc[frame]
        self.plot_widget.setYRange(y_current - 100, y_current + 100, padding=0)  # Y axis already inverted
    
    def update_plot(self):
        """Redraw the entire plot with current data"""
        if self.pd_dataframe is None or self.current_bodypart is None:
            return
        
        # Clear previous plot items
        self.plot_widget.clear()
        
        # Re-add crosshair
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        
        self.bodypart_curves.clear()
        self.marker_items.clear()
        
        frames = self.pd_dataframe['bodyparts coords'].values
        
        # Plot all selected bodyparts
        colors = self._generate_colors(len(self.selected_bodyparts))
        for i, bp in enumerate(self.selected_bodyparts):
            if bp in self.bodypart_checkboxes and self.bodypart_checkboxes[bp].isChecked():
                y_data = self.pd_dataframe[f'{bp} {self.axis}'].values
                
                # Determine if this is the current bodypart
                if bp == self.current_bodypart:
                    pen = pg.mkPen(color=colors[i], width=2)
                else:
                    pen = pg.mkPen(color=colors[i], width=1, style=Qt.DashLine)
                
                curve = self.plot_widget.plot(frames, y_data, pen=pen, name=bp)
                self.bodypart_curves[bp] = curve
        
        # Plot low likelihood points in gray for current bodypart
        low_likelihood_mask = self.pd_dataframe[f'{self.current_bodypart} likelihood'] < self.likelihood_threshold
        if low_likelihood_mask.any():
            low_frames = frames[low_likelihood_mask]
            low_y = self.pd_dataframe[f'{self.current_bodypart} {self.axis}'].values[low_likelihood_mask]
            scatter = pg.ScatterPlotItem(low_frames, low_y, size=3, 
                                        brush=pg.mkBrush(200, 200, 200, 150))
            self.plot_widget.addItem(scatter)
        
        # Plot footfall markers
        self._plot_footfall_markers()
        
        # Highlight current frame
        if 0 <= self.n_current_frame < len(self.pd_dataframe):
            y_current = self.pd_dataframe[f'{self.current_bodypart} {self.axis}'].iloc[self.n_current_frame]
            current_marker = pg.ScatterPlotItem([self.n_current_frame], [y_current], 
                                                size=15, brush=pg.mkBrush(255, 0, 0), 
                                                pen=pg.mkPen('darkred', width=2), symbol='o')
            self.plot_widget.addItem(current_marker)
        
        # Update zoom if enabled
        if self.zoom_enabled:
            self.update_zoom()
        else:
            self.reset_zoom()
    
    def _plot_footfall_markers(self):
        """Plot footfall detection markers"""
        count_confirmed = 0
        
        for i, t in enumerate(self.t_pred):
            bp = self.bodypart_list[i]
            if bp != bp or bp is None:  # Check for NaN
                bp = self.current_bodypart
            
            y_val = self.pd_dataframe[f'{bp} {self.axis}'].iloc[t]
            
            # Determine marker color based on confirmation and slip/fall type
            if self.confirmed[i]:
                count_confirmed += 1
                
                # Check if it's a slip or fall
                if i < len(self.slip_fall_val):
                    if self.slip_fall_val[i] == 'slip':
                        brush = pg.mkBrush(255, 165, 0)  # Orange for slip
                        symbol = 't1'  # Triangle down
                    elif self.slip_fall_val[i] == 'fall':
                        brush = pg.mkBrush(255, 0, 0)  # Red for fall
                        symbol = 't1'  # Triangle down
                    else:
                        brush = pg.mkBrush(0, 200, 0)  # Green for confirmed, unmarked
                        symbol = 't1'  # Triangle down
                else:
                    brush = pg.mkBrush(0, 200, 0)  # Green for confirmed
                    symbol = 't1'  # Triangle down
                
                # Plot confirmed footfall marker
                marker = pg.ScatterPlotItem([t], [y_val], size=12, brush=brush, 
                                            pen=pg.mkPen('darkgreen', width=1), symbol=symbol)
                self.plot_widget.addItem(marker)
                
                # Add number label (smaller font)
                text = pg.TextItem(str(count_confirmed), color='g', anchor=(0.5, 1.5))
                text.setFont(pg.QtGui.QFont("Arial", 8))
                text.setPos(t, y_val)
                self.plot_widget.addItem(text)
                
                # Plot start marker if exists
                if i < len(self.start_pred) and not np.isnan(self.start_pred[i]) and bp == self.current_bodypart:
                    start_frame = int(self.start_pred[i])
                    y_start = self.pd_dataframe[f'{self.current_bodypart} {self.axis}'].iloc[start_frame]
                    
                    start_marker = pg.ScatterPlotItem([start_frame], [y_start], size=10, 
                                                      brush=pg.mkBrush(255, 140, 0), 
                                                      pen=pg.mkPen('darkorange', width=1), symbol='t')
                    self.plot_widget.addItem(start_marker)
                    
                    # Draw line from start to peak
                    line = pg.PlotDataItem([start_frame, t], [y_start, y_val], 
                                          pen=pg.mkPen(color='orange', width=2, style=Qt.DashLine))
                    self.plot_widget.addItem(line)
                
                # Plot end marker if exists
                if i < len(self.end_pred) and not np.isnan(self.end_pred[i]) and bp == self.current_bodypart:
                    end_frame = int(self.end_pred[i])
                    y_end = self.pd_dataframe[f'{self.current_bodypart} {self.axis}'].iloc[end_frame]
                    
                    end_marker = pg.ScatterPlotItem([end_frame], [y_end], size=10, 
                                                    brush=pg.mkBrush(147, 112, 219), 
                                                    pen=pg.mkPen('indigo', width=1), symbol='t1')
                    self.plot_widget.addItem(end_marker)
                    
                    # Draw line from peak to end
                    line = pg.PlotDataItem([t, end_frame], [y_val, y_end], 
                                          pen=pg.mkPen(color='purple', width=2, style=Qt.DashLine))
                    self.plot_widget.addItem(line)
            else:
                # Unconfirmed footfall in gray
                marker = pg.ScatterPlotItem([t], [y_val], size=10, 
                                            brush=pg.mkBrush(128, 128, 128, 100), 
                                            pen=pg.mkPen('gray', width=1), symbol='t1')
                self.plot_widget.addItem(marker)
    
    def toggle_bodypart(self, bodypart, state):
        """Toggle visibility of a bodypart trace"""
        if bodypart in self.bodypart_curves:
            self.bodypart_curves[bodypart].setVisible(state == Qt.Checked)
        else:
            # If not visible, redraw
            self.update_plot()
    
    def _generate_colors(self, n):
        """Generate n distinct colors for bodyparts"""
        colors = []
        for i in range(n):
            hue = int(360 * i / n)
            color = QColor.fromHsv(hue, 200, 200)
            colors.append((color.red(), color.green(), color.blue()))
        return colors
    
    def add_legend(self):
        """Add interactive legend to the plot"""
        legend = self.plot_widget.addLegend()
        legend.setOffset((10, 10))

