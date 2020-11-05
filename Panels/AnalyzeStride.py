import wx
from wx.lib.stattext import GenStaticText as StaticText
from Functions import SlipFunctions, ConfigFunctions, KinematicsFunctions
import os
import yaml
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import numpy as np  # remove later; for testing
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=ResourceWarning)


TEST = False


class AnalyzeStridePanel(wx.Panel):

    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # load parameters to set dimension of frames and graphs
        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width = configs['window_width']
        self.window_height = configs['window_height']
        self.frame_rate = configs['frame_rate']
        self.treadmill_speed = configs['treadmill_speed']
        self.rolling_window = configs['rolling_window']
        self.change_threshold = configs['change_threshold']
        self.pixels_per_cm = configs['pixels_per_cm']
        self.stance_threshold = configs['stance_threshold']
        self.treadmill_y = configs['treadmill_y']
        self.cutoff_f = configs['lowpass_filter_cutoff']

        self.first_sizer_widgets = []
        self.second_sizer_widgets = []
        self.has_imported_file = False
        self.dirname = os.getcwd()

        self.first_sizer = wx.GridBagSizer(0, 0)

        self.header = wx.StaticText(
            self, -1, "Kinematics Analysis", size=(500, 100))
        font = wx.Font(20, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.header.SetFont(font)
        self.first_sizer.Add(self.header, pos=(0, 0), span=(
            2, 5), flag=wx.LEFT | wx.TOP, border=25)

        self.instructions = wx.StaticText(
            self, -1, "Load the csv output from DeepLabCut and automatically extract stride and kinematics parameters.")
        self.first_sizer.Add(self.instructions, pos=(
            2, 0), span=(1, 3), flag=wx.LEFT | wx.TOP, border=25)

        self.FirstPage()

    def FirstPage(self):

        self.import_csv_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        self.first_sizer.Add(self.import_csv_button,
                             pos=(6, 0), flag=wx.LEFT, border=25)
        self.first_sizer_widgets.append(self.import_csv_button)

        self.import_csv_text = wx.StaticText(self, label="Import a csv file for stride extraction and calculate kinematics parameters. "
                                             + "\nThe file should be DeepLabCut output (or of a similar format) to ensure proper parsing!")
        self.first_sizer.Add(self.import_csv_text, pos=(
            6, 1), flag=wx.LEFT, border=25)
        self.first_sizer_widgets.append(self.import_csv_text)

        self.pred_text = wx.StaticText(
            self, label="\nThe chosen algorithm will automatically extract strides using csv input.")
        self.first_sizer.Add(self.pred_text, pos=(7, 1),
                             flag=wx.LEFT, border=25)
        self.first_sizer_widgets.append(self.pred_text)

        self.method_label = wx.StaticText(
            self, label='Select algorithm for slip prediction:')
        self.first_sizer.Add(self.method_label, pos=(
            8, 1), flag=wx.LEFT | wx.TOP, border=25)
        self.first_sizer_widgets.append(self.method_label)
        self.method_label.Hide()

        methods = ['Rate of change', 'Threshold']
        self.method_choices = wx.ComboBox(self, choices=methods)
        self.first_sizer.Add(self.method_choices, pos=(
            8, 2), flag=wx.LEFT | wx.TOP, border=25)
        self.first_sizer_widgets.append(self.method_choices)
        self.method_choices.Bind(wx.EVT_COMBOBOX, self.OnMethod)
        self.method_choices.Hide()

        self.bodypart_label = wx.StaticText(
            self, label='Labelled bodypart to use for stride extraction:')
        self.first_sizer.Add(self.bodypart_label, pos=(
            9, 1), flag=wx.LEFT | wx.TOP, border=25)
        self.first_sizer_widgets.append(self.bodypart_label)
        self.bodypart_label.Hide()

        self.bodyparts = []
        self.bodypart_choices = wx.ComboBox(self, choices=self.bodyparts)
        self.first_sizer.Add(self.bodypart_choices, pos=(
            9, 2), flag=wx.LEFT | wx.TOP, border=25)
        self.bodypart_choices.Hide()

        self.save_pred_button = wx.Button(
            self, id=wx.ID_ANY, label="Save extracted stride start and end frame numbers")
        self.first_sizer.Add(self.save_pred_button, pos=(
            10, 1), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=25)
        self.first_sizer_widgets.append(self.save_pred_button)
        self.save_pred_button.Hide()

        self.import_new_csv_button = wx.Button(
            self, id=wx.ID_ANY, label="Import a different file")
        self.first_sizer.Add(self.import_new_csv_button, pos=(
            10, 2), flag=wx.TOP | wx.BOTTOM, border=25)
        self.first_sizer_widgets.append(self.import_new_csv_button)
        self.import_new_csv_button.Hide()

        self.extract_parameters_button = wx.Button(
            self, id=wx.ID_ANY, label='Extract parameters')
        self.first_sizer.Add(self.extract_parameters_button, pos=(
            11, 1), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=25)
        self.first_sizer_widgets.append(self.extract_parameters_button)
        self.extract_parameters_button.Hide()

        self.save_parameters_button = wx.Button(
            self, id=wx.ID_ANY, label='Save extracted parameters')
        self.first_sizer.Add(self.save_parameters_button, pos=(
            11, 2), flag=wx.TOP | wx.BOTTOM, border=25)
        self.first_sizer_widgets.append(self.save_parameters_button)
        self.save_parameters_button.Hide()

    #     self.import_video_button = wx.Button(self, id=wx.ID_ANY, label="Import")

    #     self.first_sizer.Add(self.import_video_button, pos = (11, 0), flag = wx.LEFT | wx.TOP, border = 25)
    #     self.first_sizer_widgets.append(self.import_video_button)
    #     self.import_video_button.Hide()

    #     self.import_video_text = wx.StaticText(self, label = "Import the corresponding video file for validation. ")
    #     self.first_sizer.Add(self.import_video_text, pos=(11, 1), flag = wx.LEFT | wx.TOP, border = 25)
    #     self.first_sizer_widgets.append(self.import_video_text)
    #     self.import_video_text.Hide()

    #     self.validate_button = wx.Button(self, id=wx.ID_ANY, label="Validate")

    #     self.first_sizer.Add(self.validate_button, pos = (12, 1), flag = wx.TOP | wx.LEFT, border = 25)
    #     self.first_sizer_widgets.append(self.validate_button)
    #     self.validate_button.Hide()

    #     self.import_new_video_button = wx.Button(self, id=wx.ID_ANY, label="Import a different video")
    #     self.first_sizer.Add(self.import_new_video_button, pos = (12, 2), flag = wx.TOP, border = 25)
    #     self.first_sizer_widgets.append(self.import_new_video_button)
    #     self.import_new_video_button.Hide()

        self.save_pred_button.Bind(wx.EVT_BUTTON, self.SavePredFunc)
        self.import_csv_button.Bind(wx.EVT_BUTTON, self.ImportKinematicsCSV)
        self.import_new_csv_button.Bind(wx.EVT_BUTTON, self.ImportKinematicsCSV)
        self.extract_parameters_button.Bind(wx.EVT_BUTTON, self.ExtractParameters)
        self.save_parameters_button.Bind(wx.EVT_BUTTON, self.SaveParametersFunc)
    #     # self.import_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)
    #     # self.validate_button.Bind(wx.EVT_BUTTON, self.DisplaySecondPage)
    #     # self.import_new_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)

    #     if TEST is True:
    #         self.filename, self.df, self.bodyparts, self.video, self.video_name = KinematicsFunctions.test(TEST)

        self.SetSizer(self.first_sizer)

        self.GetParent().Layout()

    def ImportKinematicsCSV(self, e):
        if not TEST:
            import_dialog = wx.FileDialog(
                self, 'Choose a file', self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', wx.FD_OPEN)

            if import_dialog.ShowModal() == wx.ID_OK:
                self.csv_dirname = import_dialog.GetDirectory()
                self.filename = os.path.join(
                    self.csv_dirname, import_dialog.GetFilename())
                self.df, self.filename = KinematicsFunctions.read_file(
                    self.filename)
                self.df, self.bodyparts = KinematicsFunctions.fix_column_names(
                    self.df)
                # adjust x axis location according treadmill movement speed (pixel per frame)
                self.df = KinematicsFunctions.treadmill_correction(
                    self.df, self.bodyparts, self.treadmill_speed)
        try:
            if self.df is not None:
                self.has_imported_file = True
                self.import_csv_text.SetLabel(
                    f"File imported! \n\n{self.filename}\n")
                self.GetParent().Layout()

                self.method_label.Show()

                # self.method_choices.SetSelection(0)
                self.method_selection = self.method_choices.GetValue()
                self.method_choices.Show()
                self.GetParent().Layout()

                self.bodypart_label.Show()
                self.GetParent().Layout()
                self.extract_parameters_button.Hide()
                self.save_parameters_button.Hide()
                # update widget with values from imported file
                bodypart_choices = wx.ComboBox(self, choices=self.bodyparts)
                self.first_sizer.Replace(
                    self.bodypart_choices, bodypart_choices)
                self.bodypart_choices = bodypart_choices

                # self.bodypart_choices.SetSelection(0)
                self.bodypart = self.bodypart_choices.GetValue()
                self.first_sizer_widgets.append(self.bodypart_choices)
                self.bodypart_choices.Bind(wx.EVT_COMBOBOX, self.OnBodypart)
                self.bodypart_choices.Show()
                self.GetParent().Layout()

                self.ExtractStrides(self)
                self.GetParent().Layout()

        except AttributeError:
            # user cancelled file import in pop up
            pass

    def OnMethod(self, e):

        self.selected_bodyparts = self.bodypart_choices.GetValue()
        self.method_selection = self.method_choices.GetValue()
        try:
            self.ExtractStrides(self)
            self.GetParent().Layout()
        except:
            # awaiting bodypart selection
            self.pred_text.SetLabel(
                "\nStride extraction using the selected method and bodypart failed! \n(No suprathreshold value was found, or the rate of change was too even.)\n")
            # print(self.start_times, self.end_times, self.durations)
            self.GetParent().Layout()
            pass

    def OnBodypart(self, e):

        self.selected_bodyparts = self.bodypart_choices.GetValue()
        try:
            self.ExtractStrides(self)
            self.GetParent().Layout()
        except:
            # awaiting method selection
            # print(self.start_times, self.end_times, self.durations)

            self.pred_text.SetLabel(
                "\nStride extraction using the selected method and bodypart failed! \n(No suprathreshold value was found, or the rate of change was too even.)\n")
            self.GetParent().Layout()
            pass

    def ExtractStrides(self, e):

        self.import_csv_button.Disable()

        self.start_times, self.end_times, self.durations = [], [], []

        self.start_times, self.end_times, self.durations = KinematicsFunctions.find_strides(
            self.df, self.selected_bodyparts, method=self.method_selection, rolling_window=self.rolling_window,
            change_threshold=self.change_threshold)

        self.n_strides = len(self.durations)

        self.pred_text.SetLabel(
            f"\nThe algorithm extracted {self.n_strides} strides with an average duration of {np.mean(self.durations):.2f} frames.\n")

        self.save_pred_button.Show()
        self.import_new_csv_button.Show()
        # self.import_video_button.Show()
        # self.import_video_text.Show()
        self.extract_parameters_button.Show()
        self.GetParent().Layout()

    def ExtractParameters(self, e):
        self.pred_text.SetLabel(
            f"\nWorking hard to extract 39 kinematic parameters from {self.n_strides} strides...\n")
        self.GetParent().Layout()
        
        self.parameters = KinematicsFunctions.extract_parameters(
            self.frame_rate, self.pixels_per_cm, self.df, self.stance_threshold, self.treadmill_y, self.cutoff_f, self.start_times, self.end_times)
        self.save_parameters_button.Show()

        self.pred_text.SetLabel(
            f"\nExtracted 39 kinematic parameters from {self.n_strides} strides! Ready to export parameters.\n")

        self.GetParent().Layout()

    def SavePredFunc(self, e):

        with wx.FileDialog(self, 'Save current prediction as... ',
                           self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*',
                           wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as save_pred_dialog:

            if save_pred_dialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = save_pred_dialog.GetPath()

            try:
                KinematicsFunctions.make_output(
                    pathname, self.start_times, self.end_times, self.durations)

            except IOError:
                wx.LogError(
                    f"Cannot save current data in file {pathname}. Try another location or filename?")

    def SaveParametersFunc(self, e):

        with wx.FileDialog(self, 'Save extracted parameters as... ',
                           self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*',
                           wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as save_pred_dialog:

            if save_pred_dialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = save_pred_dialog.GetPath()

            try:
                KinematicsFunctions.make_parameters_output(
                    pathname, self.parameters)

            except IOError:
                wx.LogError(
                    f"Cannot save current data in file {pathname}. Try another location or filename?")
