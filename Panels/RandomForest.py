import wx
from wx.lib.stattext import GenStaticText as StaticText
from Functions import ConfigFunctions, DataAnalysisFunctions
import os
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import numpy as np  # remove later; for testing
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=ResourceWarning)


class RandomForestPanel(wx.Panel):

    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # load parameters to set dimension of frames and graphs
        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width = configs['window_width']
        self.window_height = configs['window_height']
        self.dirname = os.getcwd()
        self.folder1 = ''
        self.folder2 = ''

        self.sizer = wx.GridBagSizer(0, 0)

        self.header = wx.StaticText(
            self, -1, "Random Forest Classification", size=(500, 100))
        font = wx.Font(20, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.header.SetFont(font)
        self.sizer.Add(
            self.header,
            pos=(0, 0), span=(2, 5), flag=wx.LEFT | wx.TOP, border=25)
        instructions = "Classify animals using extracted gait kinematic parameters and identify important parameters."
        self.instructions = wx.StaticText(
            self, -1, instructions, size=(self.window_width, 50))
        font = wx.Font(15, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.instructions.SetFont(font)
        self.sizer.Add(
            self.instructions,
            pos=(2, 0), span=(1, 6), flag=wx.LEFT, border=25)

        self.data_widgets = []

        #################################
        self.select_group1_button = wx.Button(
            self, id=wx.ID_ANY, label="Group 1")
        self.sizer.Add(
            self.select_group1_button,
            pos=(3, 0), flag=wx.LEFT | wx.TOP, border=25)
        self.data_widgets.append(self.select_group1_button)
        self.Bind(
            wx.EVT_BUTTON,
            lambda event, group_no=0: self.SelectGroupFolder(event, group_no),
            self.select_group1_button)

        self.select_group1_text = wx.StaticText(
            self,
            label="\nPlease select kinematic data folder of the first group.")
        self.sizer.Add(
            self.select_group1_text,
            pos=(3, 1), flag=wx.LEFT | wx.TOP, border=25)
        self.data_widgets.append(self.select_group1_text)
        #################################
        self.select_group2_button = wx.Button(
            self, id=wx.ID_ANY, label="Group 2")
        self.sizer.Add(
            self.select_group2_button,
            pos=(4, 0), flag=wx.LEFT | wx.TOP, border=25)
        self.data_widgets.append(self.select_group2_button)
        self.Bind(
            wx.EVT_BUTTON,
            lambda event, group_no=1: self.SelectGroupFolder(event, group_no),
            self.select_group2_button)

        self.select_group2_text = wx.StaticText(
            self,
            label="\nPlease select kinematic data folder of the second group.")
        self.sizer.Add(
            self.select_group2_text,
            pos=(4, 1), flag=wx.LEFT | wx.TOP, border=25)
        self.data_widgets.append(self.select_group2_text)
        #################################
        self.select_output_folder_button = wx.Button(
            self, id=wx.ID_ANY, label='Select output path')
        self.sizer.Add(
            self.select_output_folder_button,
            pos=(7, 1), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=25)
        self.data_widgets.append(self.select_output_folder_button)
        self.select_output_folder_button.Bind(
            wx.EVT_BUTTON, self.SelectOutputFolder)
        self.select_output_folder_button.Show()

        self.SetSizer(self.sizer)
        self.GetParent().Layout()

    def SelectGroupFolder(self, e, group_no):

        import_dialog = wx.DirDialog(
            self,
            f'Select the folder that contains gait kinematic parameters of your group {group_no+1}',
            self.dirname, style=wx.DD_DEFAULT_STYLE)
        # Show the dialog and retrieve the user response.
        if import_dialog.ShowModal() == wx.ID_OK:
            # load directory
            path = import_dialog.GetPath()
            if group_no == 0:
                self.folder1 = path
                self.select_group1_text.SetLabel(
                    f'Data folder of the first group: \n{path}')

                self.group1_name_input = wx.TextCtrl(self, value='')
                self.sizer.Add(
                    self.group1_name_input,
                    pos=(3, 2), flag=wx.LEFT | wx.TOP, border=25)
                self.data_widgets.append(self.group1_name_input)
                self.group1_name_input.Show()

                self.group1_name_input_button = wx.Button(
                    self, id=wx.ID_ANY, label="Confirm group name")
                self.Bind(
                    wx.EVT_BUTTON,
                    lambda event,
                    group_no=0: self.OnGroupNameInput(event, group_no),
                    self.group1_name_input_button)
                self.sizer.Add(
                    self.group1_name_input_button,
                    pos=(3, 3), flag=wx.LEFT | wx.TOP, border=25)
                self.data_widgets.append(self.group1_name_input_button)
                self.group1_name_input_button.Show()

            elif group_no == 1:
                self.folder2 = path
                self.select_group2_text.SetLabel(
                    f'Data folder of the second group: \n{path}')

                self.group2_name_input = wx.TextCtrl(self, value='')
                self.sizer.Add(
                    self.group2_name_input,
                    pos=(4, 2), flag=wx.LEFT | wx.TOP, border=25)
                self.data_widgets.append(self.group2_name_input)
                self.group2_name_input.Show()

                self.group2_name_input_button = wx.Button(
                    self, id=wx.ID_ANY, label="Confirm group name")
                self.Bind(
                    wx.EVT_BUTTON,
                    lambda event,
                    group_no=1: self.OnGroupNameInput(event, group_no),
                    self.group2_name_input_button)
                self.sizer.Add(
                    self.group2_name_input_button,
                    pos=(4, 3), flag=wx.LEFT | wx.TOP, border=25)
                self.data_widgets.append(self.group2_name_input_button)
                self.group2_name_input_button.Show()
        else:
            if group_no == 0:
                self.folder1 = ''
                self.select_group1_text.SetLabel(
                    '\nPlease select kinematic data folder of the first group.')
            elif group_no == 1:
                self.folder2 = ''
                self.select_group2_text.SetLabel(
                    '\nPlease select kinematic data folder of the second group.')

        self.GetParent().Layout()

        # Destroy the dialog.
        import_dialog.Destroy()

    def OnGroupNameInput(self, e, group_no):

        if group_no == 0:
            self.group_name_1 = str(self.group1_name_input.GetValue())
            self.select_group1_text.SetLabel(
                f'\nGroup 1: {self.group_name_1}, data located '
                'at {self.folder1}')
        elif group_no == 1:
            self.group_name_2 = str(self.group2_name_input.GetValue())
            self.select_group2_text.SetLabel(
                f'\nGroup 2: {self.group_name_2}, data located '
                'at {self.folder2}')

        self.GetParent().Layout()

    def SelectOutputFolder(self, e):

        export_dialog = wx.DirDialog(
            self, 'Save analysis results to... ',
            self.dirname, style=wx.DD_DEFAULT_STYLE)

        if export_dialog.ShowModal() == wx.ID_OK:
            self.output_path = export_dialog.GetPath()

            self.output_path_text = wx.StaticText(
                self,
                label="\nAnalysis results will be saved to "
                f"{self.output_path}")
            self.sizer.Add(
                self.output_path_text,
                pos=(7, 2), flag=wx.LEFT | wx.TOP, border=25)
            self.data_widgets.append(self.output_path_text)

            self.random_forest_button = wx.Button(
                self, id=wx.ID_ANY,
                label='Run random forest classification')
            self.sizer.Add(
                self.random_forest_button,
                pos=(8, 1), flag=wx.LEFT | wx.TOP, border=25)
            self.data_widgets.append(self.random_forest_button)
            self.random_forest_button.Bind(wx.EVT_BUTTON, self.RandomForest)
            self.random_forest_button.Show()
        else:
            self.output_path = ''

        export_dialog.Destroy()
        self.GetParent().Layout()

    def RandomForest(self, e):

        group_1_file_list = DataAnalysisFunctions.find_files(self.folder1)
        group_2_file_list = DataAnalysisFunctions.find_files(self.folder2)
        file_lists = [group_1_file_list, group_2_file_list]
        group_names = [self.group_name_1, self.group_name_2]
        combined_df = DataAnalysisFunctions.combine_files(
            file_lists, group_names, self.output_path, 'strides')

        DataAnalysisFunctions.random_forest(combined_df, self.output_path)

        self.random_forest_result_text = wx.StaticText(
            self, label="\nAnalysis completed!")
        self.sizer.Add(
            self.random_forest_result_text,
            pos=(8, 2), flag=wx.LEFT | wx.TOP, border=25)
        self.data_widgets.append(self.random_forest_result_text)
        self.GetParent().Layout()
