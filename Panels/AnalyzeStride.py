import wx
from Functions import ConfigFunctions, KinematicsFunctions
import os
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=ResourceWarning)


class AnalyzeStridePanel(wx.Panel):

    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width = configs['window_width']
        self.window_height = configs['window_height']
        self.frame_rate = configs['frame_rate']
        self.cutoff_f = configs['lowpass_filter_cutoff']
        self.px_to_cm_speed_ratio = configs['px_to_cm_speed_ratio']
        self.analysis_type = "Treadmill"
        self.extra_bodyparts = False
        self.pixels_per_cm = configs['pixels_per_cm']
        if configs['cm_speed'] == '':
            self.cm_speed = None
        else:   
            self.cm_speed = configs['cm_speed']
        self.dragging_filter = bool(configs['dragging_filter'])
        self.no_outlier_filter = bool(configs['no_outlier_filter'])
    
        self.stride_widgets = []
        self.has_input_path = False
        self.has_imported_file = False
        self.dirname = os.getcwd()

        self.sizer = wx.GridBagSizer(0, 0)

        self.header = wx.StaticText(
            self, -1, "Kinematic Analysis", size=(500, 100))
        font = wx.Font(20, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.header.SetFont(font)
        self.sizer.Add(self.header, pos=(0, 0), span=(
            2, 4), flag=wx.LEFT | wx.TOP, border=25)

        self.instructions = wx.StaticText(
            self, -1, "Load the csv file of bodypart coordinates and extract kinematic parameters.", size=(self.window_width,50))
        font = wx.Font(15,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.instructions.SetFont(font)
        self.sizer.Add(self.instructions, pos=(
            2, 0), span=(1, 4), flag=wx.LEFT, border=25)

        self.Stride_UI()


    def Stride_UI(self):

        analysis_types = ["Treadmill", "Spontaneous walking"]
        self.analysis_type_rbox = wx.RadioBox(self, id=wx.ID_ANY, 
            label="Kinematic experimental set-up", choices=analysis_types,
            majorDimension=1, style=wx.RA_SPECIFY_ROWS)

        self.analysis_type_rbox.Bind(wx.EVT_RADIOBOX, self.onAnalysisTypeRadioBox)
        self.sizer.Add(self.analysis_type_rbox, pos=(4, 1), flag=wx.LEFT|wx.BOTTOM, border=25)
        self.stride_widgets.append(self.analysis_type_rbox)

        self.extra_bodyparts_checkbox = wx.CheckBox(self, id=wx.ID_ANY,
            label="Include forelimb, trunk, and tail analysis")
        self.extra_bodyparts_checkbox.Bind(wx.EVT_CHECKBOX, self.onExtraBodypartsCheckbox)
        self.sizer.Add(self.extra_bodyparts_checkbox, pos=(5, 1), flag=wx.LEFT|wx.BOTTOM, border=25)
        self.stride_widgets.append(self.extra_bodyparts_checkbox)

        self.import_csv_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        self.sizer.Add(self.import_csv_button,
                            pos=(6, 0), flag=wx.LEFT, border=25)
        self.stride_widgets.append(self.import_csv_button)

        self.import_csv_text = wx.StaticText(self, label="Import a csv file for stride extraction and calculate kinematic parameters. "
                                            + "\nThe file should be DeepLabCut output (or of a similar format) to ensure proper parsing!")
        self.sizer.Add(self.import_csv_text, pos=(
            6, 1), flag=wx.LEFT, border=25)
        self.stride_widgets.append(self.import_csv_text)
        self.import_csv_button.Bind(wx.EVT_BUTTON, self.ImportKinematicsCSV)
        
        self.import_folder_button = wx.Button(self, id=wx.ID_ANY, label="Bulk import")
        self.sizer.Add(self.import_folder_button, pos=(7, 0), flag=wx.LEFT | wx.TOP, border=25)
        self.stride_widgets.append(self.import_folder_button)

        self.import_folder_text = wx.StaticText(self, label="Import all csv files from a folder for stride extraction and calculate kinematics parameters. "
                                            + "\n\nThe files should be DeepLabCut output (or of a similar format) to ensure proper parsing!"
                                            + "\n\nIf using a treadmill set-up, all recordings should have the same treadmill speed (if using manual input) or "
                                            + "\nfollow the provided pixel to centimeter speed ratio (if using automatic speed detection)."
                                            + "\n\nIf using a spontaneous overground walking set-up, all recordings should have the same px-cm length ratio,"
                                            + "\ni.e., recording distance and camera angle, among other details of the set-up.")
        self.sizer.Add(self.import_folder_text, pos=(7, 1), flag=wx.LEFT | wx.TOP, border=25)
        self.stride_widgets.append(self.import_folder_text)
        self.import_folder_button.Bind(wx.EVT_BUTTON, self.BulkImportKinematicsCSV)

        self.SetSizer(self.sizer)
        self.GetParent().Layout()


    def onAnalysisTypeRadioBox(self, e):

        # print(self.analysis_type_rbox.GetStringSelection())
        self.analysis_type = self.analysis_type_rbox.GetStringSelection()

        if self.analysis_type == "Treadmill":

            try:
                self.px_cm_length_ratio_input.Destroy()
                self.px_cm_length_ratio_input_button.Destroy()
            except:
                pass

            try:
                self.method_label.Destroy()
                self.method_choices.Destroy()
                self.method_note_text.Destroy()
            except:
                pass

            self.method_label = wx.StaticText(
                self, label='Select method for px-to-cm speed conversion:')
            self.sizer.Add(self.method_label, pos=(
                8, 1), flag=wx.LEFT | wx.TOP, border=25)
            self.stride_widgets.append(self.method_label)
            self.method_label.Show()

            methods = ['Semi-automated', 'Fully automated']
            self.method_choices = wx.ComboBox(self, choices=methods)
            self.sizer.Add(self.method_choices, pos=(
                8, 2), flag=wx.LEFT | wx.TOP, border=25)
            self.stride_widgets.append(self.method_choices)
            self.method_choices.Bind(wx.EVT_COMBOBOX, self.OnMethod)
            self.method_choices.Show()

            self.method_note_text = wx.StaticText(
                self, label="\nNote: please use semi-automated function when running kinematic analysis on a new experimental set-up."
                + "\nThe pixel/frame speed and the corresponding px-to-cm speed ratio will be estimated and displayed. "
                + "\nThe ratio can then be re-used for fully automated analysis, or for finding a more precise relationship, "
                + "\ne.g., through a regression between the px-to-cm speed ratio at different speeds.")
            self.sizer.Add(self.method_note_text, pos=(9, 1),
                                flag=wx.LEFT | wx.TOP, border=25)
            self.stride_widgets.append(self.method_note_text)
            self.method_note_text.Show()

        elif self.analysis_type == "Spontaneous walking":
            
            try:
                self.px_cm_ratio_input.Destroy()
                self.px_cm_ratio_input_button.Destroy()
            except:
                pass
            try:
                self.method_label.Destroy()
                self.method_choices.Destroy()
                self.method_note_text.Destroy()
            except:
                pass

            self.method_label = wx.StaticText(
                self, label='Select method for px-to-cm length conversion:')
            self.sizer.Add(self.method_label, pos=(
                8, 1), flag=wx.LEFT | wx.TOP, border=25)
            self.stride_widgets.append(self.method_label)
            self.method_label.Show()

            methods = ['Manual input']
            self.method_choices = wx.ComboBox(self, choices=methods)
            self.sizer.Add(self.method_choices, pos=(
                8, 2), flag=wx.LEFT | wx.TOP, border=25)
            self.stride_widgets.append(self.method_choices)
            self.method_choices.Bind(wx.EVT_COMBOBOX, self.OnMethod)
            self.method_choices.Show()

            self.method_note_text = wx.StaticText(
                self, 
                label=f"Using a spontaneous overground walking set-up. Parameters will be extracted automatically."
                        + f"\n\nCurrently using a length conversion ratio of {self.pixels_per_cm} pixels / cm (needs to be obtained manually).")
            self.sizer.Add(self.method_note_text, pos=(9, 1),
                                flag=wx.LEFT | wx.TOP, border=25)
            self.stride_widgets.append(self.method_note_text)
            self.method_note_text.Show()

        self.SetSizer(self.sizer)
        self.GetParent().Layout()


        return self.analysis_type_rbox.GetStringSelection()

    def onExtraBodypartsCheckbox(self, e):

        if self.extra_bodyparts_checkbox.GetValue():
            self.extra_bodyparts = True
        else:
            self.extra_bodyparts = False


    def ImportKinematicsCSV(self, e):

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
                
        try:
            if self.df is not None:
                self.has_imported_file = True
                self.has_input_path = False
                self.import_csv_text.SetLabel(
                    f"File imported! \n\n{self.filename}\n")
                self.GetParent().Layout()

                configs = ConfigFunctions.load_config('./config.yaml')
                self.cutoff_f = configs['lowpass_filter_cutoff']
                
                if self.analysis_type == 'Treadmill':
                    self.px_to_cm_speed_ratio = configs['px_to_cm_speed_ratio']
                    if configs['cm_speed'] == '':
                        self.cm_speed = 1
                    else:
                        self.cm_speed = configs['cm_speed']
                elif self.analysis_type == 'Spontaneous walking':
                    self.pixels_per_cm = configs['pixels_per_cm']

                self.method_label.Show()
                self.method_choices.Show()
                self.method_note_text.Show()

                try:    
                    self.select_output_folder_button.Hide()
                    self.select_output_folder_button.Destroy()
                    self.bulk_extract_parameters_button.Hide()
                    self.bulk_extract_parameters_button.Destroy()
                    self.extract_parameters_text.Hide()
                    self.extract_parameters_text.Destroy()
                except:
                    pass

                self.GetParent().Layout()

        except AttributeError:
            # user cancelled file import in pop up
            self.GetParent().Layout()
            pass


    def BulkImportKinematicsCSV(self, e):
        import_dialog = wx.DirDialog(
            self, 'Choose a folder', self.dirname, style=wx.DD_DEFAULT_STYLE)
        # Show the dialog and retrieve the user response.
        if import_dialog.ShowModal() == wx.ID_OK:
            self.input_path = import_dialog.GetPath()

            self.files = []
            for file in os.listdir(self.input_path):
                if file.endswith('.csv'):
                    self.files.append(file)
            
            try:
                if len(self.files) > 0:
                    self.df = None
                    self.has_input_path = True
                    self.has_imported_file = False
                    self.import_csv_text.SetLabel(
                        f"Loaded {len(self.files)} csv files found in {self.input_path}.\n")
                    self.GetParent().Layout()

                    configs = ConfigFunctions.load_config('./config.yaml')
                    self.cutoff_f = configs['lowpass_filter_cutoff']
                    if self.analysis_type == 'Treadmill':
                        self.px_to_cm_speed_ratio = configs['px_to_cm_speed_ratio']
                        if configs['cm_speed'] == '':
                            self.cm_speed = None
                        else:   
                            self.cm_speed = configs['cm_speed']
                    elif self.analysis_type == 'Spontaneous walking':
                        self.pixels_per_cm = configs['pixels_per_cm']

                    self.method_label.Show()
                    self.method_choices.Show()
                    self.method_note_text.Show()
                    self.GetParent().Layout()
                    try:    
                        
                        self.select_output_path_button.Hide()
                        self.select_output_path_button.Destroy()
                        self.extract_parameters_text.Hide()
                        self.extract_parameters_text.Destroy()
                        self.extract_parameters_button.Hide()
                        self.extract_parameters_button.Destroy()
                        
                    except:
                        pass
                    self.GetParent().Layout()

                    return self.input_path, self.files, len(self.files)

                else:
                    self.has_input_path = False

            except AttributeError:
                # user cancelled file import in pop up
                self.GetParent().Layout()
                pass

        else:
            self.input_path = ''

        import_dialog.Destroy()


    def OnMethod(self, e):

        if self.analysis_type == "Treadmill":
            self.method_selection = self.method_choices.GetValue()

            if self.method_selection == "Semi-automated":
                self.method_note_text.SetLabel(
                    "\nCurrent cm speed input: {self.cm_speed} cm / s."
                    + "\nThe pixel/frame speed and the corresponding px-to-cm speed ratio will be estimated and displayed. "
                    + "\nThe ratio can then be re-used for fully automated analysis, or for finding a more precise relationship, "
                    + "\ne.g., through a regression between the px-to-cm speed ratio at different speeds.")
                
                try:
                    self.px_cm_ratio_input.Destroy()
                    self.px_cm_ratio_input_button.Destroy()
                except:
                    pass
                
                try:
                    self.px_cm_length_ratio_input.Destroy()
                    self.px_cm_length_ratio_input_button.Destroy()
                except:
                    pass

                try:
                    self.cm_speed_input.Show()
                    self.cm_speed_input_button.Show()
                except:
                    self.cm_speed_input = wx.TextCtrl(self, value = str(self.cm_speed))
                    self.sizer.Add(self.cm_speed_input, pos= (9, 2), flag = wx.LEFT | wx.TOP, border = 25)
                    self.stride_widgets.append(self.cm_speed_input)

                    self.cm_speed_input_button = wx.Button(self, id = wx.ID_ANY, label = "Update speed (cm/s)")
                    self.cm_speed_input_button.Bind(wx.EVT_BUTTON, self.UpdateSpeed)
                    self.sizer.Add(self.cm_speed_input_button, pos = (9, 3), flag = wx.LEFT | wx.TOP, border = 25)
                    self.stride_widgets.append(self.cm_speed_input_button)

            elif self.method_selection == "Fully automated":

                self.method_note_text.SetLabel(
                    f"Using px to cm speed ratio: 1 pixel / frame = {self.px_to_cm_speed_ratio} cm / s.")

                try:
                    self.cm_speed_input.Destroy()
                    self.cm_speed_input_button.Destroy()
                except: # doesn't exist
                    pass

                try:
                    self.px_cm_length_ratio_input.Destroy()
                    self.px_cm_length_ratio_input_button.Destroy()
                except: # doesn't exist
                    pass

                try:
                    self.px_cm_ratio_input.Show()
                    self.px_cm_ratio_input_button.Show()
                except: # doesn't exist
                    self.px_cm_ratio_input = wx.TextCtrl(self, value = str(self.px_to_cm_speed_ratio))
                    self.sizer.Add(self.px_cm_ratio_input, pos= (9, 2), flag = wx.LEFT | wx.TOP, border = 25)
                    self.stride_widgets.append(self.px_cm_ratio_input)

                    self.px_cm_ratio_input_button = wx.Button(self, id = wx.ID_ANY, label = "Update pixel-to-cm speed ratio")
                    self.px_cm_ratio_input_button.Bind(wx.EVT_BUTTON, self.UpdateRatio)
                    self.sizer.Add(self.px_cm_ratio_input_button, pos = (9, 3), flag = wx.LEFT | wx.TOP, border = 25)
                    self.stride_widgets.append(self.px_cm_ratio_input_button)
        
        elif self.analysis_type == "Spontaneous walking":

            self.method_note_text.SetLabel(
                f"Using a spontaneous overground walking set-up. Parameters will be extracted automatically."
                    + f"\n\nCurrently using a length conversion ratio of {self.pixels_per_cm} pixels / cm (needs to be obtained manually).")

            try:
                self.cm_speed_input.Destroy()
                self.cm_speed_input_button.Destroy()
            except: # doesn't exist
                pass

            try:
                self.px_cm_ratio_input.Destroy()
                self.px_cm_ratio_input.Destroy()
            except: # doesn't exist
                pass

            try:
                self.px_cm_length_ratio_input.Show()
                self.px_cm_length_ratio_input_button.Show()
            except:
                self.px_cm_length_ratio_input = wx.TextCtrl(self, value = str(self.pixels_per_cm))
                self.sizer.Add(self.px_cm_length_ratio_input, pos= (9, 2), flag = wx.LEFT | wx.TOP, border = 25)
                self.stride_widgets.append(self.px_cm_length_ratio_input)

                self.px_cm_length_ratio_input_button = wx.Button(self, id = wx.ID_ANY, label = "Update px/cm length ratio")
                self.px_cm_length_ratio_input_button.Bind(wx.EVT_BUTTON, self.UpdateLengthRatio)
                self.sizer.Add(self.px_cm_length_ratio_input_button, pos = (9, 3), flag = wx.LEFT | wx.TOP, border = 25)
                self.stride_widgets.append(self.px_cm_length_ratio_input_button)
        
        self.EstimateParams(self)
        self.SetSizer(self.sizer)
        self.GetParent().Layout()


    def UpdateSpeed(self, e):

        self.cm_speed = float(self.cm_speed_input.GetValue())
        self.EstimateParams(self)
        self.GetParent().Layout()


    def UpdateRatio(self, e):

        self.px_to_cm_speed_ratio = float(self.px_cm_ratio_input.GetValue())
        self.EstimateParams(self)
        self.GetParent().Layout()


    def UpdateLengthRatio(self, e):
        self.pixels_per_cm = float(self.px_cm_length_ratio_input.GetValue())
        self.EstimateParams(self)
        self.GetParent().Layout()


    def EstimateParams(self, e):

        if self.has_imported_file:
            try:
                self.extract_parameters_text.Destroy()
                self.select_output_folder_button.Destroy()
                self.bulk_extract_parameters_button.Destroy()
            except:
                pass
            
            if self.analysis_type == "Treadmill":

                if self.method_selection == 'Semi-automated':

                    self.est_cm_speed, self.est_px_speed, self.est_pixels_per_cm, self.est_px_to_cm_speed_ratio = KinematicsFunctions.estimate_speed(self.df, 'toe', self.cm_speed, None, self.frame_rate)

                    self.method_note_text.SetLabel(
                        f"Recorded: {self.cm_speed} cm / s;"
                        + f"\n\nEstimated: \nPixel speed: {self.est_px_speed} px / frame."
                        + f"\nLength conversion: {self.est_pixels_per_cm} px / cm."
                        + f"\npx-to-cm speed ratio: 1 px / frame = {self.est_px_to_cm_speed_ratio} cm / s."
                        + "\n(The ratio can then be re-used for fully automated analysis, or for finding a more precise relationship, "
                        + "\ne.g., through a regression between the px-to-cm speed ratio at different speeds.)")
                
                elif self.method_selection == 'Fully automated':

                    self.est_cm_speed, self.est_px_speed, self.est_pixels_per_cm, self.est_px_to_cm_speed_ratio = KinematicsFunctions.estimate_speed(self.df, 'toe', None, self.px_to_cm_speed_ratio, self.frame_rate)

                    self.method_note_text.SetLabel(
                        f'Recorded: 1 px / frame = {self.px_to_cm_speed_ratio} cm / s.'
                        + f"\n\nEstimated: \nPixel speed: {self.est_px_speed} px / frame."
                        + f"\nRecording cm speed: {self.est_cm_speed} cm / s."
                        + f"\nLength conversion: {self.est_pixels_per_cm} pixels per cm.")

            elif self.analysis_type == "Spontaneous walking":
                self.method_note_text.SetLabel(
                    f"Using a spontaneous overground walking set-up. Parameters will be extracted automatically."
                    + f"\n\nCurrently using a length conversion ratio of {self.pixels_per_cm} pixels / cm (needs to be obtained manually)."
                )

            try:
                self.select_output_path_button.Show()
            except:
                self.select_output_path_button = wx.Button(
                    self, id=wx.ID_ANY, label='Select output path')
                self.sizer.Add(self.select_output_path_button, pos=(
                    11, 1), flag=wx.TOP | wx.LEFT | wx.BOTTOM, border=25)
                self.stride_widgets.append(self.select_output_path_button)
                self.select_output_path_button.Bind(wx.EVT_BUTTON, self.SelectOutputPath)
                self.select_output_path_button.Show()
            try:
                self.extract_parameters_text.Show()
            except:
                self.extract_parameters_text = wx.StaticText(self, label="")
                self.sizer.Add(self.extract_parameters_text, pos=(
                    12, 1), flag=wx.TOP | wx.BOTTOM, border=25)
                self.stride_widgets.append(self.extract_parameters_text)
                self.extract_parameters_text.Show()
            self.SetSizer(self.sizer)
            self.GetParent().Layout()

        elif self.has_input_path:
            try:
                self.extract_parameters_text.Destroy()
                self.select_output_path_button.Destroy()
                self.extract_parameters_button.Destroy()
            except:
                pass
            if self.analysis_type == "Treadmill":

                if self.method_selection == 'Semi-automated':
                    self.method_note_text.SetLabel(
                        f"Estimating pixel speed at {self.cm_speed} cm / s, which will be used for all files in the folder."
                        + f"\nPlease make sure all recordings have the same speed setting.")
                elif self.method_selection == 'Fully automated':

                    self.method_note_text.SetLabel(
                        f"Using px to cm speed ratio: 1 pixel / frame = {self.px_to_cm_speed_ratio} cm / s."
                        + f"\nPlease make sure all recordings have the same set up (e.g., distance from camera).")

            elif self.analysis_type == "Spontaneous walking":

                self.method_note_text.SetLabel(
                        f"\n\nCurrently using a length conversion ratio of {self.pixels_per_cm} pixels / cm (needs to be obtained manually)."
                        + "\nPlease make sure all recordings have the same px-cm length conversion ratio.")

            try:
                self.select_output_folder_button.Show()
            except:
                self.select_output_folder_button = wx.Button(
                    self, id=wx.ID_ANY, label='Select output folder')
                self.sizer.Add(self.select_output_folder_button, pos=(
                    11, 1), flag=wx.TOP | wx.BOTTOM, border=25)
                self.stride_widgets.append(self.select_output_folder_button)
                self.select_output_folder_button.Bind(wx.EVT_BUTTON, self.SelectOutputFolder)
                self.select_output_folder_button.Show()
            try:
                self.extract_parameters_text.Show()
            except:
                self.extract_parameters_text = wx.StaticText(self, label="")
                self.sizer.Add(self.extract_parameters_text, pos=(
                    12, 1), flag=wx.TOP | wx.BOTTOM, border=25)
                self.stride_widgets.append(self.extract_parameters_text)
                self.extract_parameters_text.Show()

            self.SetSizer(self.sizer)
            self.GetParent().Layout()


    def SelectOutputFolder(self, e):
        export_dialog = wx.DirDialog(
            self, 'Choose a folder to save parameter extraction results', self.dirname, style=wx.DD_DEFAULT_STYLE)
        if export_dialog.ShowModal() == wx.ID_OK:
            self.output_path = export_dialog.GetPath()
            self.extract_parameters_text.SetLabel(f"Saving to {self.output_path}. You can now start parameter extraction. \n\n"
            + "Parameter extraction can take a few minutes per file (depending on your computer processors). Please be patient.")
        else:
            pass
        export_dialog.Destroy()
        if self.output_path is not None:
            self.GetParent().SetStatusText("Ready to extract parameters!")
        try:
            self.bulk_extract_parameters_button = wx.Button(
                self, id=wx.ID_ANY, label='Start parameter extraction')
            self.sizer.Add(self.bulk_extract_parameters_button, pos=(
                11, 2), flag=wx.TOP | wx.BOTTOM, border=25)
            self.stride_widgets.append(self.bulk_extract_parameters_button)
            self.bulk_extract_parameters_button.Bind(wx.EVT_BUTTON, self.BulkExtractParameters)
            self.bulk_extract_parameters_button.Show()
        except:
            pass
        self.GetParent().Layout()


    def BulkExtractParameters(self, e):
        
        for i, file in enumerate(self.files):

            self.GetParent().SetStatusText(f"Extracting parameters for file {i+1} out of {len(self.files)}...")
            self.GetParent().Layout()
            self.filename = os.path.join(self.input_path, file)
            self.df, self.filename = KinematicsFunctions.read_file(self.filename)
            self.df, _ = KinematicsFunctions.fix_column_names(self.df)
            
            plot_path = os.path.join(self.output_path, f'truncated_stickplot_{file.replace(".csv", ".svg")}')
            if self.analysis_type == "Treadmill":
            
                if self.method_selection == 'Semi-automated':

                    self.est_cm_speed, self.est_px_speed, self.est_pixels_per_cm, self.est_px_to_cm_speed_ratio = KinematicsFunctions.estimate_speed(self.df, 'toe', self.cm_speed, None, self.frame_rate)

                    parameters, pd_dataframe_coords, is_stance, bodyparts = KinematicsFunctions.extract_parameters(self.frame_rate, self.df, self.cutoff_f, 'toe', 
                        cm_speed = self.cm_speed, px_to_cm_speed_ratio = self.est_px_to_cm_speed_ratio)
                    parameters_truncated = KinematicsFunctions.return_continuous(parameters, 10, True, pd_dataframe_coords, bodyparts, is_stance, plot_path)
                    
                elif self.method_selection == 'Fully automated':

                    self.est_cm_speed, self.est_px_speed, self.est_pixels_per_cm, self.est_px_to_cm_speed_ratio = KinematicsFunctions.estimate_speed(self.df, 'toe', None, self.px_to_cm_speed_ratio, self.frame_rate)

                    parameters, pd_dataframe_coords, is_stance, bodyparts = KinematicsFunctions.extract_parameters(self.frame_rate, self.df, self.cutoff_f, 'toe', 
                        cm_speed = self.est_cm_speed, px_to_cm_speed_ratio = self.px_to_cm_speed_ratio)
                    parameters_truncated = KinematicsFunctions.return_continuous(parameters, 10, True, pd_dataframe_coords, bodyparts, is_stance, plot_path)
                
            elif self.analysis_type == "Spontaneous walking":

                parameters, pd_dataframe_coords, is_stance, bodyparts = KinematicsFunctions.extract_spontaneous_parameters(self.frame_rate, self.df, self.cutoff_f, self.pixels_per_cm, self.no_outlier_filter, self.dragging_filter)
                parameters_truncated = KinematicsFunctions.return_continuous(parameters, 5, True, pd_dataframe_coords, bodyparts, is_stance, plot_path)
            
            KinematicsFunctions.make_parameters_output(os.path.join(self.output_path, f'parameters_{file}'), parameters)
            KinematicsFunctions.make_parameters_output(os.path.join(self.output_path, f'continuous_strides_parameters_{file}'), parameters_truncated)                    
            
        KinematicsFunctions.make_averaged_output(self.output_path) # average of all steps per animal
        KinematicsFunctions.make_averaged_output(self.output_path, truncated=True) # average of continuous steps per animal

        self.GetParent().SetStatusText(
            f"\nKinematic parameters have been extracted and saved to {self.output_path}!\n")


    def SelectOutputPath(self, e):

        export_dialog = wx.FileDialog(self, 'Save parameter extraction results to... ',
                            self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*',
                            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if export_dialog.ShowModal() == wx.ID_OK:
            self.output_path = export_dialog.GetPath()
            self.extract_parameters_text.SetLabel(f"Saving to {self.output_path}. You can now start parameter extraction. \n\n"
            + "Parameter extraction can take a few minutes per file (depending on your computer processors). Please be patient.")
            self.GetParent().SetStatusText("Ready to extract parameters!")
            try:
                self.extract_parameters_button.Show()
            except:
                self.extract_parameters_button = wx.Button(
                    self, id=wx.ID_ANY, label='Start parameter extraction')
                self.sizer.Add(self.extract_parameters_button, pos=(
                    11, 2), flag=wx.TOP | wx.BOTTOM, border=25)
                self.stride_widgets.append(self.extract_parameters_button)
                self.extract_parameters_button.Bind(wx.EVT_BUTTON, self.ExtractParameters)
            self.extract_parameters_button.Show()
            self.GetParent().Layout()
        else:
            pass

        export_dialog.Destroy()


    def ExtractParameters(self, e):

        self.GetParent().SetStatusText(
            f"\nWorking hard to extract 44 kinematic parameters for {self.filename}...\n")
        self.GetParent().Layout()

        plot_path = f'{self.output_path}_truncated_stickplot.svg'
        if self.analysis_type == "Treadmill":
            if self.method_selection == 'Semi-automated':
                parameters, pd_dataframe_coords, is_stance, bodyparts = KinematicsFunctions.extract_parameters(self.frame_rate, self.df, self.cutoff_f, 'toe', 
                    cm_speed = self.cm_speed, px_to_cm_speed_ratio = self.est_px_to_cm_speed_ratio)
            elif self.method_selection == 'Fully automated':
                parameters, pd_dataframe_coords, is_stance, bodyparts = KinematicsFunctions.extract_parameters(self.frame_rate, self.df, self.cutoff_f, 'toe', 
                    cm_speed = self.est_cm_speed, px_to_cm_speed_ratio = self.px_to_cm_speed_ratio)
            parameters_truncated = KinematicsFunctions.return_continuous(parameters, 10, True, pd_dataframe_coords, bodyparts, is_stance, plot_path)

        elif self.analysis_type == "Spontaneous walking":
            parameters, pd_dataframe_coords, is_stance, bodyparts = KinematicsFunctions.extract_spontaneous_parameters(self.frame_rate, self.df, self.cutoff_f, self.pixels_per_cm, self.no_outlier_filter, self.dragging_filter)
            parameters_truncated = KinematicsFunctions.return_continuous(parameters, 5, True, pd_dataframe_coords, bodyparts, is_stance, plot_path)
            KinematicsFunctions.make_parameters_output(self.output_path, parameters_truncated)                    

        KinematicsFunctions.make_parameters_output(self.output_path, parameters)
        KinematicsFunctions.make_parameters_output(os.path.join(os.path.dirname(self.output_path), f'continuous_strides_parameters_{self.filename}.csv'), parameters_truncated)

        self.GetParent().SetStatusText(
            f"\nKinematic parameters have been extracted and saved to {self.output_path}! Ready for between-group analysis\n")