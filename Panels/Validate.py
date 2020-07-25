import wx
from wx.lib.stattext import GenStaticText as StaticText
from Functions import ValidateFunctions, ConfigFunctions
import os
import yaml
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import numpy as np # remove later; for testing
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=ResourceWarning)

    #################################
    # (automatically find csv output from same session?)
    # "start another" button
    # get window size and resize elements?
    # prompt user to save before going to another panel
    # add option to display prediction vs validated slips
    # update button after reject
    # --> only display validate button if method, bodypart and files are selected
    # --> fix next / prev pred after adding new slip (sort issue)
    #################################

TEST = False
# set to True: import default files to reduce clicks :)
# default files are set in ValidateFunctions.test()

class ValidatePanel(wx.Panel):


    def __init__(self, parent):
        
        wx.Panel.__init__(self, parent=parent)

        # load parameters to set dimension of frames and graphs
        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width, self.window_height, self.frame_rate = configs['window_width'], configs['window_height'], configs['frame_rate']
        self.first_sizer_widgets = []
        self.second_sizer_widgets = []
        self.has_imported_file = False
        self.dirname = os.getcwd()
        
        self.first_sizer = wx.GridBagSizer(0, 0)

        self.header = wx.StaticText(self, -1, "Validate")
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.header.SetFont(font)
        self.first_sizer.Add(self.header, pos = (0, 0), span = (2, 5), flag = wx.LEFT|wx.TOP, border = 25)

        self.instructions = wx.StaticText(self, -1, "Load the csv output from DeepLabCut and validate behavioral predictions manually.")
        self.first_sizer.Add(self.instructions, pos = (2, 0), span = (1, 3), flag = wx.LEFT|wx.TOP, border=25)

        self.FirstPage()


    def ImportCSV(self, e):
        if not TEST:
            import_dialog = wx.FileDialog(self, 'Choose a file', self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', wx.FD_OPEN)

            if import_dialog.ShowModal() == wx.ID_OK:
                self.csv_dirname = import_dialog.GetDirectory()
                self.filename = os.path.join(self.csv_dirname, import_dialog.GetFilename())
                self.df, self.filename = ValidateFunctions.read_file(self.filename)
                self.df, self.bodyparts = ValidateFunctions.fix_column_names(self.df)
                # self.filtered_df = ValidateFunctions.filter_predictions(self.df, self.bodyparts[0], self.threshold)
        try: 
            if self.df is not None:
                self.has_imported_file = True
                self.import_csv_text.SetLabel(f"File imported! \n\n{self.filename}\n")
                self.GetParent().Layout()

                self.method_label.Show()

                self.method_choices.SetSelection(0)
                self.method_selection = self.method_choices.GetValue()
                self.method_choices.Show()

                self.bodypart_label.Show()
                self.bodypart_choices.Hide()

                bodypart_choices = wx.CheckListBox(self, choices = self.bodyparts)
                self.first_sizer.Replace(self.bodypart_choices, bodypart_choices)
                self.bodypart_choices = bodypart_choices
                # self.bodypart_choices.SetCheckedItems([0])
                # self.bodypart = self.bodyparts[list(self.bodypart_choices.GetCheckedItems())[0]]
                self.first_sizer_widgets.append(self.bodypart_choices)
                self.bodypart_choices.Bind(wx.EVT_CHECKLISTBOX, self.OnBodypart)
                self.bodypart_choices.Show()

                # self.MakePrediction(self)

                self.GetParent().Layout()

        except AttributeError:
            # user cancelled file import in pop up
            pass


    def MakePrediction(self, e):

        self.import_csv_button.Disable()

        n_pred, depth_pred, t_pred, start_pred, end_pred, bodypart_list_pred = 0,[],[],[],[],[]

        for bodypart in self.selected_bodyparts:

            n_pred_temp, depth_pred_temp, t_pred_temp, start_pred_temp, end_pred_temp = \
                ValidateFunctions.find_slips(self.df, bodypart, 'y', panel=self, method = self.method_selection)
            n_pred += n_pred_temp
            depth_pred.extend(depth_pred_temp)
            t_pred.extend(t_pred_temp)
            start_pred.extend(start_pred_temp)
            end_pred.extend(t_pred_temp)
            bodypart_list_pred.extend([bodypart for i in range(n_pred_temp)])


        depth_pred = ValidateFunctions.sort_list(t_pred, depth_pred)
        start_pred = ValidateFunctions.sort_list(t_pred, start_pred)
        end_pred = ValidateFunctions.sort_list(t_pred, end_pred)
        bodypart_list_pred = ValidateFunctions.sort_list(t_pred, bodypart_list_pred)
        t_pred = sorted(t_pred)

        to_remove = ValidateFunctions.find_duplicates(t_pred)
        for ind in range(len(to_remove)-1, -1, -1):
            i = to_remove[ind]
            _,_,_,_,_ = t_pred.pop(i), depth_pred.pop(i), end_pred.pop(i), start_pred.pop(i), bodypart_list_pred.pop(i)
            n_pred -= 1

        self.n_pred, self.depth_pred, self.t_pred, self.start_pred, self.end_pred, self.bodypart_list_pred = n_pred, depth_pred, t_pred, start_pred, end_pred, bodypart_list_pred
        self.pred_text.SetLabel(f"\nThe algorithm predicted {self.n_pred} slips with an average depth of {np.mean(self.depth_pred):.2f} pixels.\n")

        self.save_pred_button.Show()
        self.import_new_csv_button.Show()
        self.import_video_button.Show()
        self.import_video_text.Show()
        self.GetParent().Layout()

            
    def ImportVideo(self, e):

        if not TEST:
            import_dialog = wx.FileDialog(self, 'Choose a file', self.dirname, '', 
                'Video files (*.avi)|*.avi|Video files (*.mp4)|*.mp4|Video files (*.webm)|*.webm|Video files (*.mov)|*.mov|All files(*.*)|*.*', wx.FD_OPEN)

            if import_dialog.ShowModal() == wx.ID_OK:
                self.video_dirname = import_dialog.GetDirectory()
                self.video = os.path.join(self.video_dirname, import_dialog.GetFilename())
                self.video_name = self.video.split('/')[-1]

        if self.video is not None:
            self.has_imported_video = True
            self.import_video_text.SetLabel(f"Video imported! \n\n{self.video_name}\n\nYou can validate the prediction now.")
                    
            self.import_video_button.Disable()
            self.validate_button.Show()
            self.import_new_video_button.Show()

            self.GetParent().Layout()


    def SavePredFunc(self, e):

        with wx.FileDialog(self, 'Save current prediction as... ', \
                           self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', \
                           wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as save_pred_dialog:
                            
            if save_pred_dialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = save_pred_dialog.GetPath()

            try:
                ValidateFunctions.make_output(pathname, self.t_pred, self.depth_pred, self.start_pred, self.end_pred, self.bodypart_list_pred)

            except IOError:
                wx.LogError(f"Cannot save current data in file {pathname}. Try another location or filename?")
                

    def MarkSlip(self, e):

        sender = e.GetEventObject()
        isChecked = sender.GetValue()

        if isChecked:
            if self.n_frame not in self.t_val:
                self.n_val += 1
                self.t_val.append(self.n_frame)
                self.depth_val.append(np.nan)
                self.start_val.append(np.nan)
                self.end_val.append(np.nan)
                self.bodypart_list_val.append(np.nan)
        else:
            if self.n_frame in self.t_val:
                self.n_val -= 1
                index = self.t_val.index(self.n_frame)
                self.depth_val.pop(index)
                self.t_val.pop(index)
                self.start_val.pop(index)
                self.end_val.pop(index)
                self.bodypart_list_val.pop(index)

    def MarkFrame(self, e, mark_type):

        sender = e.GetEventObject()
        isChecked = sender.GetValue()

        if mark_type == "slip": 
            if isChecked:
                if self.n_frame not in self.t_val:
                    self.n_val += 1
                    self.t_val.append(self.n_frame)
                    self.depth_val.append(np.nan)
                    self.start_val.append(np.nan)
                    self.end_val.append(np.nan)
                    self.bodypart_list_val.append(np.nan)

                    self.depth_val = ValidateFunctions.sort_list(self.t_val, self.depth_val)
                    self.start_val = ValidateFunctions.sort_list(self.t_val, self.start_val)
                    self.end_val = ValidateFunctions.sort_list(self.t_val, self.end_val)
                    self.bodypart_list_val = ValidateFunctions.sort_list(self.t_val, self.bodypart_list_val)
                    self.t_val = sorted(self.t_val)

            else:
                if self.n_frame in self.t_val:
                    self.n_val -= 1
                    index = self.t_val.index(self.n_frame)
                    self.depth_val.pop(index)
                    self.t_val.pop(index)
                    self.start_val.pop(index)
                    self.end_val.pop(index)
                    self.bodypart_list_val.pop(index)

        # assuming there is always an existing next/prev prediction
        # that matches the frame to be labelled as start/end of slip
        elif mark_type == "start":
            
            if self.n_frame in self.t_val:
                index = self.t_val.index(self.n_frame)
            else:
                index = self.t_val.index(self.next_val)
            
            if isChecked:
                if self.n_frame not in self.start_val:
                    self.start_val.pop(index)
                    self.start_val.insert(index, self.n_frame)
            else:
                if self.n_frame in self.start_val:
                    self.start_val.pop(index)
                    self.start_val.insert(index, np.nan)
            
        elif mark_type == "end":
            if self.n_frame in self.t_val:
                index = self.t_val.index(self.n_frame)
            else:
                index = self.t_val.index(self.prev_val)

            if isChecked:
                if self.n_frame not in self.end_val:
                    self.end_val.pop(index)
                    self.end_val.insert(index, self.n_frame)
            else:
                if self.n_frame in self.end_val:
                    self.end_val.pop(index)
                    self.end_val.insert(index, np.nan)

    def OnValidate(self, e):

        self.val_check_box.SetValue(True)

        if self.n_frame not in self.t_val:
            self.n_val += 1
            self.t_val.append(self.n_frame)
            self.depth_val.append(np.nan)
            self.start_val.append(np.nan)
            self.end_val.append(np.nan)
            self.bodypart_list_val.append(np.nan)

        self.n_frame = self.next_pred
        self.slider.SetValue(self.n_frame)
        self.slider_label.SetLabel(str(self.n_frame + 1))

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.prev_val, self.next_val = ValidateFunctions.find_neighbors(self.n_frame, self.t_val)

        ValidateFunctions.ControlButton(self)
        ValidateFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def OnReject(self, e):

        self.val_check_box.SetValue(False)

        if self.n_frame in self.t_val:
            self.n_val -= 1
            index = self.t_val.index(self.n_frame)
            self.depth_val.pop(index)
            self.t_val.pop(index)
            self.start_val.pop(index)
            self.end_val.pop(index)
            self.bodypart_list_val.pop(index)

        self.n_frame = self.next_pred
        self.slider.SetValue(self.n_frame)
        self.slider_label.SetLabel(str(self.n_frame + 1))

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.prev_val, self.next_val = ValidateFunctions.find_neighbors(self.n_frame, self.t_val)

        ValidateFunctions.ControlButton(self)
        ValidateFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def OnSliderScroll(self, e):

        obj = e.GetEventObject()
        self.n_frame = obj.GetValue() - 1

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.prev_val, self.next_val = ValidateFunctions.find_neighbors(self.n_frame, self.t_val)
        self.slider_label.SetLabel(str(self.n_frame + 1))
        
        ValidateFunctions.ControlButton(self)
        ValidateFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def SwitchFrame(self, e, new_frame):

        if type(new_frame) is int:
            self.n_frame = self.n_frame + new_frame
        elif new_frame == 'next_pred':
            # self.n_frame = self.next_pred
            self.n_frame = self.next_val
        elif new_frame == 'prev_pred':
            # self.n_frame = self.prev_pred
            self.n_frame = self.prev_val
        elif self.n_frame in self.t_val:
            index = self.t_val.index(self.n_frame)
            if self.start_val[index] is not np.nan and new_frame == 'start': 
                self.n_frame = self.start_val[index]
            elif self.end_val[index] is not np.nan and new_frame == 'end':
                self.n_frame = self.end_val[index]              
            
        self.slider.SetValue(self.n_frame)
        self.slider_label.SetLabel(str(self.n_frame + 1))

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.prev_val, self.next_val = ValidateFunctions.find_neighbors(self.n_frame, self.t_val)

        ValidateFunctions.ControlButton(self)
        ValidateFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def SaveValFunc(self, e):

        with wx.FileDialog(self, 'Save validated results as... ', \
                           self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', \
                           wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as save_pred_dialog:
                            
            if save_pred_dialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = save_pred_dialog.GetPath()

            try:
                ValidateFunctions.make_output(pathname, self.t_val, self.depth_val, self.start_val, self.end_val, self.bodypart_list_val)

            except IOError:
                wx.LogError(f"Cannot save current data in file {pathname}. Try another location or filename?")


    def FirstPage(self):

        self.import_csv_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        self.first_sizer.Add(self.import_csv_button, pos = (6, 0), flag = wx.LEFT, border = 25)
        self.first_sizer_widgets.append(self.import_csv_button)

        self.import_csv_text = wx.StaticText(self, label = "Import a csv file for slip prediction. " + "\nThe file should be DeepLabCut output (or of a similar format) to ensure proper parsing!")
        self.first_sizer.Add(self.import_csv_text, pos=(6, 1), flag = wx.LEFT, border = 25)
        self.first_sizer_widgets.append(self.import_csv_text)

        self.pred_text = wx.StaticText(self, label = "\nThe chosen algorithm will make a prediction using csv input.")
        self.first_sizer.Add(self.pred_text, pos= (7, 1) , flag = wx.LEFT, border = 25)      
        self.first_sizer_widgets.append(self.pred_text)  

        self.method_label = wx.StaticText(self, label = 'Select algorithm for slip prediction:') 
        self.first_sizer.Add(self.method_label, pos=(8, 1), flag = wx.LEFT | wx.TOP, border = 25)
        self.first_sizer_widgets.append(self.method_label)  
        self.method_label.Hide()

        methods = ['Threshold', 'Deviation', 'Baseline']
        self.method_choices = wx.ComboBox(self, choices = methods)
        self.first_sizer.Add(self.method_choices, pos= (8, 2) , flag = wx.LEFT | wx.TOP, border = 25)
        self.first_sizer_widgets.append(self.method_choices)
        self.method_choices.Bind(wx.EVT_COMBOBOX, self.OnMethod)
        self.method_choices.Hide()

        
        self.bodypart_label = wx.StaticText(self, label = 'Bodypart to validate:') 
        self.first_sizer.Add(self.bodypart_label, pos=(9, 1), flag = wx.LEFT | wx.TOP, border = 25)
        self.first_sizer_widgets.append(self.bodypart_label)  
        self.bodypart_label.Hide()

        self.bodyparts = []
        self.bodypart_choices = wx.CheckListBox(self, choices = self.bodyparts)
        self.first_sizer.Add(self.bodypart_choices, pos = (9, 2) , flag = wx.LEFT | wx.TOP, border = 25)
        self.bodypart_choices.Hide()

        #################################
        # add algorithm selection menu

        self.save_pred_button = wx.Button(self, id=wx.ID_ANY, label="Save initial prediction")
        self.first_sizer.Add(self.save_pred_button, pos = (10, 1), flag = wx.TOP | wx.LEFT | wx.BOTTOM, border = 25)
        self.first_sizer_widgets.append(self.save_pred_button)
        self.save_pred_button.Hide()

        self.import_new_csv_button = wx.Button(self, id=wx.ID_ANY, label="Import a different file")
        self.first_sizer.Add(self.import_new_csv_button, pos = (10, 2), flag = wx.TOP | wx.BOTTOM, border = 25)
        self.first_sizer_widgets.append(self.import_new_csv_button)
        self.import_new_csv_button.Hide()

        self.import_video_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        
        self.first_sizer.Add(self.import_video_button, pos = (11, 0), flag = wx.LEFT | wx.TOP, border = 25)
        self.first_sizer_widgets.append(self.import_video_button)
        self.import_video_button.Hide()
        
        self.import_video_text = wx.StaticText(self, label = "Import the corresponding video file for validation. ")
        self.first_sizer.Add(self.import_video_text, pos=(11, 1), flag = wx.LEFT | wx.TOP, border = 25)
        self.first_sizer_widgets.append(self.import_video_text)
        self.import_video_text.Hide()

        self.validate_button = wx.Button(self, id=wx.ID_ANY, label="Validate")
        
        self.first_sizer.Add(self.validate_button, pos = (12, 1), flag = wx.TOP | wx.LEFT, border = 25)
        self.first_sizer_widgets.append(self.validate_button)
        self.validate_button.Hide()

        self.import_new_video_button = wx.Button(self, id=wx.ID_ANY, label="Import a different video")
        self.first_sizer.Add(self.import_new_video_button, pos = (12, 2), flag = wx.TOP, border = 25)
        self.first_sizer_widgets.append(self.import_new_video_button)
        self.import_new_video_button.Hide()

        ################################################################
        # user input here

        # self.bodypart = 'HR' # modify this to include user selection
        # self.axis = 'y'
        # self.threshold = 0.4 # modify this to include user selection

        self.save_pred_button.Bind(wx.EVT_BUTTON, self.SavePredFunc)
        self.import_csv_button.Bind(wx.EVT_BUTTON, self.ImportCSV)
        self.import_new_csv_button.Bind(wx.EVT_BUTTON, self.ImportCSV)
        self.import_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)
        self.validate_button.Bind(wx.EVT_BUTTON, self.DisplaySecondPage)
        self.import_new_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)

        if TEST is True:
            self.filename, self.df, self.bodyparts, self.video, self.video_name = ValidateFunctions.test(TEST)

        self.SetSizer(self.first_sizer)
                
        self.GetParent().Layout()



    def SecondPage(self):

        for widget in self.first_sizer_widgets:
            try:
                widget.Hide()
                widget.Destroy()
            except:
                # widget has already been destroyed
                pass

        self.second_sizer_widgets = []
        self.second_sizer = wx.GridBagSizer(0, 0)

        # initialize checkboxes (related to n_frame and plotting)
        self.start_check_box = wx.CheckBox(self, label="Start of slip")
        self.val_check_box = wx.CheckBox(self, label="Slip")
        self.end_check_box = wx.CheckBox(self, label="End of slip")

        if self.n_pred is not None:
            self.n_frame = self.t_pred[0]
            # self.val_check_box.SetValue(True)
        else:
            self.n_frame = 0
            # self.val_check_box.SetValue(False)

        # if self.n_frame in self.start_pred:
        #     self.start_check_box.SetValue(True)
        
        # if self.n_frame in self.end_pred:
        #     self.end_check_box.SetValue(True)

        # initialize validation results
        self.likelihood_threshold = 0
        self.n_val, self.depth_val, self.t_val, self.start_val, self.end_val, self.bodypart_list_val = \
            self.n_pred, self.depth_pred[:], self.t_pred[:], self.start_pred[:], self.end_pred[:], self.bodypart_list_pred[:]

        # display frame from video
        frame = ValidateFunctions.plot_frame(self.video, self.n_frame, 
            (self.window_width-50) / 200, (self.window_height // 3) // 100, int(self.frame_rate))
        self.frame_canvas = FigureCanvas(self, -1, frame)
        self.second_sizer.Add(self.frame_canvas, pos= (8, 0), span = (6, 0),flag = wx.LEFT, border = 25)
        self.second_sizer_widgets.append(self.frame_canvas)
        self.Fit()

        self.validate_button = wx.Button(self, id=wx.ID_ANY, label="Confirm")
        self.validate_button.Bind(wx.EVT_BUTTON, self.OnValidate)
        self.second_sizer.Add(self.validate_button, pos = (8, 3), flag =  wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, border = 25)
        self.second_sizer_widgets.append(self.validate_button)

        self.reject_button = wx.Button(self, id=wx.ID_ANY, label="Reject")
        self.reject_button.Bind(wx.EVT_BUTTON, self.OnReject)
        self.second_sizer.Add(self.reject_button, pos = (8, 4), flag =  wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border = 25)
        self.second_sizer_widgets.append(self.reject_button)

        # display prev / next buttons
        self.prev_pred_button = wx.Button(self, id=wx.ID_ANY, label="<- prev prediction")
        self.prev_pred_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'prev_pred' : self.SwitchFrame(event, new_frame))
        self.frame_label = wx.StaticText(self, label='Frame')
        self.next_pred_button = wx.Button(self, id=wx.ID_ANY, label="next prediction ->")
        self.next_pred_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'next_pred' : self.SwitchFrame(event, new_frame))

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.prev_val, self.next_val = ValidateFunctions.find_neighbors(self.n_frame, self.t_val)

        self.second_sizer.Add(self.prev_pred_button, pos = (9, 1), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        self.second_sizer_widgets.append(self.prev_pred_button)
        self.second_sizer.Add(self.frame_label, pos = (9, 3), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.LEFT, border = -15)
        self.second_sizer_widgets.append(self.frame_label)
        self.second_sizer.Add(self.next_pred_button, pos = (9, 5), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        self.second_sizer_widgets.append(self.next_pred_button)

        self.prev_button = wx.Button(self, id=wx.ID_ANY, label="<")
        self.prev_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = -1 : self.SwitchFrame(event, new_frame))
        self.second_sizer_widgets.append(self.prev_button)
        self.next_button = wx.Button(self, id=wx.ID_ANY, label=">")
        self.next_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 1 : self.SwitchFrame(event, new_frame))
        self.second_sizer_widgets.append(self.next_button)

        self.slider_label = wx.StaticText(self, label=str(self.n_frame + 1))

        self.prev10_button = wx.Button(self, id=wx.ID_ANY, label="<<")
        self.prev10_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = -10 : self.SwitchFrame(event, new_frame))
        self.next10_button = wx.Button(self, id=wx.ID_ANY, label=">>")
        self.next10_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 10 : self.SwitchFrame(event, new_frame))

        self.second_sizer.Add(self.prev10_button, pos = (10, 1), span = (0,1), flag = wx.LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL, border = 25)
        self.second_sizer.Add(self.prev_button, pos = (10, 2), span = (0,1), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)        
        self.second_sizer.Add(self.slider_label, pos = (10, 3), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.LEFT, border = -15)
        self.second_sizer.Add(self.next_button, pos = (10, 5), span = (0,1), flag = wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border = -25)
        self.second_sizer.Add(self.next10_button, pos = (10, 6), span = (0,1))

        self.second_sizer_widgets.append(self.prev10_button)
        self.second_sizer_widgets.append(self.prev_button)
        self.second_sizer_widgets.append(self.slider_label)
        self.second_sizer_widgets.append(self.next_button)
        self.second_sizer_widgets.append(self.next10_button)


        self.start_check_box.Bind(wx.EVT_CHECKBOX, lambda event, mark_type = 'start' : self.MarkFrame(event, mark_type))
        self.second_sizer.Add(self.start_check_box, pos = (11, 1), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 20)
        self.second_sizer_widgets.append(self.start_check_box)

        self.val_check_box.Bind(wx.EVT_CHECKBOX, lambda event, mark_type = 'slip' : self.MarkFrame(event, mark_type))
        self.second_sizer.Add(self.val_check_box, pos = (11, 3), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 20)
        self.second_sizer_widgets.append(self.val_check_box)

        self.end_check_box.Bind(wx.EVT_CHECKBOX, lambda event, mark_type = 'end' : self.MarkFrame(event, mark_type))
        self.second_sizer.Add(self.end_check_box, pos = (11, 5), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 20)
        self.second_sizer_widgets.append(self.end_check_box)

        self.to_start_button = wx.Button(self, id=wx.ID_ANY, label="< start of slip")
        self.to_start_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'start' : self.SwitchFrame(event, new_frame))
        self.second_sizer.Add(self.to_start_button, pos = (12, 1), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.to_start_button)

        self.to_end_button = wx.Button(self, id=wx.ID_ANY, label="end of slip >")
        self.to_end_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'end' : self.SwitchFrame(event, new_frame))
        self.second_sizer.Add(self.to_end_button, pos = (12, 5), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.to_end_button)

        self.save_val_button = wx.Button(self, id=wx.ID_ANY, label="Save")
        self.save_val_button.Bind(wx.EVT_BUTTON, self.SaveValFunc)
        self.second_sizer.Add(self.save_val_button, pos = (13, 3), span = (0, 1), flag = wx.TOP | wx.BOTTOM, border = 15)
        self.second_sizer_widgets.append(self.save_val_button)

        self.restart_button = wx.Button(self, id=wx.ID_ANY, label="Load new file")
        self.restart_button.Bind(wx.EVT_BUTTON, self.DisplayFirstPage)
        self.second_sizer.Add(self.restart_button, pos = (13, 4), span = (0, 2), flag = wx.TOP | wx.BOTTOM, border = 15)
        self.second_sizer_widgets.append(self.restart_button)

        # display location graphs

        if self.n_frame in self.t_pred:
            self.bodypart = self.bodypart_list_pred[self.t_pred.index(self.n_frame)]
        elif self.bodypart is None:
            self.bodypart = self.selected_bodyparts[0]

        graph = ValidateFunctions.plot_labels(self.df, self.n_frame, self.method_selection, self.t_val, self.start_val, \
            self.end_val, (self.window_width-50) / 100, (self.window_height // 3) // 100, self.bodypart, 'y', self.likelihood_threshold)
        self.graph_canvas = FigureCanvas(self, -1, graph)
        self.second_sizer.Add(self.graph_canvas, pos = (14, 0), span = (1, 7), flag = wx.TOP | wx.LEFT, border = 25) 
        self.second_sizer_widgets.append(self.graph_canvas)       
        self.Fit()

        # display slider
        self.slider = wx.Slider(self, value=self.n_frame+1, minValue=1, maxValue=len(self.df),
            style=wx.SL_HORIZONTAL)
        self.slider.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        self.second_sizer.Add(self.slider, pos=(15, 0), span = (2, 7), flag = wx.LEFT | wx.EXPAND | wx.TOP | wx.RIGHT, border = 25)
        self.second_sizer_widgets.append(self.slider)

        self.likelihood_label = wx.StaticText(self, label = "Set likelihood threshold (data below threshold are labeled grey; between 0 and 1)")
        self.second_sizer.Add(self.likelihood_label, pos= (17, 0), span = (1, 3),flag = wx.LEFT | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.likelihood_label)

        self.likelihood_input = wx.TextCtrl(self, value = "0")
        self.second_sizer.Add(self.likelihood_input, pos= (17, 3), flag = wx.LEFT | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.likelihood_input)

        self.likelihood_button = wx.Button(self, id = wx.ID_ANY, label = "Update")
        self.likelihood_button.Bind(wx.EVT_BUTTON, self.OnLikelihood)
        self.second_sizer.Add(self.likelihood_button, pos = (17, 4), flag = wx.LEFT | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.likelihood_button)

        ValidateFunctions.ControlButton(self)
        ValidateFunctions.ControlPrediction(self)

        self.SetSizer(self.second_sizer)
        # self.Layout()
        self.GetParent().Layout()

    def DisplayFirstPage(self, e):

        for widget in self.second_sizer_widgets:
            try:
                widget.Hide()
                widget.Destroy()
            except:
                # widget has already been destroyed
                pass


        self.first_sizer_widgets = []
        self.first_sizer = wx.GridBagSizer(0, 0)

        self.first_sizer.Add(self.header, pos = (0, 0), span = (2, 5), flag = wx.LEFT|wx.TOP, border = 25)
        self.first_sizer.Add(self.instructions, pos = (2, 0), span = (1, 3), flag = wx.LEFT|wx.TOP, border=25)
        self.FirstPage()
        self.GetParent().Layout()

    def DisplaySecondPage(self, e):

        self.SecondPage()

    def OnMethod(self, e):

        self.selected_bodyparts = [self.bodyparts[i] for i in list(self.bodypart_choices.GetCheckedItems())]
        self.method_selection = self.method_choices.GetValue()
        try:
            self.MakePrediction(self)
            self.GetParent().Layout()
        except RuntimeWarning:
            # awaiting bodypart selection
            self.pred_text.SetLabel("\n\n")
            pass

    def OnBodypart(self, e):

        self.selected_bodyparts = [self.bodyparts[i] for i in list(self.bodypart_choices.GetCheckedItems())]
        try:
            self.MakePrediction(self)
            self.GetParent().Layout()
        except:
            # awaiting bodypart selection
            self.pred_text.SetLabel("\n\n")
            pass

    def OnLikelihood(self, e):

        self.likelihood_threshold = float(self.likelihood_input.GetValue())

        graph = ValidateFunctions.plot_labels(self.df, self.n_frame, self.t_val, self.start_val, \
            self.end_val, (self.window_width-50) / 100, (self.window_height // 3) // 100, self.bodypart, 'y', self.likelihood_threshold)
        graph_canvas = FigureCanvas(self, -1, graph)
        self.second_sizer.Replace(self.graph_canvas, graph_canvas)
        self.second_sizer_widgets.remove(self.graph_canvas) 
        self.graph_canvas = graph_canvas
        self.second_sizer_widgets.append(self.graph_canvas)
        self.Fit()

        self.GetParent().Layout()