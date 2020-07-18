import wx
from wx.lib.stattext import GenStaticText as StaticText
from Functions import ValidateFunctions, ConfigFunctions
import os
import yaml
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import numpy as np # remove later; for testing

    #################################
    # display file and video name after import
    # (automatically find csv output from same session?)
    # generate plots --> plot onset and offset?
    # add checkbox --> clickable button? to label slips
    # --> predict peaks (start with baseline correction & scipy find peak)
    # prev and next prediction button
    # start another button
    # --> get window size and resize elements?
    # --> input bodypart name & likelihood threshold
    # prompt user to save before going to another panel
    # regenerate panel after leaving
    #################################

TEST = True
# set to True: import default files to reduce clicks :)
# default files are set in ValidateFunctions.test()

# currently using default slip detecting algorithm ("baseline")
# which overestimates
# will be finetuned soon
# default self.bodypart = 'HR'

class ValidatePanel(wx.Panel):


    def __init__(self, parent):
        
        wx.Panel.__init__(self, parent=parent)

        # load parameters to set dimension of frames and graphs
        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width, self.window_height = configs['window_width'], configs['window_height']

        self.has_imported_file = False

        self.dirname = os.getcwd()

        # create a sizer to manage the layout of child widgets
        self.sizer = wx.GridBagSizer(0, 0)

        self.header = wx.StaticText(self, -1, "Validate")
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.header.SetFont(font)
        self.sizer.Add(self.header, pos = (0, 0), flag = wx.LEFT|wx.TOP, border = 25)

        self.instructions = wx.StaticText(self, -1, "Load the csv output from DeepLabCut and validate behavioral predictions manually.")
        self.sizer.Add(self.instructions, pos = (1, 0), span = (1, 3), flag = wx.LEFT|wx.TOP, border=25)

        self.import_csv_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        self.import_csv_button.Bind(wx.EVT_BUTTON, self.ImportCSV)
        self.sizer.Add(self.import_csv_button, pos = (8, 0), flag = wx.LEFT, border = 25)

        self.import_csv_text = wx.StaticText(self, label = "Import a csv file for slip prediction. " + "\nThe file should be DeepLabCut output (or of a similar format) to ensure proper parsing!")
        self.sizer.Add(self.import_csv_text, pos=(8, 1), flag = wx.LEFT, border = 25)

        self.pred_text = wx.StaticText(self, label = "The default algorithm will make a prediction using csv input.")
        self.sizer.Add(self.pred_text, pos= (9, 1) , flag = wx.LEFT | wx.TOP, border = 25)        

        self.save_pred_button = wx.Button(self, id=wx.ID_ANY, label="Save initial prediction")
        self.save_pred_button.Bind(wx.EVT_BUTTON, self.SavePredFunc)
        self.sizer.Add(self.save_pred_button, pos = (10, 1), flag = wx.TOP | wx.LEFT | wx.BOTTOM, border = 25)
        self.save_pred_button.Hide()

        self.import_new_csv_button = wx.Button(self, id=wx.ID_ANY, label="Import a different file")
        self.import_new_csv_button.Bind(wx.EVT_BUTTON, self.ImportCSV)
        self.sizer.Add(self.import_new_csv_button, pos = (10, 2), flag = wx.TOP | wx.BOTTOM, border = 25)
        self.import_new_csv_button.Hide()

        self.import_video_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        self.import_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)
        self.sizer.Add(self.import_video_button, pos = (11, 0), flag = wx.LEFT | wx.TOP, border = 25)
        self.import_video_button.Hide()
        
        self.import_video_text = wx.StaticText(self, label = "Import the corresponding video file for validation. ")
        self.sizer.Add(self.import_video_text, pos=(11, 1), flag = wx.LEFT | wx.TOP, border = 25)
        self.import_video_text.Hide()

        self.validate_button = wx.Button(self, id=wx.ID_ANY, label="Validate")
        self.validate_button.Bind(wx.EVT_BUTTON, self.DisplayValidationFunc)
        self.sizer.Add(self.validate_button, pos = (12, 1), flag = wx.TOP | wx.LEFT, border = 25)
        self.validate_button.Hide()

        self.import_new_video_button = wx.Button(self, id=wx.ID_ANY, label="Import a different video")
        self.import_new_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)
        self.sizer.Add(self.import_new_video_button, pos = (12, 2), flag = wx.TOP, border = 25)
        self.import_new_video_button.Hide()


        ################################################################
        # user input here

        self.bodypart = 'HR' # modify this to include user selection
        self.axis = 'y'
        self.threshold = 0.4 # modify this to include user selection



        if TEST is True:
            self.filename, self.df, self.filtered_df, self.video = ValidateFunctions.test(TEST)


        self.SetSizer(self.sizer)

        self.Layout()



    def ImportCSV(self, e):
        if not TEST:
            import_dialog = wx.FileDialog(self, 'Choose a file', self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', wx.FD_OPEN)

            if import_dialog.ShowModal() == wx.ID_OK and not TEST:
                self.csv_dirname = import_dialog.GetDirectory()
                self.filename = os.path.join(self.csv_dirname, import_dialog.GetFilename())
                self.df, self.filename = ValidateFunctions.read_file(self.filename)
                self.df = ValidateFunctions.fix_column_names(self.df)
                self.filtered_df = ValidateFunctions.filter_predictions(self.df, self.bodypart, self.threshold)
        try: 
            if self.df is not None:
                self.has_imported_file = True
                self.import_csv_text.SetLabel("File imported! ")
                self.MakePrediction(self)

                self.Layout()

        except AttributeError:
            pass


    def MakePrediction(self, e):

        self.import_csv_button.Disable()

        n_pred, depth_pred, t_pred, start_pred, end_pred = ValidateFunctions.find_slips(self.df, self.bodypart, self.axis, 'baseline') 
        self.n_pred, self.depth_pred, self.t_pred, self.start_pred, self.end_pred = n_pred, depth_pred, t_pred, start_pred, end_pred
        self.pred_text.SetLabel(f"The algorithm predicted {self.n_pred} slips with an average depth of {np.mean(self.depth_pred):.2f} pixels.")

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

        if self.video is not None:
            self.has_imported_video = True
            self.import_video_text.SetLabel("Video imported! You can validate the prediction now.")
                    
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
                ValidateFunctions.make_output(pathname, self.t_pred, self.depth_pred, self.start_pred, self.end_pred)

            except IOError:
                wx.LogError(f"Cannot save current data in file {pathname}. Try another location or filename?")
                

    def DisplayValidationFunc(self, e):

        # remove previous content
        self.import_csv_button.Hide()
        self.import_csv_button.Destroy()

        self.import_csv_text.Hide()
        self.import_csv_text.Destroy()

        self.import_new_csv_button.Hide()
        self.import_new_csv_button.Destroy()

        self.pred_text.Hide()
        self.pred_text.Destroy()

        self.save_pred_button.Hide()
        self.save_pred_button.Destroy()

        self.import_video_button.Hide()
        self.import_video_button.Destroy()

        self.import_video_text.Hide()
        self.import_video_text.Destroy()

        self.import_new_video_button.Hide()
        self.import_new_video_button.Destroy()

        self.validate_button.Hide()
        self.validate_button.Destroy()

        configs = ConfigFunctions.load_config('./config.yaml')
        self.frame_rate = configs['frame_rate']
        
        if self.n_pred is not None:
            self.n_frame = self.t_pred[0]
        else:
            self.n_frame = 0
        self.checkbox = wx.CheckBox(self, label='Slip')
        self.checkbox.SetValue(True)

        # initialize validation results
        self.n_val, self.depth_val, self.t_val, self.start_val, self.end_val = self.n_pred, self.depth_pred[:], self.t_pred[:], self.start_pred[:], self.end_pred[:]

        # display frame from video
        frame = ValidateFunctions.plot_frame(self.video, 10, 
            (self.window_width-50) / 200, (self.window_height // 3) // 100, int(self.frame_rate))
        self.frame_canvas = FigureCanvas(self, -1, frame)
        self.sizer.Add(self.frame_canvas, pos= (8, 0), span = (4, 0),flag = wx.LEFT, border = 25)
        self.Fit()
        self.checkbox.Bind(wx.EVT_CHECKBOX, self.MarkSlip)
        self.sizer.Add(self.checkbox, pos = (8, 1), flag = wx.LEFT | wx.TOP, border = 25)

        # display prev / next buttons
        self.prev_pred_button = wx.Button(self, id=wx.ID_ANY, label="<- prev prediction")
        self.prev_pred_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'prev_pred' : self.SwitchFrame(event, new_frame))
        self.frame_label = wx.StaticText(self, label='Frame')
        self.next_pred_button = wx.Button(self, id=wx.ID_ANY, label="next prediction ->")
        self.next_pred_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'next_pred' : self.SwitchFrame(event, new_frame))

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)

        self.sizer.Add(self.prev_pred_button, pos = (9, 1), span = (0,2), flag = wx.LEFT | wx.RIGHT, border = 25)
        self.sizer.Add(self.frame_label, pos = (9, 3), span = (0,1), flag = wx.LEFT, border = 15)
        self.sizer.Add(self.next_pred_button, pos = (9, 4), span = (0,2), flag = wx.LEFT | wx.RIGHT, border = 25)

        self.prev_button = wx.Button(self, id=wx.ID_ANY, label="<")
        self.prev_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = -1 : self.SwitchFrame(event, new_frame))
        self.next_button = wx.Button(self, id=wx.ID_ANY, label=">")
        self.next_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 1 : self.SwitchFrame(event, new_frame))

        self.slider_label = wx.StaticText(self, label=str(self.n_frame))

        self.prev10_button = wx.Button(self, id=wx.ID_ANY, label="<<")
        self.prev10_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = -10 : self.SwitchFrame(event, new_frame))
        self.next10_button = wx.Button(self, id=wx.ID_ANY, label=">>")
        self.next10_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 10 : self.SwitchFrame(event, new_frame))

        ValidateFunctions.ControlButton(self)

        self.sizer.Add(self.prev10_button, pos = (10, 1), flag = wx.LEFT, border = 25)
        self.sizer.Add(self.prev_button, pos = (10, 2))        
        self.sizer.Add(self.slider_label, pos = (10, 3), flag = wx.LEFT | wx.RIGHT, border = 25)
        self.sizer.Add(self.next_button, pos = (10, 4))
        self.sizer.Add(self.next10_button, pos = (10, 5))

        self.save_val_button = wx.Button(self, id=wx.ID_ANY, label="Save validated results")
        self.save_val_button.Bind(wx.EVT_BUTTON, self.SaveValFunc)
        self.sizer.Add(self.save_val_button, pos = (11, 4), span = (1, 3), flag = wx.TOP | wx.BOTTOM, border = 40)

        # display location graphs
        graph = ValidateFunctions.plot_labels(self.df, self.n_frame, self.t_pred, self.start_pred, \
            self.end_pred, (self.window_width-50) / 100, (self.window_height // 3) // 100, self.bodypart, self.axis, self.threshold)
        self.graph_canvas = FigureCanvas(self, -1, graph)
        self.sizer.Add(self.graph_canvas, pos = (12, 0), span = (1, 6), flag = wx.TOP | wx.LEFT, border = 25)        
        self.Fit()

        # display slider
        self.slider = wx.Slider(self, value=self.n_frame, minValue=1, maxValue=len(self.df),
            style=wx.SL_HORIZONTAL)
        self.slider.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        self.sizer.Add(self.slider, pos=(13, 0), span = (2, 6), flag = wx.LEFT | wx.EXPAND | wx.TOP, border = 25)

        self.SetSizer(self.sizer)
        self.GetParent().Layout()

        wx.MessageBox("This function is still under development. Thanks for your patience! :)")


    def MarkSlip(self, e):
        # if isChecked: 
        sender = e.GetEventObject()
        isChecked = sender.GetValue()

        if isChecked:
            if self.n_frame not in self.t_pred:
                self.n_val += 1
                self.t_val.append(self.n_frame)
                self.depth_val.append(np.nan)
                self.start_val.append(np.nan)
                self.end_val.append(np.nan)
        else:
            if self.n_frame in self.t_pred:
                self.n_val -= 1
                index = self.t_pred.index(self.n_frame)
                self.depth_val.pop(index)
                self.t_val.pop(index)
                self.start_val.pop(index)
                self.end_val.pop(index)


    def OnSliderScroll(self, e):

        obj = e.GetEventObject()
        self.n_frame = obj.GetValue() - 1

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.slider_label.SetLabel(str(self.n_frame + 1))
        
        ValidateFunctions.ControlButton(self)

        self.DisplayPlots()


    def SwitchFrame(self, e, new_frame):

        if type(new_frame) is int:
            self.n_frame = self.n_frame + new_frame
        elif new_frame == 'next_pred':
            self.n_frame = self.next_pred
        elif new_frame == 'prev_pred':
            self.n_frame = self.prev_pred

        self.slider.SetValue(self.n_frame)
        self.slider_label.SetLabel(str(self.n_frame + 1))

        self.prev_pred, self.next_pred = ValidateFunctions.find_neighbors(self.n_frame, self.t_pred)

        ValidateFunctions.ControlButton(self)

        self.DisplayPlots()


    def SaveValFunc(self, e):

        with wx.FileDialog(self, 'Save validated results as... ', \
                           self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', \
                           wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as save_pred_dialog:
                            
            if save_pred_dialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = save_pred_dialog.GetPath()

            try:
                ValidateFunctions.make_output(pathname, self.t_val, self.depth_val, self.start_val, self.end_val)

            except IOError:
                wx.LogError(f"Cannot save current data in file {pathname}. Try another location or filename?")
