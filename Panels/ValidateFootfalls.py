import wx
from wx.lib.stattext import GenStaticText as StaticText
from Functions import FootfallFunctions, ConfigFunctions
import os, sys
import yaml
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import numpy as np # remove later; for testing
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=ResourceWarning)

TEST = False
# set to True: import default files to reduce clicks :)
# default files are set in FootfallFunctions.test()

class ValidateFootfallPanel(wx.Panel):


    def __init__(self, parent):
        
        wx.Panel.__init__(self, parent=parent)

        # load parameters to set dimension of frames and graphs
        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width, self.window_height, self.frame_rate, self.likelihood_threshold, self.depth_threshold, self.threshold = \
            configs['window_width'], configs['window_height'], configs['frame_rate'], configs['likelihood_threshold'], configs['depth_threshold'], configs['threshold']
        self.first_sizer_widgets = []
        self.second_sizer_widgets = []
        self.has_imported_file = False
        self.dirname = os.getcwd()
        
        self.first_sizer = wx.GridBagSizer(0, 0)

        self.header = wx.StaticText(self, -1, "Ladder rung (footfall) analysis", size=(500,100))
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.header.SetFont(font)
        self.first_sizer.Add(self.header, pos = (0, 0), span = (1, 5), flag = wx.LEFT|wx.TOP, border = 25)

        self.instructions = wx.StaticText(self, -1, "Load the csv file of bodypart coordinates (e.g., from DLC) and validate detected footfalls.", 
                                            size=(self.window_width,30))
        font = wx.Font(15,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.instructions.SetFont(font)
        self.first_sizer.Add(self.instructions, pos = (1, 0), span = (1, 5), flag = wx.LEFT, border=25)
        self.first_sizer_widgets.append(self.instructions)
        self.FirstPage()


    def ImportBehavioralCSV(self, e):
        if not TEST:
            import_dialog = wx.FileDialog(self, 'Choose a file', self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', wx.FD_OPEN)

            if import_dialog.ShowModal() == wx.ID_OK:
                self.csv_dirname = import_dialog.GetDirectory()
                self.filename = os.path.join(self.csv_dirname, import_dialog.GetFilename())
                self.df, self.filename = FootfallFunctions.read_file(self.filename)
                self.df, self.bodyparts = FootfallFunctions.fix_column_names(self.df)
        try: 
            if self.df is not None:
                self.has_imported_file = True
                self.import_csv_text.SetLabel(f"File imported! \n\n{self.filename}\n")
                # self.GetParent().Layout()

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
            # print(sys.exc_info())
            pass


    def MakePrediction(self, e):

        self.import_csv_button.Disable()

        n_pred, depth_pred, t_pred, start_pred, end_pred, bodypart_list_pred = 0,[],[],[],[],[]

        for bodypart in self.selected_bodyparts:
            n_pred_temp, depth_pred_temp, t_pred_temp, start_pred_temp, end_pred_temp = \
                FootfallFunctions.find_footfalls(self.df, bodypart, 'y', panel=self, method = self.method_selection, likelihood_threshold = self.likelihood_threshold, depth_threshold = self.depth_threshold, threshold = self.threshold)
            n_pred += n_pred_temp
            depth_pred.extend(depth_pred_temp)
            t_pred.extend(t_pred_temp)
            start_pred.extend(start_pred_temp)
            end_pred.extend(end_pred_temp)
            bodypart_list_pred.extend([bodypart for i in range(n_pred_temp)])


        depth_pred = FootfallFunctions.sort_list(t_pred, depth_pred)
        start_pred = FootfallFunctions.sort_list(t_pred, start_pred)
        end_pred = FootfallFunctions.sort_list(t_pred, end_pred)
        bodypart_list_pred = FootfallFunctions.sort_list(t_pred, bodypart_list_pred)
        t_pred = sorted(t_pred)

        to_remove = FootfallFunctions.find_duplicates(t_pred)
        for ind in range(len(to_remove)-1, -1, -1):
            i = to_remove[ind]
            _,_,_,_,_ = t_pred.pop(i), depth_pred.pop(i), end_pred.pop(i), start_pred.pop(i), bodypart_list_pred.pop(i)
            n_pred -= 1

        self.n_pred, self.depth_pred, self.t_pred, self.start_pred, self.end_pred, self.bodypart_list_pred = n_pred, depth_pred, t_pred, start_pred, end_pred, bodypart_list_pred
        self.confirmed = [0]*self.n_pred
        # print(self.confirmed)
        self.pred_text.SetLabel(f"\nThe algorithm predicted {self.n_pred} footfalls {np.mean(self.depth_pred):.2f} pixels deep on average.\n")

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
                if '/' in self.video:
                    self.video_name = self.video.split('/')[-1]
                elif '\\' in self.video:
                    self.video_name = self.video.split('\\')[-1]

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
                FootfallFunctions.make_output(pathname, self.df, self.t_pred, self.depth_pred, self.start_pred, self.end_pred, self.bodypart_list_pred, self.frame_rate)

            except IOError:
                wx.LogError(f"Cannot save current data in file {pathname}. Try another location or filename?")
                

    # def MarkFootfall(self, e):

    #     sender = e.GetEventObject()
    #     isChecked = sender.GetValue()

    #     if isChecked:
    #         if self.n_frame not in self.t_val:
    #             self.n_val += 1
    #             self.t_val.append(self.n_frame)
    #             self.depth_val.append(np.nan)
    #             self.start_val.append(np.nan)
    #             self.end_val.append(np.nan)
    #             self.bodypart_list_val.append(np.nan)
    #             self.confirmed.append(1)
    #     else:
    #         if self.n_frame in self.t_val:
    #             self.n_val -= 1
    #             index = self.t_val.index(self.n_frame)
    #             self.depth_val.pop(index)
    #             self.t_val.pop(index)
    #             self.start_val.pop(index)
    #             self.end_val.pop(index)
    #             self.bodypart_list_val.pop(index)
    #             self.confirmed.pop(index)

    def MarkFrame(self, e, mark_type):

        sender = e.GetEventObject()
        isChecked = sender.GetValue()

        if mark_type == "confirmed": 
            if isChecked:
                if self.n_frame not in self.t_val:
                    self.n_val += 1
                    self.t_val.append(self.n_frame)
                    self.depth_val.append(np.nan)
                    self.start_val.append(np.nan)
                    self.end_val.append(np.nan)
                    self.bodypart_list_val.append(self.bodypart_to_plot.GetValue())
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

                self.GetParent().Layout()

            else:
                if self.n_frame in self.t_val:
                    self.n_val -= 1
                    index = self.t_val.index(self.n_frame)
                    # self.depth_val.pop(index)
                    # self.t_val.pop(index)
                    # self.start_val.pop(index)
                    # self.end_val.pop(index)
                    # self.bodypart_list_val.pop(index)
                    # self.confirmed.pop(index)
                    self.confirmed[index] = 0

        # assuming there is always an existing next/prev prediction
        # that matches the frame to be labelled as start/end of footfall
        elif mark_type == "start":
            _, self.next_val_confirmed = FootfallFunctions.find_confirmed_neighbors(self.n_frame, self.t_val, self.confirmed, end=self.t_val_id)

            if self.n_frame in self.t_val:
                index = self.t_val.index(self.n_frame)
            elif self.next_val_confirmed != 0:
                index = self.t_val.index(self.next_val_confirmed)
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
            self.prev_val_confirmed, _ = FootfallFunctions.find_confirmed_neighbors(self.n_frame, self.t_val, self.confirmed, start=self.t_val_id)
            if self.n_frame in self.t_val:
                index = self.t_val.index(self.n_frame)
            elif self.prev_val_confirmed != 0:
                index = self.t_val.index(self.prev_val_confirmed)
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

                FootfallFunctions.ControlButton(self)

        FootfallFunctions.DisplayPlots(self)
        self.GetParent().Layout()

    def OnValidate(self, e):

        self.val_check_box.SetValue(True)

        if self.n_frame not in self.t_val:
            self.n_val += 1
            self.t_val.append(self.n_frame)
            self.depth_val.append(np.nan)
            self.start_val.append(np.nan)
            self.end_val.append(np.nan)
            self.bodypart_list_val.append(self.bodypart)
            self.confirmed.append(1)

            self.depth_val = FootfallFunctions.sort_list(self.t_val, self.depth_val)
            self.start_val = FootfallFunctions.sort_list(self.t_val, self.start_val)
            self.end_val = FootfallFunctions.sort_list(self.t_val, self.end_val)
            self.bodypart_list_val = FootfallFunctions.sort_list(self.t_val, self.bodypart_list_val)
            self.confirmed = FootfallFunctions.sort_list(self.t_val, self.confirmed)
            self.t_val = sorted(self.t_val)

            self.t_val_id = self.t_val.index(self.n_frame)
            self.t_val_max += 1

            if self.t_val_id+1 >= self.t_val_max:
                self.prev_val = self.t_val[self.t_val_max-1]
                self.next_val = 0
                self.t_val_id += 1
            else:
                self.prev_val = self.t_val[self.t_val_id]
                self.next_val = self.t_val[self.t_val_id+2]
                self.t_val_id += 1


        else:
            index = self.t_val.index(self.n_frame)
            self.confirmed[index] = 1
            # print(self.confirmed)

            self.n_frame = self.next_pred
            self.slider.SetValue(self.n_frame)
            self.slider_label.SetLabel(str(self.n_frame + 1))

            # self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred)
            # self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val)
            if self.t_pred_id+1 >= self.t_pred_max:
                self.prev_pred = self.t_pred[self.t_pred_max-1]
                self.next_pred = 0
                self.t_pred_id += 1
            else:
                # print(self.t_pred_id, self.t_pred_max)
                self.prev_pred = self.t_pred[self.t_pred_id]
                self.next_pred = self.t_pred[self.t_pred_id+2]
                self.t_pred_id += 1
            
            if self.t_val_id+1 >= self.t_val_max:
                self.prev_val = self.t_val[self.t_val_max-1]
                self.next_val = 0
                self.t_val_id += 1
            else:
                self.prev_val = self.t_val[self.t_val_id]
                self.next_val = self.t_val[self.t_val_id+2]
                self.t_val_id += 1

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def OnReject(self, e):

        self.val_check_box.SetValue(False)

        if self.n_frame in self.t_val:
            self.n_val -= 1
            index = self.t_val.index(self.n_frame)
            # self.depth_val.pop(index)
            # self.t_val.pop(index)
            # self.start_val.pop(index)
            # self.end_val.pop(index)
            # self.bodypart_list_val.pop(index)
            self.confirmed[index] = 0


        self.n_frame = self.next_pred
        self.slider.SetValue(self.n_frame)
        self.slider_label.SetLabel(str(self.n_frame + 1))

        # self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred)
        # self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val)

        if self.t_pred_id+1 >= self.t_pred_max:
            self.prev_pred = self.t_pred[-1]
            self.next_pred = 0
            self.t_pred_id += 1
        else:
            self.prev_pred = self.t_pred[self.t_pred_id]
            self.next_pred = self.t_pred[self.t_pred_id+2]
            self.t_pred_id += 1
        
        if self.t_val_id+1 >= self.t_val_max:
            self.prev_val = self.t_val[-1]
            self.next_val = 0
            self.t_val_id += 1
        else:
            self.prev_val = self.t_val[self.t_val_id]
            self.next_val = self.t_val[self.t_val_id+2]
            self.t_val_id += 1

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def OnSliderScroll(self, e):

        obj = e.GetEventObject()
        self.n_frame = obj.GetValue() - 1

        # self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred)
        # self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val)
        if self.n_frame > self.prev_pred and self.n_frame < self.t_pred[self.t_pred_id]:
            self.next_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame < self.next_pred and self.n_frame > self.t_pred[self.t_pred_id]:
            self.prev_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame <= self.prev_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, end = self.t_pred_id)
            if self.prev_pred == 0:
                self.t_pred_id = 0
            elif self.next_pred == 0:
                self.t_pred_id = self.t_pred_max
            else:
                index = self.t_pred.index(self.prev_pred)
                self.t_pred_id = index + 1
        elif self.n_frame >= self.next_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, start = self.t_pred_id)
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
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, end = self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1
        elif self.n_frame >= self.next_val:
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, start = self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1

        self.slider_label.SetLabel(str(self.n_frame + 1))
        
        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def SwitchFrame(self, e, new_frame):

        if type(new_frame) is int:
            self.n_frame = self.n_frame + new_frame
        elif new_frame == 'next_pred':
            self.n_frame = self.next_val
        elif new_frame == 'prev_pred':
            self.n_frame = self.prev_val
        elif self.n_frame in self.t_val:
            index = self.t_val.index(self.n_frame)
            if self.start_val[index] is not np.nan and new_frame == 'start': 
                self.n_frame = self.start_val[index]
                self.next_val = self.t_val[index]
            elif self.end_val[index] is not np.nan and new_frame == 'end':
                self.n_frame = self.end_val[index]
                self.prev_val = self.t_val[index]
            
        self.slider.SetValue(self.n_frame)
        self.slider_label.SetLabel(str(self.n_frame + 1))

        if self.n_frame > self.prev_pred and self.n_frame < self.t_pred[self.t_pred_id]:
            self.next_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame < self.next_pred and self.n_frame > self.t_pred[self.t_pred_id]:
            self.prev_pred = self.t_pred[self.t_pred_id]
        elif self.n_frame <= self.prev_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, end = self.t_pred_id)
            if self.prev_pred == 0:
                self.t_pred_id = 0
            elif self.next_pred == 0:
                self.t_pred_id = self.t_pred_max
            else:
                index = self.t_pred.index(self.prev_pred)
                self.t_pred_id = index + 1
        elif self.n_frame >= self.next_pred:
            self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred, start = self.t_pred_id)
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
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, end = self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1
        elif self.n_frame >= self.next_val:
            self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val, start = self.t_val_id)
            if self.prev_val == 0:
                self.t_val_id = 0
            elif self.next_val == 0:
                self.t_val_id = self.t_val_max
            else:
                index = self.t_val.index(self.prev_val)
                self.t_val_id = index + 1

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)

        self.GetParent().Layout()


    def SaveValFunc(self, e):

        with wx.FileDialog(self, 'Save validated results as... ', \
                           self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', \
                           wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as save_pred_dialog:
                            
            if save_pred_dialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = save_pred_dialog.GetPath()

            try:
                FootfallFunctions.make_output(pathname, self.df, self.t_val, self.depth_val, self.start_val, self.end_val, self.bodypart_list_val, self.frame_rate, self.confirmed, True)

            except IOError:
                wx.LogError(f"Cannot save current data in file {pathname}. Try another location or filename?")


    def FirstPage(self):

        self.import_csv_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        self.first_sizer.Add(self.import_csv_button, pos = (6, 0), flag = wx.LEFT, border = 25)
        self.first_sizer_widgets.append(self.import_csv_button)

        self.import_csv_text = wx.StaticText(self, label = "Import a csv file for footfall detection. " + "\nThe file should be DeepLabCut output (or of a similar format) to ensure proper parsing!")
        self.first_sizer.Add(self.import_csv_text, pos=(6, 1), flag = wx.LEFT, border = 25)
        self.first_sizer_widgets.append(self.import_csv_text)

        self.pred_text = wx.StaticText(self, label = "\nThe chosen algorithm will make a detection using csv input.")
        self.first_sizer.Add(self.pred_text, pos= (7, 1) , flag = wx.LEFT, border = 25)      
        self.first_sizer_widgets.append(self.pred_text)  

        self.method_label = wx.StaticText(self, label = 'Select algorithm for footfall detection:') 
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

        self.save_pred_button = wx.Button(self, id=wx.ID_ANY, label="Save detected footfalls")
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

        self.save_pred_button.Bind(wx.EVT_BUTTON, self.SavePredFunc)
        self.import_csv_button.Bind(wx.EVT_BUTTON, self.ImportBehavioralCSV)
        self.import_new_csv_button.Bind(wx.EVT_BUTTON, self.ImportBehavioralCSV)
        self.import_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)
        self.validate_button.Bind(wx.EVT_BUTTON, self.DisplaySecondPage)
        self.import_new_video_button.Bind(wx.EVT_BUTTON, self.ImportVideo)

        if TEST is True:
            self.filename, self.df, self.bodyparts, self.video, self.video_name = FootfallFunctions.test(TEST)

        self.Fit()
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
        self.start_check_box = wx.CheckBox(self, label="Start of footfall")
        self.val_check_box = wx.CheckBox(self, label="Confirmed")
        self.end_check_box = wx.CheckBox(self, label="End of footfall")

        if self.n_pred is not None:
            self.n_frame = self.t_pred[0]
            self.t_pred_id = 0
            self.t_val_id = 0
            self.val_check_box.SetValue(self.confirmed[0])
        else:
            self.n_frame = 0
            self.t_pred_id = None
            self.t_val_id = None
        self.n_val, self.depth_val, self.t_val, self.start_val, self.end_val, self.bodypart_list_val = \
            self.n_pred, self.depth_pred[:], self.t_pred[:], self.start_pred[:], self.end_pred[:], self.bodypart_list_pred[:]

        # initialize pred and val footfall time indices
        self.t_pred_max = len(self.t_pred) - 1
        self.t_val_max = len(self.t_val) - 1

        if self.n_frame in self.t_pred:
            self.bodypart = self.bodypart_list_pred[self.t_pred.index(self.n_frame)]
        else:
            self.bodypart = self.selected_bodyparts[0]


        self.file_info_button = wx.Button(self, id=wx.ID_ANY, label="Show file information")
        self.file_info_button.Bind(wx.EVT_BUTTON, self.display_info)
        self.second_sizer.Add(self.file_info_button, pos = (7, 0), flag = wx.LEFT | wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, border=25)
        self.second_sizer_widgets.append(self.file_info_button)

        self.zoom = False
        self.zoom_image = False

        # display frame from video
        self.zoom_frame_button = wx.Button(self, id=wx.ID_ANY, label="Zoom in")
        self.zoom_frame_button.Bind(wx.EVT_BUTTON, self.zoom_frame)
        self.second_sizer.Add(self.zoom_frame_button, pos = (7, 1), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT | wx.BOTTOM, border = 25)
        self.second_sizer_widgets.append(self.zoom_frame_button)

        frame = FootfallFunctions.plot_frame(self.video, self.n_frame, 
            8, 4, int(self.frame_rate), self.df, self.bodypart, self.zoom_image)
        self.frame_canvas = FigureCanvas(self, -1, frame)

        self.second_sizer.Add(self.frame_canvas, pos= (8, 0), span = (6, 2), flag = wx.LEFT, border = 25)
        self.second_sizer_widgets.append(self.frame_canvas)

        self.validate_button = wx.Button(self, label="Confirm (space)")
        self.Bind(wx.EVT_BUTTON, self.OnValidate, self.validate_button)
        self.second_sizer.Add(self.validate_button, pos = (8, 4), flag =  wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, border = 25)
        self.second_sizer_widgets.append(self.validate_button)

        self.reject_button = wx.Button(self, id=wx.ID_ANY, label="Reject")
        self.reject_button.Bind(wx.EVT_BUTTON, self.OnReject)
        self.second_sizer.Add(self.reject_button, pos = (8, 5), flag =  wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border = 25)
        self.second_sizer_widgets.append(self.reject_button)

        # display prev / next buttons
        self.prev_pred_button = wx.Button(self, id=wx.ID_ANY, label="<- prev detected footfall (A)")
        self.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'prev_pred' : self.SwitchFrame(event, new_frame), self.prev_pred_button)
        self.frame_label = wx.StaticText(self, label='Frame')
        self.next_pred_button = wx.Button(self, id=wx.ID_ANY, label="next detected footfall (D) ->")
        self.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'next_pred' : self.SwitchFrame(event, new_frame), self.next_pred_button)


        self.prev_pred, self.next_pred = FootfallFunctions.find_neighbors(self.n_frame, self.t_pred)
        self.prev_val, self.next_val = FootfallFunctions.find_neighbors(self.n_frame, self.t_val)

        self.second_sizer.Add(self.prev_pred_button, pos = (9, 2), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        self.second_sizer_widgets.append(self.prev_pred_button)
        self.second_sizer.Add(self.frame_label, pos = (9, 4), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.LEFT, border = -15)
        self.second_sizer_widgets.append(self.frame_label)
        self.second_sizer.Add(self.next_pred_button, pos = (9, 6), span = (0,1), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
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

        self.second_sizer.Add(self.prev10_button, pos = (10, 2), span = (0,1), flag = wx.LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 10)
        self.second_sizer.Add(self.prev_button, pos = (10, 3), span = (0,1), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 10)        
        self.second_sizer.Add(self.slider_label, pos = (10, 4), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 10)
        self.second_sizer.Add(self.next_button, pos = (10, 6), span = (0,1), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 10)
        self.second_sizer.Add(self.next10_button, pos = (10, 7), span = (0,1), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 10)

        self.second_sizer_widgets.append(self.prev10_button)
        self.second_sizer_widgets.append(self.prev_button)
        self.second_sizer_widgets.append(self.slider_label)
        self.second_sizer_widgets.append(self.next_button)
        self.second_sizer_widgets.append(self.next10_button)

        self.start_check_box.Bind(wx.EVT_CHECKBOX, lambda event, mark_type = 'start' : self.MarkFrame(event, mark_type))
        self.second_sizer.Add(self.start_check_box, pos = (11, 2), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 20)
        self.second_sizer_widgets.append(self.start_check_box)

        self.val_check_box.Bind(wx.EVT_CHECKBOX, lambda event, mark_type = 'confirmed' : self.MarkFrame(event, mark_type))
        self.second_sizer.Add(self.val_check_box, pos = (11, 4), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 20)
        self.second_sizer_widgets.append(self.val_check_box)

        self.end_check_box.Bind(wx.EVT_CHECKBOX, lambda event, mark_type = 'end' : self.MarkFrame(event, mark_type))
        self.second_sizer.Add(self.end_check_box, pos = (11, 6), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 20)
        self.second_sizer_widgets.append(self.end_check_box)

        self.to_start_button = wx.Button(self, id=wx.ID_ANY, label="< start of footfall")
        self.to_start_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'start' : self.SwitchFrame(event, new_frame))
        self.second_sizer.Add(self.to_start_button, pos = (12, 2), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.to_start_button)

        self.bodypart_to_plot = wx.ComboBox(self, choices = self.selected_bodyparts)
        self.second_sizer.Add(self.bodypart_to_plot, pos= (12, 4) , flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.bodypart_to_plot)
        self.bodypart_to_plot.Bind(wx.EVT_COMBOBOX, self.OnBodypartPlot)

        self.zoom_button = wx.Button(self, id=wx.ID_ANY, label="Zoom in plot")
        self.zoom_button.Bind(wx.EVT_BUTTON, self.zoom_plot)
        self.second_sizer.Add(self.zoom_button, pos = (12, 5), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.zoom_button)

        self.to_end_button = wx.Button(self, id=wx.ID_ANY, label="end of footfall >")
        self.to_end_button.Bind(wx.EVT_BUTTON, lambda event, new_frame = 'end' : self.SwitchFrame(event, new_frame))
        self.second_sizer.Add(self.to_end_button, pos = (12, 6), span = (0,2), flag = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.to_end_button)

        self.save_val_button = wx.Button(self, id=wx.ID_ANY, label="Save (Ctrl-S)")
        self.Bind(wx.EVT_BUTTON, self.SaveValFunc, self.save_val_button)
        self.second_sizer.Add(self.save_val_button, pos = (13, 4), span = (0, 1), flag = wx.TOP | wx.BOTTOM, border = 15)
        self.second_sizer_widgets.append(self.save_val_button)

        accelerator_list = [(wx.ACCEL_NORMAL, wx.WXK_SPACE, self.validate_button.GetId()), 
                            (wx.ACCEL_NORMAL, ord('a'), self.prev_pred_button.GetId()), 
                            (wx.ACCEL_NORMAL, ord('d'), self.next_pred_button.GetId()),
                            (wx.ACCEL_CTRL, ord('s'), self.save_val_button.GetId())]
        self.accel_tbl = wx.AcceleratorTable(accelerator_list)
        self.SetAcceleratorTable(self.accel_tbl)


        self.restart_button = wx.Button(self, id=wx.ID_ANY, label="Analyze new file")
        self.restart_button.Bind(wx.EVT_BUTTON, self.DisplayFirstPage)
        self.second_sizer.Add(self.restart_button, pos = (13, 5), span = (0, 2), flag = wx.TOP | wx.BOTTOM, border = 15)
        self.second_sizer_widgets.append(self.restart_button)

        # display location graphs
        graph = FootfallFunctions.plot_labels(self.df, self.n_frame, self.method_selection, self.t_val, self.start_val, \
            self.end_val, (self.window_width-50) / 100, self.window_height / 100, self.bodypart, self.bodypart_list_val, self.selected_bodyparts, 'y', self.likelihood_threshold, self.confirmed)
        self.graph_canvas = FigureCanvas(self, -1, graph)
        self.second_sizer.Add(self.graph_canvas, pos = (14, 0), span = (1, 8), flag = wx.TOP | wx.LEFT, border = 25) 
        self.second_sizer_widgets.append(self.graph_canvas)       

        # display slider
        self.slider = wx.Slider(self, value=self.n_frame+1, minValue=1, maxValue=len(self.df),
            style=wx.SL_HORIZONTAL)
        self.slider.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        self.second_sizer.Add(self.slider, pos=(15, 0), span = (2, 8), flag = wx.LEFT | wx.EXPAND | wx.TOP | wx.RIGHT, border = 25)
        self.second_sizer_widgets.append(self.slider)

        self.likelihood_label = wx.StaticText(self, label = "Set likelihood threshold (data below threshold are labeled grey; between 0 and 1)")
        self.second_sizer.Add(self.likelihood_label, pos= (17, 0), span = (1, 3),flag = wx.LEFT | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.likelihood_label)

        self.likelihood_input = wx.TextCtrl(self, value = str(self.likelihood_threshold))
        self.second_sizer.Add(self.likelihood_input, span = (1, 1), pos= (17, 3), flag = wx.LEFT | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.likelihood_input)

        self.likelihood_button = wx.Button(self, id = wx.ID_ANY, label = "Update")
        self.likelihood_button.Bind(wx.EVT_BUTTON, self.OnLikelihood)
        self.second_sizer.Add(self.likelihood_button, span = (1, 1), pos = (17, 4), flag = wx.LEFT | wx.TOP, border = 25)
        self.second_sizer_widgets.append(self.likelihood_button)

        self.Fit()
        FootfallFunctions.ControlButton(self)
        FootfallFunctions.ControlPrediction(self)

        self.SetSizer(self.second_sizer)
        # self.Layout()
        self.GetParent().Layout()

    def onTest(self, event):
        print("refreshed!")

    def OnBodypartPlot(self, e):

        self.bodypart = self.bodypart_to_plot.GetValue()
        if self.n_frame in self.t_val:
            index = self.t_val.index(self.n_frame)
            self.bodypart_list_val[index] = self.bodypart

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self, set_bodypart=False)

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
        self.second_sizer_widgets = []
        self.has_imported_file = False
        self.video = None
        self.df = None
        self.filename = None
        self.bodyparts = None
        self.n_pred, self.depth_pred, self.t_pred, self.start_pred, self.end_pred, self.bodypart_list_pred = 0,[],[],[],[],[]
        self.n_val, self.depth_val, self.t_val, self.start_val, self.end_val, self.bodypart_list_val = 0,[],[],[],[],[]

        self.dirname = os.getcwd()
        
        self.first_sizer = wx.GridBagSizer(0, 0)

        self.header = wx.StaticText(self, -1, "Validate", size=(500,100))
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.header.SetFont(font)
        self.first_sizer.Add(self.header, pos = (0, 0), span = (2, 5), flag = wx.LEFT|wx.TOP, border = 25)

        self.instructions = wx.StaticText(self, -1, "Load the csv file of bodypart coordinates (e.g., from DLC) and validate detected footfalls.")
        self.first_sizer.Add(self.instructions, pos = (2, 0), span = (1, 3), flag = wx.LEFT|wx.TOP|wx.BOTTOM, border=25)
        self.first_sizer_widgets.append(self.instructions)
        self.SetSizer(self.first_sizer)

        self.FirstPage()
        self.Fit()
        self.GetParent().Layout()


    def DisplaySecondPage(self, e):

        self.SecondPage()
        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)

        self.GetParent().Layout()
        

    def OnMethod(self, e):

        self.selected_bodyparts = [self.bodyparts[i] for i in list(self.bodypart_choices.GetCheckedItems())]
        self.method_selection = self.method_choices.GetValue()
        if self.method_selection == 'Threshold': 
            if self.threshold == '':
                self.pred_text_extra = 'Using automatic threshold (mean + 1 SD of y coordinate for each bodypart). \n'
            else:
                self.pred_text_extra = f'Current threshold: {self.threshold} px. \n'
        else:
            self.pred_text_extra = ''

        try:
            self.MakePrediction(self)
            self.GetParent().Layout()
        except:
            # awaiting bodypart selection
            self.pred_text.SetLabel(f"\n{self.pred_text_extra}No footfalls detected! Try selecting a different bodypart or method?\n")
            # print(sys.exc_info())
            pass

    def OnBodypart(self, e):
        
        if self.method_selection == 'Threshold': 
            if self.threshold == '':
                self.pred_text_extra = 'Using automatic threshold (mean + 1 SD of y coordinate for each bodypart). \n'
            else:
                self.pred_text_extra = f'Current threshold: {self.threshold} px. \n'
        else:
            self.pred_text_extra = ''

        self.selected_bodyparts = [self.bodyparts[i] for i in list(self.bodypart_choices.GetCheckedItems())]
        try:
            self.MakePrediction(self)
            self.GetParent().Layout()
        except:
            # awaiting method selection
            self.pred_text.SetLabel(f"\n{self.pred_text_extra}No footfalls detected! Try selecting a different bodypart or method?\n")
            pass

    def OnLikelihood(self, e):

        self.likelihood_threshold = float(self.likelihood_input.GetValue())

        FootfallFunctions.ControlButton(self)
        FootfallFunctions.DisplayPlots(self)

        self.GetParent().Layout()

    def zoom_plot(self, e):
        # change status
        if not self.zoom:
            self.zoom = True
            self.zoom_button.SetLabel('Zoom out')
        else:
            self.zoom = False
            self.zoom_button.SetLabel('Zoom in')

        FootfallFunctions.DisplayPlots(self)
        self.GetParent().Layout()


    def zoom_frame(self, e):
        # change status
        if not self.zoom_image:
            self.zoom_image = True
            self.zoom_frame_button.SetLabel('Zoom out')
        else:
            self.zoom_image = False
            self.zoom_frame_button.SetLabel('Zoom in')

        FootfallFunctions.DisplayPlots(self)
        self.GetParent().Layout()


    def display_info(self, e):
        
        wx.MessageBox(f"Currently validating detected footfalls for \n\n{self.filename} \n\nand \n\n{self.video_name}",
                        "File information",
                        wx.OK|wx.ICON_INFORMATION)
