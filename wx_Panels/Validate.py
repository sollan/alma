import wx
from wx.lib.stattext import GenStaticText as StaticText
from Functions import ValidateFunctions
# from wx_Panels import Validate
import os


class ValidatePanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

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

        self.import_button = wx.Button(self, id=wx.ID_ANY, label="Import")
        self.import_button.Bind(wx.EVT_BUTTON, self.ImportFunc)
        self.sizer.Add(self.import_button, pos = (8, 0), flag = wx.LEFT, border = 25)

        self.import_text = wx.StaticText(self, label = "Import a csv file for validation. " + "\nThe file should be DeepLabCut output (or of a similar format) to ensure proper parsing!")
        self.sizer.Add(self.import_text, pos=(8, 1), flag = wx.LEFT, border = 25)




        self.bodypart = 'HR' # modify this to include user selection
        self.threshold = 0.4 # modify this to include user selection


        self.SetSizer(self.sizer)

        # self.SetLabel('Validate')
        # self.Centre()
        # self.Layout()



    def ImportFunc(self, e):
        
        import_dialog = wx.FileDialog(self, 'Choose a file', self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', wx.FD_OPEN)

        if import_dialog.ShowModal() == wx.ID_OK:
            self.dirname = import_dialog.GetDirectory()
            self.filename = os.path.join(self.dirname, import_dialog.GetFilename())
            self.df, self.filename = ValidateFunctions.read_file(self.filename)
            self.df = ValidateFunctions.fix_column_names(self.df)
            self.df = ValidateFunctions.filter_predictions(self.df, self.bodypart, self.threshold)
        if self.df is not None:
            self.has_imported_file = True
            self.import_text.SetLabel("File imported! ")
            self.MakePrediction(self)


    def MakePrediction(self, e):

        assert self.df is not None, "Cannot read file. Try another one?"

        n_pred, depth_pred, t_pred, start_pred, end_pred = ValidateFunctions.find_slips(self.df, self.bodypart) 
        self.n_pred, self.depth_pred, self.t_pred, self.start_pred, self.end_pred = n_pred, depth_pred, t_pred, start_pred, end_pred
        self.pred_text = wx.StaticText(self, label = f"The algorithm predicted {self.n_pred} slips with an average depth of {self.depth_pred:.2f} pixels. \nYou can validate the prediction now.")
        self.sizer.Add(self.pred_text, pos= (9, 1) , flag = wx.ALL, border = 25)
        self.SetSizer(self.sizer)

        self.validate_button = wx.Button(self, id=wx.ID_ANY, label="Validate")
        self.validate_button.Bind(wx.EVT_BUTTON, self.ValidateFunc)
        self.sizer.Add(self.validate_button, pos = (10, 1), flag = wx.LEFT, border = 25)

        self.save_pred_button = wx.Button(self, id=wx.ID_ANY, label="Export Prediction")
        self.save_pred_button.Bind(wx.EVT_BUTTON, self.SavePredFunc)
        self.sizer.Add(self.save_pred_button, pos = (10, 2), flag = wx.LEFT, border = 25)

        self.Layout()

        #################################
        # add button to go to validation
        #  & refresh view
        #################################


    def ValidateFunc(self, e):

        import matplotlib as mpl
        ## Import the matplotlib backend for wxPython
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
        import numpy as np
        self.import_button.Hide()
        self.pred_text.Hide()

        x = np.arange(10)
        y = np.arange(10)
        ## This is to work with matplotlib inside wxPython
        ## I prefer to use the object oriented API
        self.figure  = mpl.figure.Figure(figsize=(15, 5))
        self.axes    = self.figure.add_subplot(111)
        self.canvas  = FigureCanvas(self, -1, self.figure)
        ## This is to avoid showing the plot area. When set to True
        ## clicking the button show the canvas and the canvas covers the button
        ## since the canvas is placed by default in the top left corner of the window
        # self.canvas.Show(False)
        self.axes.scatter(x, y)
        # self.canvas.Show(True)

        # self.png = wx.Image('../top_40_train.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        # self.sizer.Add(wx.StaticBitmap(self, -1, self.png, (10, 5), (self.png.GetWidth(), self.png.GetHeight())), pos = (0, 1))

        # wx.MessageBox("This function is still under development. Thanks for your patience! :)")


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

                
    # def Validate(self, e):
        
    #     if self.df is not None:
    #         self.import_text.SetLabel("Import another csv file for validation")

    #     self.Hide()
    #     home_frame.ValidatePanel = Validate.ValidatePanel(self)
    #     home_frame.ValidatePanel.Show()

    



    #################################
    # import csv file (analyzed)  
    # (automatically find csv output from same session?)
    # --> generate plots
    # predict peaks (start with baseline correction & scipy find peak)
    # display axis plot, current frame, and frame with opencv
    # slider to adjust frames
    # tick box to select slip (start with slip, onset / end for future
    # duration calculations)
    # confirm / finish button
    # export option to save manual labels as csv
    #################################


    # def show_img(self):

    #     png = wx.Image('../top_40_train.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
    #     wx.StaticBitmap(self, -1, png, (10, 5), (png.GetWidth(), png.GetHeight()))

    #     self.has_imported_file = True
    #     self.sizer = wx.GridBagSizer(0, 0)

    #     self.header = wx.StaticText(self, -1, "Validate")
    #     font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
    #     self.header.SetFont(font)
    #     self.sizer.Add(self.header, pos = (0, 0), flag = wx.LEFT|wx.TOP, border = 25)

    #     self.slider = wx.Slider(self, value=200, minValue=1, maxValue=2000,
    #                     style=wx.SL_HORIZONTAL)
    #     self.slider.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
    #     self.sizer.Add(self.slider, pos=(1, 0), flag=wx.ALL|wx.EXPAND, border=5)

    #     self.txt = wx.StaticText(self, label='300')
    #     self.sizer.Add(self.txt, pos=(1, 1), flag=wx.TOP|wx.RIGHT, border=5)

    #     self.sizer.AddGrowableCol(0)
    #     self.SetSizer(self.sizer)



    # def OnSliderScroll(self, e):

    #     obj = e.GetEventObject()
    #     val = obj.GetValue()

    #     self.txt.SetLabel(str(val))