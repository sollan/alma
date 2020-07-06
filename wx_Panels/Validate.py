import wx
from wx.lib.stattext import GenStaticText as StaticText
from Functions import ValidateFunctions
import os


class ValidatePanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        self.dirname = os.getcwd()

        # create a sizer to manage the layout of child widgets
        self.SliderSizer = wx.GridBagSizer(10, 10)

        self.import_button = wx.Button(self, id=wx.ID_ANY, label="Press Me")
        self.SliderSizer.Add(self.import_button, pos = (1, 0), flag = wx.ALL, border=25)
        self.import_button.Bind(wx.EVT_BUTTON, self.ImportFunc)

        self.bodypart = 'HR'
        self.threshold = 0.4

        # self.SliderSizer.Add(self.import_button, 0, wx.ALIGN_CENTER) 

        self.slider = wx.Slider(self, value=200, minValue=1, maxValue=2000,
                        style=wx.SL_HORIZONTAL)
        self.slider.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        self.SliderSizer.Add(self.slider, pos=(2, 0), flag=wx.ALL|wx.EXPAND, border=5)

        self.txt = wx.StaticText(self, label='300')
        self.SliderSizer.Add(self.txt, pos=(2, 1), flag=wx.TOP|wx.RIGHT, border=5)

        self.SliderSizer.AddGrowableCol(0)
        self.SetSizer(self.SliderSizer)

        self.SetLabel('Validate')
        # self.Centre()
        # self.Layout()

    def OnSliderScroll(self, e):

        obj = e.GetEventObject()
        val = obj.GetValue()

        self.txt.SetLabel(str(val))


    def ImportFunc(self, e):
        
        dlg=wx.FileDialog(self, 'Choose a file', self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.dirname = dlg.GetDirectory()
            self.filename = os.path.join(self.dirname, dlg.GetFilename())
            self.df, self.filename = ValidateFunctions.read_file(self.filename)
            self.df = ValidateFunctions.fix_column_names(self.df)
            self.df = ValidateFunctions.filter_predictions(self.df, self.bodypart, self.threshold)

    



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