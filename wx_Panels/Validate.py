import wx
from wx.lib.stattext import GenStaticText as StaticText


class ValidatePanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # create a sizer to manage the layout of child widgets
        self.SliderSizer = wx.GridBagSizer(600, 100)

        self.sld = wx.Slider(self, value=200, minValue=1, maxValue=2000,
                        style=wx.SL_HORIZONTAL)
        self.sld.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        self.SliderSizer.Add(self.sld, pos=(0, 0), flag=wx.ALL|wx.EXPAND, border=15)

        self.txt = wx.StaticText(self, label='300')
        self.SliderSizer.Add(self.txt, pos=(0, 1), flag=wx.TOP|wx.RIGHT, border=15)

        self.SliderSizer.AddGrowableCol(0)
        self.SetSizer(self.SliderSizer)

        self.SetLabel('Validate')
        # self.Centre()
        # self.Layout()

    def OnSliderScroll(self, e):

        obj = e.GetEventObject()
        val = obj.GetValue()

        self.txt.SetLabel(str(val))



    #################################
    # --> import csv file (analyzed) 
    # (automatically find csv output from same session?)
    # generate plots
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