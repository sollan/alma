import wx
from wx.lib.stattext import GenStaticText as StaticText

class HeaderPanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        
        header = wx.StaticText(self, label="Slip detector")
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        header.SetFont(font)
        # self.header.Wrap(200)

        # create a sizer to manage the layout of child widgets
        self.HeaderSizer = wx.BoxSizer(wx.VERTICAL)
        self.HeaderSizer.Add(header, wx.SizerFlags().Border(wx.TOP|wx.LEFT, 35))
        self.SetSizer(self.HeaderSizer)
        # self.Layout()


class AnalyzePanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        
        header = wx.StaticText(self, -1, "Analyze", size=(500,100))
        font = wx.Font(20,wx.DEFAULT,wx.NORMAL,wx.NORMAL)
        # font = header.GetFont()
        # font.PointSize += 2
        # font = font.Bold()
        header.SetFont(font)

        # create a sizer to manage the layout of child widgets
        self.AnalyzeSizer = wx.BoxSizer(wx.VERTICAL)
        self.AnalyzeSizer.Add(header, wx.SizerFlags().Border(wx.TOP|wx.LEFT, 35))
        self.SetSizer(self.AnalyzeSizer)
        self.Layout()

    # def show_img(self):

    #     png = wx.Image('../top_40_train.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
    #     wx.StaticBitmap(self, -1, png, (10, 5), (png.GetWidth(), png.GetHeight()))



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

        
