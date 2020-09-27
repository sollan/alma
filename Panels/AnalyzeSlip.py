import wx
from wx.lib.stattext import GenStaticText as StaticText


class AnalyzeSlipPanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        
        header = wx.StaticText(self, -1, "Behavioral Analysis (slip during ladder rung)", size=(500,100))
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        header.SetFont(font)

        ################################
        # integrate deeplabcut functions (analyze video)
        # produce csv files

        self.AnalyzeSlipSizer = wx.BoxSizer(wx.VERTICAL)
        self.AnalyzeSlipSizer.Add(header, wx.SizerFlags().Border(wx.TOP|wx.LEFT, 35))
        self.SetSizer(self.AnalyzeSlipSizer)
        self.Layout()