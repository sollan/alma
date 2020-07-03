import wx
from wx.lib.stattext import GenStaticText as StaticText


class StartPanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        
        header = wx.StaticText(self, label="Slip detector")
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        header.SetFont(font)
        # self.header.Wrap(200)

        #################################
        # add some instruction text
        
        self.StartSizer = wx.BoxSizer(wx.VERTICAL)
        self.StartSizer.Add(header, wx.SizerFlags().Border(wx.TOP|wx.LEFT, 35))
        self.SetSizer(self.StartSizer)