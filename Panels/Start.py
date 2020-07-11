import wx
from wx.lib.stattext import GenStaticText as StaticText


class StartPanel(wx.Panel):


    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        self.sizer = wx.GridBagSizer(0, 0)
        
        self.header = wx.StaticText(self, -1, "Slip detector", size=(400,60))
        font = wx.Font(20,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.header.SetFont(font)

        #################################
        # add some instruction text
        
        self.sizer.Add(self.header, pos = (0, 0), flag = wx.ALL, border = 25)

        self.intro_text = wx.StaticText(self, 
            label = "Completed functions:\n" +
                    "1. Action - Validate: Predict slips and save prediction as file\n" +
                    "2. Action - Validate: Display original video and browse frames")
        
        self.sizer.Add(self.intro_text, pos= (1, 0) , flag = wx.ALL, border = 25)
        self.SetSizer(self.sizer)