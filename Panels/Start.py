import wx
from Functions import ConfigFunctions
import numpy as np


class StartPanel(wx.Panel):

    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        self.sizer = wx.GridBagSizer(0, 0)

        #################################
        configs = ConfigFunctions.load_config('./config.yaml')
        self.window_width = configs['window_width']
        self.window_height = configs['window_height']

        self.header = wx.StaticText(
            self, -1, "Automated Limb Motion Analysis (ALMA)",
            size=(self.window_width, 60))
        font = wx.Font(30, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.header.SetFont(font)
        self.sizer.Add(
            self.header,
            pos=(1, 0), span=(1, 2), flag=wx.ALL, border=25)

        self.intro_text = wx.StaticText(self,
            label="Select behavioral test to analyze:\n")
        font = wx.Font(15, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.intro_text.SetFont(font)
        
        ladder_rung_img = wx.Bitmap('./Resources/image_ladder_rung.png')
        w, h = ladder_rung_img.GetWidth(), ladder_rung_img.GetHeight()
        ladder_rung_img = wx.Bitmap.ConvertToImage(ladder_rung_img)
        ladder_rung_img.Rescale(
            w//(3000//self.window_width),
            h//(3000//self.window_width))
        ladder_rung_img = wx.Bitmap(ladder_rung_img)
        self.ladder_rung_button = wx.BitmapButton(
            self, id=wx.ID_ANY, bitmap=ladder_rung_img)

        kinematics_img = wx.Bitmap('./Resources/image_kinematics.png')
        kinematics_img = wx.Bitmap.ConvertToImage(kinematics_img)
        kinematics_img.Rescale(
            w//(3000//self.window_width),
            h//(3000//self.window_width))
        kinematics_img = wx.Bitmap(kinematics_img)
        self.kinematics_button = wx.BitmapButton(
            self, id=wx.ID_ANY, bitmap=kinematics_img)

        self.sizer.Add(
            self.intro_text, pos=(2, 0), flag=wx.ALL, border=25)
        self.sizer.Add(
            self.ladder_rung_button, pos=(3, 0), flag=wx.ALL, border=25)
        self.sizer.Add(
            self.kinematics_button, pos=(3, 1), flag=wx.ALL, border=25)
        # self.sizer.Add(control)

        self.ladder_rung_button.Bind(wx.EVT_BUTTON, parent.on_analyze_footfall)
        self.kinematics_button.Bind(wx.EVT_BUTTON, parent.on_analyze_stride)

        self.SetSizer(self.sizer)
        self.GetParent().Layout()
