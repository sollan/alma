import wx
from Panels import Start, AnalyzeStride, ValidateFootfalls


class HomeFrame(wx.Frame):


    def __init__(self, *args, **kw):
        

        super(HomeFrame, self).__init__(*args, **kw)


        self.StartPanel = Start.StartPanel(self)
        self.StartPanel.Hide()

        # self.AnalyzeFootfallPanel = AnalyzeFootfall.AnalyzeFootfallPanel(self)
        # self.AnalyzeFootfallPanel.Hide()
        
        self.ValidateFootfallPanel = ValidateFootfalls.ValidateFootfallPanel(self)
        self.ValidateFootfallPanel.Hide()

        self.AnalyzeStridePanel = AnalyzeStride.AnalyzeStridePanel(self)
        self.AnalyzeStridePanel.Hide()

        self.current_panel = self.StartPanel


        # configure UI organization
        self.main_sizer = wx.GridBagSizer(0, 0)
        self.main_sizer.SetEmptyCellSize((0, 0))
        self.main_sizer.Add(self.current_panel, pos=(0, 0), span = (5, 5), flag=wx.EXPAND)
        self.SetSizer(self.main_sizer)


        self.makeMenuBar()

        # and a status bar
        self.CreateStatusBar()
        self.SetStatusText("Footfall detector ready for new job")

        self.current_panel.Show()


    def makeMenuBar(self):
        """
        more functions / menu items can be added
        for additional features
        """
        
        action_menu = wx.Menu()
        start_item = action_menu.Append(-1, "&Start page \tCtrl-S")
        # analyze_item = action_menu.Append(-1, "&Analyze footfall data...\tCtrl-N",
        #         "Predict footfalls based on csv")
        validate_item = action_menu.Append(-1, "&Validate footfall predictions...\tCtrl-L",
                "Manually validate and correct detected footfalls")
        analyze_stride_item = action_menu.Append(-1, "&Analyze kinematics / stride data...\tCtrl-K",
                "Extract strides and kinematics parameters from csv")
        # validate_stride_item = action_menu.Append(-1, "&Validate footfall predictions...\tCtrl-L",
        #         "Manually validate and correct detected footfalls")


        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT)
        help_item = help_menu.Append(wx.ID_HELP)
        # contact_item = help_menu.Append(-1, "&Contact us...\tCtrl-O",
                # "Contact us for suggestions and support")

        # configure menu_bar

        menu_bar = wx.MenuBar()
        # menu_bar.Append(file_menu, "&File")
        menu_bar.Append(action_menu, "&Action")
        # menu_bar.Append(statistics_menu, "&Statistics")
        menu_bar.Append(help_menu, "&Help")

        self.SetMenuBar(menu_bar)
        
        # configure events for menu items
        self.Bind(wx.EVT_MENU, self.on_start, start_item)
        # self.Bind(wx.EVT_MENU, self.on_quit, quit_item)
        # self.Bind(wx.EVT_MENU, self.on_analyze_footfall, analyze_footfall_item)
        self.Bind(wx.EVT_MENU, self.on_validate, validate_item)
        self.Bind(wx.EVT_MENU, self.on_analyze_stride, analyze_stride_item)
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        self.Bind(wx.EVT_MENU, self.on_help, help_item)
        # self.Bind(wx.EVT_MENU, self.on_contact, contact_item)
    
    def on_start(self, event):
        
        self.current_panel.Hide()

        self.main_sizer.Replace(self.current_panel, self.StartPanel)
        self.SetSizer(self.main_sizer)

        self.current_panel = self.StartPanel
        self.current_panel.Show()

        self.SetStatusText('Welcome!')
        self.Layout()
        self.Refresh()
    

    def on_quit(self, event):

        self.Close(True)


    # def on_analyze_footfall(self, event):
        
    #     self.current_panel.Hide()

    #     self.main_sizer.Replace(self.current_panel, self.AnalyzeFootfallPanel, pos=(0, 0), span = (5, 5), flag=wx.EXPAND)
    #     self.SetSizer(self.main_sizer)

    #     self.current_panel = self.AnalyzeFootfallPanel
    #     self.current_panel.Show()

    #     self.SetStatusText('Footfall detector ready for new job')
    #     self.Layout()
    #     self.Refresh()

    
    def on_validate(self, event):

        self.current_panel.Hide()

        self.main_sizer.Replace(self.current_panel, self.ValidateFootfallPanel)
        self.SetSizer(self.main_sizer)

        self.current_panel = self.ValidateFootfallPanel
        self.current_panel.Show()

        self.SetStatusText('Ready for automated footfall analysis')
        self.Layout()
        self.Refresh() # refresh to show slider in right proportion

    def on_analyze_stride(self, event):

        self.current_panel.Hide()

        self.main_sizer.Replace(self.current_panel, self.AnalyzeStridePanel)
        self.SetSizer(self.main_sizer)

        self.current_panel = self.AnalyzeStridePanel
        self.current_panel.Show()

        self.SetStatusText('Ready for automated kinematic analysis')
        self.Layout()
        self.Refresh()


    def on_about(self, event):
        
        wx.MessageBox( "This is a toolbox for fully automated (rodent) "\
                        "limb motion analysis with markerless bodypart tracking data. "\
                        "Compatible with csv output from DeepLabCut.",
                        "About ALMA",
                        wx.OK|wx.ICON_INFORMATION)

    def on_help(self, event):

        # wx.MessageBox("This function is still under development. Thanks for your patience! :)")
        wx.MessageBox("Please go to our GitHub Wiki for help or email us :)",
                        "Support",
                        wx.OK|wx.ICON_INFORMATION)



    # def on_contact(self, event):

    #     wx.MessageBox("This function is still under development. Thanks for your patience! :)")




if __name__ == '__main__':
    
    from Functions import ConfigFunctions

    configs = ConfigFunctions.load_config('./config.yaml')
    window_width, window_height = configs['window_width'], configs['window_height']

    app = wx.App(redirect = True)
    home_frame = HomeFrame(None, title='ALMA - Automated Limb Motion Analysis', size=(window_width, window_height))
    home_frame.Show()
    app.MainLoop()