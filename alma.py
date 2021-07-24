import wx
from Panels import Start, AnalyzeStride, AnalyzeFootfall, RandomForest, PCA


class HomeFrame(wx.Frame):

    def __init__(self, *args, **kw):
        
        super(HomeFrame, self).__init__(*args, **kw)

        self.StartPanel = Start.StartPanel(self)
        self.StartPanel.Hide()

        self.AnalyzeFootfallPanel = AnalyzeFootfall.AnalyzeFootfallPanel(self)
        self.AnalyzeFootfallPanel.Hide()

        self.AnalyzeStridePanel = AnalyzeStride.AnalyzeStridePanel(self)
        self.AnalyzeStridePanel.Hide()

        self.RandomForestPanel = RandomForest.RandomForestPanel(self)
        self.RandomForestPanel.Hide()

        self.PCAPanel = PCA.PCAPanel(self)
        self.PCAPanel.Hide()

        self.current_panel = self.StartPanel

        self.main_sizer = wx.GridBagSizer(0, 0)
        self.main_sizer.SetEmptyCellSize((0, 0))
        self.main_sizer.Add(self.current_panel, pos=(0, 0), span = (5, 5), flag=wx.EXPAND)
        self.SetSizer(self.main_sizer)

        self.makeMenuBar()
        self.CreateStatusBar()
        self.SetStatusText("Welcome!")

        self.current_panel.Show()
        self.Layout()


    def makeMenuBar(self):
        """
        more functions / menu items can be added
        for additional features
        """

        limb_motion_menu = wx.Menu()
        start_item = limb_motion_menu.Append(-1, "&Start \tCtrl-S")
        analyze_footfall_item = limb_motion_menu.Append(-1, "&Footfall detection\tCtrl-L",
                "Detect footfalls and validate results")
        analyze_stride_item = limb_motion_menu.Append(-1, "&Gait kinematic data extraction\tCtrl-K",
                "Extract gait kinematic parameters with bodypart coordinate data")
        #################################
        analysis_menu = wx.Menu()
        random_forest_item = analysis_menu.Append(-1, "&Random forest classification\tCtrl-R",
                "Random forest classification using extracted gait kinematic parameters")
        principal_component_analysis_item = analysis_menu.Append(-1, "&PCA\tCtrl-P",
                "PCA using extracted gait kinematic parameters")
        #################################
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT)
        help_item = help_menu.Append(wx.ID_HELP)
        #################################
        menu_bar = wx.MenuBar()
        menu_bar.Append(limb_motion_menu, "&Limb motion analysis")
        menu_bar.Append(analysis_menu, "&Data analysis")
        menu_bar.Append(help_menu, "&Help")
        self.SetMenuBar(menu_bar)        
        #################################
        self.Bind(wx.EVT_MENU, self.on_start, start_item)
        self.Bind(wx.EVT_MENU, self.on_analyze_footfall, analyze_footfall_item)
        self.Bind(wx.EVT_MENU, self.on_analyze_stride, analyze_stride_item)

        self.Bind(wx.EVT_MENU, self.on_random_forest, random_forest_item)
        self.Bind(wx.EVT_MENU, self.on_PCA, principal_component_analysis_item)

        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        self.Bind(wx.EVT_MENU, self.on_help, help_item)
    

    def on_start(self, e):
        
        self.current_panel.Hide()
        self.main_sizer.Replace(self.current_panel, self.StartPanel)

        self.SetSizer(self.main_sizer)

        self.current_panel = self.StartPanel
        self.current_panel.Show()
        self.SetStatusText('Welcome!')

        self.Layout()
        self.Refresh()
    

    def on_quit(self, e):

        self.Close(True)

    
    def on_analyze_footfall(self, e):

        self.current_panel.Hide()

        self.main_sizer.Replace(self.current_panel, self.AnalyzeFootfallPanel)
        self.SetSizer(self.main_sizer)

        self.current_panel = self.AnalyzeFootfallPanel
        self.current_panel.Show()

        self.SetStatusText('Ready for automated footfall analysis')
        self.Layout()
        self.Refresh() # refresh to show slider in right proportion


    def on_analyze_stride(self, e):

        self.current_panel.Hide()

        self.main_sizer.Replace(self.current_panel, self.AnalyzeStridePanel)
        self.SetSizer(self.main_sizer)

        self.current_panel = self.AnalyzeStridePanel
        self.current_panel.Show()

        self.SetStatusText('Ready for automated kinematic analysis')
        self.Layout()
        self.Refresh()


    def on_random_forest(self, e):

        self.current_panel.Hide()

        self.main_sizer.Replace(self.current_panel, self.RandomForestPanel)
        self.SetSizer(self.main_sizer)

        self.current_panel = self.RandomForestPanel
        self.current_panel.Show()

        self.SetStatusText('Run random forest classification on extracted gait kinematic parameters')
        self.Layout()
        self.Refresh() # refresh to show slider in right proportion


    def on_PCA(self, e):

        self.current_panel.Hide()

        self.main_sizer.Replace(self.current_panel, self.PCAPanel)
        self.SetSizer(self.main_sizer)

        self.current_panel = self.PCAPanel
        self.current_panel.Show()

        self.SetStatusText('Run PCA on extracted gait kinematic parameters')
        self.Layout()
        self.Refresh() # refresh to show slider in right proportion


    def on_about(self, e):
        
        wx.MessageBox( "This is a toolbox for fully automated (rodent) "\
                        "limb motion analysis with markerless bodypart tracking data. "\
                        "Compatible with csv output from DeepLabCut.",
                        "About ALMA",
                        wx.OK|wx.ICON_INFORMATION)

    def on_help(self, event):

        wx.MessageBox("Please go to our GitHub Wiki for help or email us :)",
                        "Support",
                        wx.OK|wx.ICON_INFORMATION)


if __name__ == '__main__':
    
    from Functions import ConfigFunctions

    configs = ConfigFunctions.load_config('./config.yaml')
    window_width, window_height = configs['window_width'], configs['window_height']

    app = wx.App(redirect = True)
    home_frame = HomeFrame(None, title='ALMA - Automated Limb Motion Analysis', size=(window_width, window_height))
    home_frame.Show()
    app.MainLoop()