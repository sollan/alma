import wx
import Panels
import test_frame

class HomeFrame(wx.Frame):


    def __init__(self, *args, **kw):
        
        super(HomeFrame, self).__init__(*args, **kw)

        self.HeaderPanel = Panels.HeaderPanel(self)
        # self.myGridSizer = wx.GridBagSizer(1,1)
        self.sizer = wx.GridBagSizer(5, 5)
        self.SetSizer(self.sizer)
        # self.Centre()
        # self.myGridSizer.SetEmptyCellSize((0,0))
        # self.myGridSizer.Add(self.HeaderPanel, pos=(0, 0), span=(4,8), flag=wx.EXPAND)
        # self.SetSizer(self.myGridSizer)
        btn = wx.Button(self, label="go to other frame")
        self.Bind(wx.EVT_BUTTON, self.on_button_click)
        # create a menu bar
        self.makeMenuBar()
        # and a status bar
        # self.CreateStatusBar()
        # self.SetStatusText("Slip detector ready for new job")
        # self.Show()


    def on_button_click(self, event):
        
        self.other_frame.Show()
        self.Destroy()




    def makeMenuBar(self):
        """
        more functions / menu items can be added
        for additional features
        """

        ################################################################

        file_menu = wx.Menu()      

        import_item = file_menu.Append(-1, "&Import file...\tCtrl-I",
                "Help string shown in status bar for this menu item")
        save_item = file_menu.Append(-1, "&Save...\t",
                "Save analysis result")
        save_as_item = file_menu.Append(-1, "&Save as...\tCtrl-S",
                "Save analysis result as new file")

        file_menu.AppendSeparator() 
        #-----------------------------------

        # When using a stock ID we don't need to specify the menu item's
        # label
        quit_item = file_menu.Append(wx.ID_EXIT)

        ################################################################
        
        action_menu = wx.Menu()
        analyze_item = action_menu.Append(-1, "&Analyze...\tCtrl-N",
                "Predict slips based on csv")
        validate_item = action_menu.Append(-1, "&Validate...\tCtrl-L",
                "Manually validate and correct detected slips")

        ################################################################

        statistics_menu = wx.Menu()
        t_test_item = statistics_menu.Append(-1, "&t-test...\tCtrl-T",
                "Apply t-test to slip statistics")
        anova_item = statistics_menu.Append(-1, "&ANOVA...\tCtrl-A",
                "Apply ANOVA to slip statistics")

        ################################################################

        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT)
        help_item = help_menu.Append(wx.ID_HELP)
        contact_item = help_menu.Append(-1, "&Contact us...\tCtrl-O",
                "Contact us for suggestions and support")

        # configure menu_bar

        menu_bar = wx.MenuBar()
        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(action_menu, "&Action")
        menu_bar.Append(statistics_menu, "&Statistics")
        menu_bar.Append(help_menu, "&Help")

        self.SetMenuBar(menu_bar)

if __name__ == '__main__':
    # When this module is run (not imported) then create the app, the
    # frame, show it, and start the event loop.
    app = wx.App(redirect = True)
    home_frame = HomeFrame(None, title='Slip Detector', size=(1000,800))
    home_frame.Show()
    other_frame = test_frame.Example(None, title = 'Slip Detector', size = (1000,800))
    home_frame.other_frame = other_frame
    other_frame.other_frame = home_frame
    app.MainLoop()