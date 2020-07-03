import wx
from test_app import HomeFrame

class Example(HomeFrame):

    def __init__(self, *args, **kw):
        super(Example, self).__init__(*args, **kw)
        # self.makeMenuBar()
        self.InitUI()

    def InitUI(self):

        pnl = wx.Panel(self)

        self.sizer = wx.GridBagSizer(5, 5)

        sld = wx.Slider(pnl, value=200, minValue=150, maxValue=500,
                        style=wx.SL_HORIZONTAL)

        sld.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        self.sizer.Add(sld, pos=(0, 0), flag=wx.ALL|wx.EXPAND, border=25)

        self.txt = wx.StaticText(pnl, label='200')
        self.sizer.Add(self.txt, pos=(0, 1), flag=wx.TOP|wx.RIGHT, border=25)

        self.sizer.AddGrowableCol(0)
        pnl.SetSizer(self.sizer)

        self.SetTitle('wx.Slider')
        # self.Centre()

    def OnSliderScroll(self, e):

        obj = e.GetEventObject()
        val = obj.GetValue()

        self.txt.SetLabel(str(val))



# if __name__ == '__main__':
#     # When this module is run (not imported) then create the app, the
#     # frame, show it, and start the event loop.
#     app = wx.App(redirect = True)
#     frm = HomeFrame(None, title='Slip Detector', size=(1000,800))
#     frm.Show()
#     app.MainLoop()