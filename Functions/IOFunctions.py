import pandas as pd
import wx



'''
move file import / export related functions from SlipFunctions and KinematicsFunctions
here and edit the related usage in AnalyzeStride and ValidateSlips (probably reusing most
of them)
'''

def read_file(file):
    
    pd_dataframe = pd.read_csv(file, header=[1,2])
    # filename = file.split('/')[-1].split('_')
    # filename = filename[0] + ' ' + filename[1] + ' ' + filename[2]
    filename = file.split('/')[-1]
    return pd_dataframe, filename




def ImportKinematicsCSV(Panel, e):
    if not TEST:
        import_dialog = wx.FileDialog(self, 'Choose a file', self.dirname, '', 'CSV files (*.csv)|*.csv|All files(*.*)|*.*', wx.FD_OPEN)

        if import_dialog.ShowModal() == wx.ID_OK:
            self.csv_dirname = import_dialog.GetDirectory()
            self.filename = os.path.join(self.csv_dirname, import_dialog.GetFilename())
            self.df, self.filename = read_file(self.filename)
            self.df, self.bodyparts = ValidateFunctions.fix_column_names(self.df)
            # self.filtered_df = ValidateFunctions.filter_predictions(self.df, self.bodyparts[0], self.threshold)
    try: 
        if self.df is not None:
            self.has_imported_file = True
            self.import_csv_text.SetLabel(f"File imported! \n\n{self.filename}\n")
            self.GetParent().Layout()

            self.method_label.Show()

            self.method_choices.SetSelection(0)
            self.method_selection = self.method_choices.GetValue()
            self.method_choices.Show()

            # self.bodypart_label.Show()
            # self.bodypart_choices.Hide()

            # bodypart_choices = wx.CheckListBox(self, choices = self.bodyparts)
            # self.first_sizer.Replace(self.bodypart_choices, bodypart_choices)
            # self.bodypart_choices = bodypart_choices
            # # self.bodypart_choices.SetCheckedItems([0])
            # # self.bodypart = self.bodyparts[list(self.bodypart_choices.GetCheckedItems())[0]]
            # self.first_sizer_widgets.append(self.bodypart_choices)
            # self.bodypart_choices.Bind(wx.EVT_CHECKLISTBOX, self.OnBodypart)
            # self.bodypart_choices.Show()

            self.ExtractStrides(self)

            self.GetParent().Layout()

    except AttributeError:
        # user cancelled file import in pop up
        pass