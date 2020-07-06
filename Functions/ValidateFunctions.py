import pandas as pd
import os


def read_file(file):
    
    pd_dataframe = pd.read_csv(file, header=[1,2])
    filename = file.split('/')[-1].split('_')
    filename = filename[0] + ' ' + filename[1] + ' ' + filename[2]
    return pd_dataframe, filename


def fix_column_names(pd_dataframe):
    
    header_1 = pd_dataframe.columns.get_level_values(0)
    header_2 = pd_dataframe.columns.get_level_values(1)
    col_names = []

    for i in range(len(header_1)):
        col_names.append(' '.join([header_1[i], header_2[i]]))
        
    pd_dataframe.columns = col_names
    
    return pd_dataframe


def filter_predictions(pd_dataframe, bodyparts, threshold):
    
    if type(bodyparts) is list and len(bodyparts) > 1:
        for bodypart in bodyparts:
            pd_dataframe = pd_dataframe[pd_dataframe[bodypart + ' likelihood'] >= threshold]
    
    elif type(bodyparts) is list and len(bodyparts) == 1:
        pd_dataframe = pd_dataframe[pd_dataframe[bodyparts[0] + ' likelihood'] >= threshold]
    
    elif type(bodyparts) is str:
        pd_dataframe = pd_dataframe[pd_dataframe[bodyparts + ' likelihood'] >= threshold]
        
    # raise error if any bodypart name not identical as in csv
        
    return pd_dataframe