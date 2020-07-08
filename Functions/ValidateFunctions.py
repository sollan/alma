import pandas as pd
from scipy.signal import peak_widths, find_peaks
import numpy as np
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


def find_slips(pd_dataframe, bodypart, **kwargs): 
        
    t_peaks, properties = find_peaks(-pd_dataframe['%s y'%bodypart], height=-5000, prominence=(10,100000))
    # t_peaks, properties = find_peaks(pd_dataframe, prominence=0, distance=18, height=-10, width = 0)
    # width_half = peak_widths(data, t_peaks, rel_height=0.5)
    
#         t_peaks, properties = find_peaks(-data, prominence=(10,100000), height=-5000, width = 0)
#         width_half = peak_widths(-data, t_peaks, rel_height=0.5)
    
    index = pd_dataframe['bodyparts coords'].iloc[:]
    
    is_peak = np.zeros(len(index))
    n_peaks = 0
    current_data = pd_dataframe.iloc[0]
    norm = np.max(pd_dataframe['%s y'%bodypart])
    std = np.std(pd_dataframe['%s y'%bodypart])
    
    for i in range(len(is_peak)):
        
        if i in t_peaks:
            is_peak[i] = norm-std*4
            n_peaks += 1
        
        else:
            is_peak[i] = norm-std*2
            
        current_data = pd_dataframe.iloc[i]
        
        h_peaks = np.mean(properties["prominences"])
        start_times = properties['left_bases']
        end_times = properties['right_bases']
        
    return n_peaks, h_peaks, t_peaks, start_times, end_times


def make_output(pathname, t_slips, depth_slips, start_slips, end_slips):

    df_output = pd.DataFrame({'time': t_slips,\
                            'depth': depth_slips,\
                            'start': start_slips,\
                            'end': end_slips})

    df_output.to_csv(pathname, index = False)