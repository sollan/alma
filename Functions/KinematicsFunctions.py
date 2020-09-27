import pandas as pd
from scipy.signal import peak_widths, find_peaks
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import cv2
import wx



def test(use_preset = True):
    '''
    replace comments with local file paths
    to speed up testing
    '''
    # filename = '/home/annette/Desktop/DeepLabCut/ladder rung results/Irregular_347_21dpi_croppedDLC_resnet50_Ladder RungMay12shuffle1_500000.csv'
    df, filename = read_file(filename)
    df, bodyparts = fix_column_names(df)
    # video = '/home/annette/Desktop/DeepLabCut/ladder rung results/Irregular_347_21dpi_cropped.avi'
    # video_name = 'Irregular_347_21dpi_cropped.avi'

    return filename, df, bodyparts, video, video_name


def read_file(file):
    
    pd_dataframe = pd.read_csv(file, header=[1,2])
    # filename = file.split('/')[-1].split('_')
    # filename = filename[0] + ' ' + filename[1] + ' ' + filename[2]
    filename = file.split('/')[-1]
    return pd_dataframe, filename


def fix_column_names(pd_dataframe):
    
    header_1 = pd_dataframe.columns.get_level_values(0)
    header_2 = pd_dataframe.columns.get_level_values(1)
    col_names = []
    bodyparts = []

    for i in range(len(header_1)):
        col_names.append(' '.join([header_1[i], header_2[i]]))

    for column in col_names:
        if column.endswith('y'):
            bodyparts.append(column.split(' ')[0])
        
    pd_dataframe.columns = col_names
    
    return pd_dataframe, bodyparts


def treadmill_correction(pd_dataframe, bodyparts, pixels_per_frame = 8.09):

    correction = np.arange(0, len(pd_dataframe), 1)
    correction = correction * pixels_per_frame # should be determined manually

    if type(bodyparts) is list and len(bodyparts) > 1:
        for bodypart in bodyparts:
            pd_dataframe[f'{bodypart} x'] = pd_dataframe[f'{bodypart} x'] + correction
    
    elif type(bodyparts) is list and len(bodyparts) == 1:
        pd_dataframe[f'{bodyparts[0]} x'] = pd_dataframe[f'{bodyparts[0]} x'] + correction
    
    elif type(bodyparts) is str:
        pd_dataframe[f'{bodyparts} x'] = pd_dataframe[f'{bodyparts} x'] + correction

    return pd_dataframe


def find_strides(pd_dataframe, bodypart, method = 'Rate of change', rolling_window = None, 
                 change_threshold = None, treadmill_speed = None, frame_rate = 1, threshold = None, **kwargs):

    start_times = []
    end_times = []
    durations = []

    if method == "Rate of change":
        axis = 'x'

        mean_bodypart_x = pd_dataframe[f'{bodypart} {axis}'].rolling(rolling_window).mean() if rolling_window is not None else pd_dataframe[f'{bodypart} {axis}']
        mean_bodypart_x_change = np.diff(mean_bodypart_x)

        is_stance = [i <= change_threshold for i in mean_bodypart_x_change]

        if not is_stance[0]:
            start_times.append(0)

        for i in range(1, len(is_stance)):
            if is_stance[i] != is_stance[i-1]: # change of stance / swing status
                if not is_stance[i]: # from stance to not stance
                    end_times.append(i-1)
                    start_times.append(i)
                # else: # from not stance to stance
                    # end_times.append(i)
                    # pass
        if is_stance[-1]:
            end_times.append(len(is_stance))

    elif method == 'Threshold':
        '''
        when there is a clear cut off pixel value for the entire video, 
        e.g. location of treadmill in frame
        '''
        axis = 'y'

        if threshold is None:  
            threshold = np.mean(pd_dataframe[f'{bodypart} {axis}'])
        # print(threshold)
        # pd_dataframe[f'{bodypart} {axis}'] = pd_dataframe[f'{bodypart} {axis}'].rolling(rolling_window).mean() if rolling_window is not None else pd_dataframe[f'{bodypart} {axis}']

        # y axis loc larger than threshold == limb touching treadmill == end of stride        
        on_treadmill = [i >= threshold for i in pd_dataframe[f'{bodypart} {axis}']]
        # print(on_treadmill)
        if not on_treadmill[0]:
            start_times.append(0)
        for i in range(1, len(on_treadmill)):
            if on_treadmill[i] != on_treadmill[i-1]: # change of stance / swing status
                if not on_treadmill[i]: # from not stance to stance
                    # end_times.append(i)
                    # pass
                # else: # from stance to not stance
                    end_times.append(i-1)
                    start_times.append(i)
        if on_treadmill[-1]:
            end_times.append(len(on_treadmill))

    durations = np.array(end_times) - np.array(start_times)
    # print(start_times, end_times, durations)
    return list(start_times), list(end_times), list(durations)



def make_output(pathname, start_times, end_times, durations):

    df_output = pd.DataFrame({'stride_start': start_times,
                            'stride_end': end_times,
                            'stride_duration': durations})

    df_output.to_csv(pathname, index = False)


# def load_video(filename):

#     vidcap = cv2.VideoCapture(filename)
#     return vidcap
