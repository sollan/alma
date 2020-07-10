import pandas as pd
from scipy.signal import peak_widths, find_peaks
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import cv2

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


def filter_predictions(pd_dataframe, bodypart, threshold):
    
    if type(bodypart) is list and len(bodypart) > 1:
        for bodypart in bodypart:
            pd_dataframe = pd_dataframe[pd_dataframe[bodypart + ' likelihood'] >= threshold]
    
    elif type(bodypart) is list and len(bodypart) == 1:
        pd_dataframe = pd_dataframe[pd_dataframe[bodypart[0] + ' likelihood'] >= threshold]
    
    elif type(bodypart) is str:
        pd_dataframe = pd_dataframe[pd_dataframe[bodypart + ' likelihood'] >= threshold]
        
    # raise error if any bodypart name not identical as in csv
        
    return pd_dataframe


def find_slips(pd_dataframe, bodypart, axis, **kwargs): 
        
    t_peaks, properties = find_peaks(-pd_dataframe[f'{bodypart} {axis}'], height=-5000, prominence=(10,100000))
    # t_peaks, properties = find_peaks(pd_dataframe, prominence=0, distance=18, height=-10, width = 0)
    # width_half = peak_widths(data, t_peaks, rel_height=0.5)
    
#         t_peaks, properties = find_peaks(-data, prominence=(10,100000), height=-5000, width = 0)
#         width_half = peak_widths(-data, t_peaks, rel_height=0.5)
    
    index = pd_dataframe['bodyparts coords'].iloc[:]
    
    is_peak = np.zeros(len(index))
    n_peaks = 0
    current_data = pd_dataframe.iloc[0]
    norm = np.max(pd_dataframe[f'{bodypart} {axis}'])
    std = np.std(pd_dataframe[f'{bodypart} {axis}'])
    
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


def load_video(filename):

    vidcap = cv2.VideoCapture(filename)
    return vidcap


def plot_frame(video_file, n_frame, width, height, frame_rate, baseline = 0):

    try: 
        figure = mpl.figure.Figure(figsize=(width, height))
        axes = figure.add_subplot(111)
        axes.margins(x = 0)

        vidcap = load_video(video_file)
        vidcap.set(1, n_frame)
        ret, frame = vidcap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        axes.imshow(image)
        axes.title.set_text(f'frame {n_frame} ({n_frame/frame_rate:.1f} s)')

        return figure
        
    except cv2.error:

        print(f'Frame {n_frame} cannot be displayed! (cv2 error)')


def plot_labels(pd_dataframe, width, height, bodypart, axis, threshold = 0):
    
    figure = mpl.figure.Figure(figsize=(width, height))
    axes = figure.add_subplot(111)
    axes.margins(x = 0)
    # figure.tight_layout()

    axes.scatter(pd_dataframe['bodyparts coords'], pd_dataframe[f'{bodypart} {axis}'], s = 1)
            
    axes.legend([bodypart], loc='center right')
    # axes.set_xlabel('n frame')
    # axes.set_xlim(0, len(pd_dataframe['bodyparts coords'].iloc[:]))
    # axes.set_ylabel('distance from 0 (pixel)')
    # axes.title.set_text(f'{bodypart} {axis} coordinates by frame')

    return figure