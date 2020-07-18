import pandas as pd
from scipy.signal import peak_widths, find_peaks
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import cv2
import wx

def test(use_preset = True):
    
    filename = '/home/annette/Desktop/Irregular_347_3dpi_croppedDLC_resnet50_Ladder RungMay12shuffle1_50000.csv'
    df, filename = read_file(filename)
    df = fix_column_names(df)
    filtered_df = filter_predictions(df, 'HL', 0.3)
    video = '/home/annette/Desktop/Irregular_347_3dpi_cropped.avi'

    return filename, df, filtered_df, video
    

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


def find_slips(pd_dataframe, bodypart, axis, method, **kwargs): 
        

    if method == 'deviation':
    
        t_peaks, properties = find_peaks(-pd_dataframe[f'{bodypart} {axis}'], height=-5000, prominence=(10,100000))
        n_peaks = len(t_peaks)        
        h_peaks = properties["prominences"]
        start_times = properties['left_bases']
        end_times = properties['right_bases']

    if method == 'baseline':
        
        baseline = baseline_als(pd_dataframe[f'{bodypart} {axis}'], 1, 0.01)
        corrected = pd_dataframe[f'{bodypart} {axis}'] - baseline

        t_peaks, properties = find_peaks(-corrected, height=-5000, prominence=(10,100000))
        n_peaks = len(t_peaks)
        h_peaks = properties["prominences"]
        start_times = properties['left_bases']
        end_times = properties['right_bases']
            
    return n_peaks, list(h_peaks), list(t_peaks), list(start_times), list(end_times)



def baseline_als(y, lam, p, niter=10):

    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    '''
    * based on ALS paper
    There are two parameters: 
    p for asymmetry and λ for smoothness. 
    Both have to be tuned to the data at hand. 
    We found that generally 0.001 ≤ p ≤ 0.1 is a good choice 
    (for a signal with positive peaks) 
    and 10^2 ≤ λ ≤ 10^9 , 
    but exceptions may occur. 
    In any case one should vary λ on a grid that is approximately linear for log λ
    '''
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


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


def plot_labels(pd_dataframe, n_current_frame, t_pred, start_pred, end_pred, width, height, bodypart, axis, threshold = 0):
    
    figure = mpl.figure.Figure(figsize=(width, height))
    axes = figure.add_subplot(111)
    axes.margins(x = 0)
    # figure.tight_layout()
    # low_likelihood = np.array(pd_dataframe[pd_dataframe[bodyparts + ' likelihood'] < threshold]['bodyparts coords'])
    
    axes.scatter(pd_dataframe['bodyparts coords'], pd_dataframe[f'{bodypart} {axis}'], s = 1)
    axes.scatter(pd_dataframe['bodyparts coords'].iloc[n_current_frame], pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame], marker = 'x', c = 'r')
    axes.scatter(pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < threshold]['bodyparts coords'], \
        pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < threshold][f'{bodypart} {axis}'], s = 1, c = 'lightgrey')
    # axes.legend([bodypart], loc='center right')
    # axes.set_xlabel('n frame')
    # axes.set_xlim(0, len(pd_dataframe['bodyparts coords'].iloc[:]))
    # axes.set_ylabel('distance from 0 (pixel)')
    # axes.title.set_text(f'{bodypart} {axis} coordinates by frame')

    return figure


def find_neighbors(n_current_frame, t_pred):     

    if n_current_frame in t_pred:
        current_ind = t_pred.index(n_current_frame)
        if n_current_frame == t_pred[-1]:
            return t_pred[-2], 0
        elif n_current_frame == t_pred[0]:
            return 0, t_pred[1]
        else:
            return t_pred[current_ind - 1], t_pred[current_ind + 1]
    else:
        return find_closest_neighbors(n_current_frame, t_pred, t_pred[0], t_pred[-1])


def find_closest_neighbors(n_current_frame, t_pred, low, high):

    while t_pred.index(high) - t_pred.index(low) > 1:
        mid = t_pred[(t_pred.index(high) + t_pred.index(low)) // 2]
        if n_current_frame > mid:
            return find_closest_neighbors(n_current_frame, t_pred, mid, high)
        elif n_current_frame < mid:
            return find_closest_neighbors(n_current_frame, t_pred, low, mid)

    return low, high


def ControlButton(panel):
    
    panel.prev_pred_button.Enable()
    panel.next_pred_button.Enable()
    panel.prev10_button.Enable()
    panel.next10_button.Enable()
    panel.prev_button.Enable()
    panel.next_button.Enable()

    if panel.n_frame <= panel.t_pred[0]:
        panel.prev_pred_button.Disable()
    elif panel.n_frame >= panel.t_pred[-1]:
        panel.next_pred_button.Disable()
    
    if panel.n_frame < 10:
        panel.prev10_button.Disable()
        if panel.n_frame == 0:
            panel.prev_button.Disable()
    elif panel.n_frame > len(panel.df) - 10:
        panel.next10_button.Disable()
        if panel.n_frame == len(panel.df):
            panel.next_button.Disable()


def ControlPrediction(panel):

    if panel.n_frame in panel.t_val:
        panel.checkbox.SetValue(True)
    else:
        panel.checkbox.SetValue(False)



def DisplayPlots(panel):

    try:
        frame = plot_frame(panel.video, panel.n_frame, 
        (panel.window_width-50) / 200, (panel.window_height // 3) // 100, int(panel.frame_rate))
        frame_canvas  = FigureCanvas(panel, -1, frame)
        panel.frame_canvas.Hide()
        panel.sizer.Replace(panel.frame_canvas, frame_canvas)
        panel.frame_canvas = frame_canvas
        panel.frame_canvas.Show()

        graph = plot_labels(panel.df, panel.n_frame, panel.t_pred, panel.start_pred, \
            panel.end_pred, (panel.window_width-50) / 100, (panel.window_height // 3) // 100, panel.bodypart, panel.axis, panel.threshold)
        graph_canvas = FigureCanvas(panel, -1, graph)
        panel.graph_canvas.Hide()
        panel.sizer.Replace(panel.graph_canvas, graph_canvas)
        panel.graph_canvas = graph_canvas
        panel.graph_canvas.Show()     
        panel.Fit()

        panel.SetSizer(panel.sizer)
        ControlPrediction(panel)
        panel.GetParent().Layout()
        
    except AttributeError:
        pass