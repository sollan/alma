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


def find_slips(pd_dataframe, bodypart, axis, panel = None, method = 'Baseline', window = None, threshold = None, **kwargs): 
        
    if method == 'Deviation':
        '''
        recommended for smooth / noiseless prediction
        '''
        t_peaks, properties = find_peaks(pd_dataframe[f'{bodypart} {axis}'], height=-10000, prominence=(45,100000))
        # h_peaks = properties["prominences"]
        

    elif method == 'Baseline':
        '''
        recommended for prediction with much jittering / noise
        '''
        baseline = baseline_als(pd_dataframe[f'{bodypart} {axis}'], 10**2, 0.1)
        t_peaks, properties = find_peaks(baseline, prominence=(10,100000))
        if window is None:
            if panel is not None:
                window = panel.frame_rate // 5
            else:
                window = 10
        t_peaks = adjust_times(pd_dataframe[f'{bodypart} {axis}'], t_peaks, window)
        # h_peaks = []
        # base = np.mean(pd_dataframe[f'{bodypart} {axis}'])
        # for t in t_peaks:
            # h_peaks.append(pd_dataframe[f'{bodypart} {axis}'].iloc[t] - base)
            # h_peaks.append(pd_dataframe[f'{bodypart} {axis}'].iloc[t])

    elif method == 'Threshold':
        '''
        when there is a clear cut off pixel value for the entire video, 
        e.g. location of ladder in frame
        '''
        if threshold is None:  
            threshold = np.mean(pd_dataframe[f'{bodypart} {axis}']) + np.std(pd_dataframe[f'{bodypart} {axis}'])
        adjusted = fit_threshold(pd_dataframe[f'{bodypart} {axis}'], threshold)
        t_peaks, properties = find_peaks(adjusted, prominence=(10,1000))
        h_peaks = []
        for t in t_peaks:
            # h_peaks.append(pd_dataframe[f'{bodypart} {axis}'].iloc[t] - threshold)
            h_peaks.append(pd_dataframe[f'{bodypart} {axis}'].iloc[t])

    n_peaks = len(t_peaks)
    # start_times = properties['left_bases']
    # end_times = properties['right_bases']
    start_times = t_peaks
    end_times = t_peaks
    h_peaks = pd_dataframe[f'{bodypart} {axis}'].iloc[t_peaks]
            
    return n_peaks, list(h_peaks), list(t_peaks), list(start_times), list(end_times)


def sort_list(list1, list2):
    '''
    sort list2 according to list1
    '''
    result = [x for _, x in sorted(zip(list1, list2) )] 
    return result


def find_duplicates(list1):
    to_remove = []
    for i, ele in enumerate(list1):
        if i == 0:
            prev = list1[0]
        elif ele == prev:
            to_remove.append(i)
        prev = ele
    return to_remove


def baseline_als(y, lam, p, niter=10):

    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    '''
    * based on ALS paper
    "There are two parameters: 
    p for asymmetry and λ for smoothness. 
    Both have to be tuned to the data at hand. 
    We found that generally 0.001 ≤ p ≤ 0.1 is a good choice 
    (for a signal with positive peaks) 
    and 10^2 ≤ λ ≤ 10^9 , 
    but exceptions may occur. 
    In any case one should vary λ on a grid that is approximately linear for log λ"
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


def fit_threshold(y, threshold):
    
    result = np.full(len(y), threshold)
    for i, data in enumerate(y):
        if data > threshold:
            result[i] = y[i]
    return result

def adjust_times(y, t_prediction, window):
    
    for i, t in enumerate(t_prediction):
        start = t - window
        end = t + window
        if start < 0:
            start = 0
        if end > len(y):
            end = len(y)
        sublist = list(y[start:end])
        lowest_timepoint = start + sublist.index(np.max(sublist))
        # DLC and opencv indexing: 
        # lowest point has largest axis value
        t_prediction[i] = lowest_timepoint
    
    return t_prediction


def make_output(pathname, t_slips, depth_slips, start_slips, end_slips, bodyparts):

    df_output = pd.DataFrame({'time': t_slips,
                            'depth': depth_slips,
                            'start': start_slips,
                            'end': end_slips,
                            'bodypart': bodyparts})

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
        _, frame = vidcap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        axes.imshow(image)
        axes.title.set_text(f'frame {n_frame} ({n_frame/frame_rate:.1f} s)')

        return figure
        
    except cv2.error:

        print(f'Frame {n_frame} cannot be displayed! (cv2 error)')


def plot_labels(pd_dataframe, n_current_frame, method, t_pred, start_pred, end_pred, width, height, bodypart, axis, likelihood_threshold = 0):
    
    figure = mpl.figure.Figure(figsize=(width, height))
    axes = figure.add_subplot(111)
    axes.margins(x = 0)
    # figure.tight_layout()
    # low_likelihood = np.array(pd_dataframe[pd_dataframe[bodyparts + ' likelihood'] < likelihood_threshold]['bodyparts coords'])
    axes.xaxis.set_label_position('top') 
    axes.scatter(pd_dataframe['bodyparts coords'], pd_dataframe[f'{bodypart} {axis}'], s = 1)
    axes.scatter(pd_dataframe['bodyparts coords'].iloc[n_current_frame], pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame], marker = 'x', c = 'r')
    axes.scatter(pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < likelihood_threshold]['bodyparts coords'], \
        pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < likelihood_threshold][f'{bodypart} {axis}'], s = 1, c = 'lightgrey')
    axes.invert_yaxis()

    # if method != 'Baseline':
    #     # the find minimum step in "baseline" interferes with on- and offset judgment
    #     if n_current_frame in t_pred:
    #         index = t_pred.index(n_current_frame)
    #         for i in range(start_pred[index], end_pred[index]):
    #             axes.axvspan(i, i+1, facecolor='0.2', alpha=0.5)

    return figure


def find_neighbors(n_current_frame, t_pred):     

    if len(t_pred) == 0:
        return 0, 0

    elif n_current_frame < t_pred[0]:
        return 0, t_pred[0]

    elif n_current_frame > t_pred[-1]:
        return t_pred[-1], 0

    elif n_current_frame in t_pred:
        if n_current_frame == t_pred[0]:
            return 0, t_pred[1]
        elif n_current_frame == t_pred[-1]:
            return t_pred[-2], 0
        else:
            current_ind = t_pred.index(n_current_frame)
            return t_pred[current_ind - 1], t_pred[current_ind + 1]

    else:
        return find_closest_neighbors(n_current_frame, t_pred)


def find_closest_neighbors(n_current_frame, t_pred):
    # t_pred is usually very short and validated lists 
    # can often be unsorted

    prev = t_pred[0]
    next = t_pred[-1]
    for i, t in enumerate(t_pred):
        if t > n_current_frame:
            next = t
            prev = t_pred[i-1]
            return prev, next
            break


def ControlButton(panel):
    
    panel.prev_pred_button.Enable()
    panel.next_pred_button.Enable()
    panel.prev10_button.Enable()
    panel.next10_button.Enable()
    panel.prev_button.Enable()
    panel.next_button.Enable()
    panel.to_start_button.Enable()
    panel.to_end_button.Enable()

    # if panel.n_frame <= panel.t_pred[0]:
    if panel.n_frame <= panel.t_val[0]:
        panel.prev_pred_button.Disable()
    # elif panel.n_frame >= panel.t_pred[-1]:
    elif panel.n_frame <= panel.t_val[0]:
        panel.next_pred_button.Disable()
    
    if panel.n_frame < 10:
        panel.prev10_button.Disable()
        if panel.n_frame == 0:
            panel.prev_button.Disable()
    elif panel.n_frame > len(panel.df) - 10:
        panel.next10_button.Disable()
        if panel.n_frame == len(panel.df):
            panel.next_button.Disable()

    if panel.n_frame not in panel.t_val:
        panel.to_start_button.Disable()
        panel.to_end_button.Disable()


def ControlPrediction(panel):

    if panel.n_frame in panel.t_val:
        panel.val_check_box.SetValue(True)
    else:
        panel.val_check_box.SetValue(False)

    if panel.n_frame in panel.start_val:
        panel.start_check_box.SetValue(True)
    else:
        panel.start_check_box.SetValue(False)

    if panel.n_frame in panel.end_val:
        panel.end_check_box.SetValue(True)
    else:
        panel.end_check_box.SetValue(False)


def DisplayPlots(panel):

    if panel.n_frame in panel.t_pred:
        panel.bodypart = panel.bodypart_list_pred[panel.t_pred.index(panel.n_frame)]
    else:
        pass

    try:
        frame = plot_frame(panel.video, panel.n_frame, 
        (panel.window_width-50) / 200, (panel.window_height // 3) // 100, int(panel.frame_rate))
        frame_canvas  = FigureCanvas(panel, -1, frame)
        panel.frame_canvas.Hide()
        panel.second_sizer.Replace(panel.frame_canvas, frame_canvas)
        panel.frame_canvas = frame_canvas
        panel.second_sizer_widgets.append(panel.frame_canvas)  
        panel.frame_canvas.Show()

        graph = plot_labels(panel.df, panel.n_frame, panel.method_selection, panel.t_pred, panel.start_pred, 
            panel.end_pred, (panel.window_width-50) / 100, (panel.window_height // 3) // 100, panel.bodypart, 'y', panel.likelihood_threshold)
        graph_canvas = FigureCanvas(panel, -1, graph)
        panel.graph_canvas.Hide()
        panel.second_sizer.Replace(panel.graph_canvas, graph_canvas)
        panel.graph_canvas = graph_canvas
        panel.second_sizer_widgets.append(panel.graph_canvas)
        panel.graph_canvas.Show()     
        panel.Fit()

        panel.SetSizer(panel.second_sizer)
        ControlPrediction(panel)
        panel.GetParent().Layout()
        
    except AttributeError:
        pass