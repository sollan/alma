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


def filter_predictions(t_peaks, properties, pd_dataframe, bodypart, likelihood_threshold = 0.1, depth_threshold = 0.8):
    '''
    discard found peaks if the DLC prediction at a certain timepoint is below the set likelihood threshold
    '''    
    for i, t in enumerate(properties['left_bases']):
        if pd_dataframe.iloc[t][f'{bodypart} likelihood']<likelihood_threshold:
            properties['left_bases'][i] = t_peaks[i] - 1
    for i, t in enumerate(properties['right_bases']):
        if pd_dataframe.iloc[t][f'{bodypart} likelihood']<likelihood_threshold:
            properties['right_bases'][i] = t_peaks[i] + 1
    for i, t in enumerate(properties['left_bases']):
        if pd_dataframe.iloc[t][f'{bodypart} likelihood']<likelihood_threshold:
            properties['left_bases'][i] = t_peaks[i] - 1
    for i, t in enumerate(properties['right_bases']):
        if pd_dataframe.iloc[t][f'{bodypart} likelihood']<likelihood_threshold:
            properties['right_bases'][i] = t_peaks[i] + 1

    result = np.where(pd_dataframe.iloc[list(t_peaks)][f'{bodypart} likelihood']>=likelihood_threshold)
    ind_valid_peaks = []
    result = np.array(result[0])
    ind_valid_peaks.append(result[0])


    for i in range(1, len(result)):
        prev_mid = t_peaks[result[i-1]]
        curr_mid = t_peaks[result[i]]
        # highest point between prev and current slip (recovered y distance)
        max_between_slip = min(pd_dataframe.iloc[prev_mid : curr_mid][pd_dataframe.iloc[prev_mid : curr_mid][f'{bodypart} likelihood'] >= likelihood_threshold][f'{bodypart} y'])
        max_between = np.where(pd_dataframe.iloc[prev_mid : curr_mid][f'{bodypart} y']==max_between_slip)[0][0]
        max_between = np.array(pd_dataframe.iloc[prev_mid : curr_mid]['bodyparts coords'])[max_between]
        prev_depth = pd_dataframe.iloc[t_peaks[result[i-1]]][f'{bodypart} y'] - \
                        pd_dataframe.iloc[properties['left_bases'][result[i-1]]][f'{bodypart} y']
        prev_mid_depth = pd_dataframe.iloc[t_peaks[result[i-1]]][f'{bodypart} y'] 
        prev_end = properties['right_bases'][result[i-1]]
        curr_mid_depth = pd_dataframe.iloc[t_peaks[result[i]]][f'{bodypart} y']
        curr_start = properties['left_bases'][result[i]]
        if curr_start > prev_end:
            # separate slips
            ind_valid_peaks.append(result[i])
        else:
            # overlapping
            if curr_mid_depth - max_between_slip >= depth_threshold*prev_depth:
                # recovered a percentage of prev slip depth
                properties['left_bases'][result[i]] = max_between
                ind_valid_peaks.append(result[i])
            else:
                # mark as same slip; correct prev end and depth
                if curr_mid_depth > prev_mid_depth:
                    # current prediction is deeper; adjust mid and start of slip timing
                    ind_valid_peaks.pop()
                    ind_valid_peaks.append(result[i])
                    properties['left_bases'][result[i]] = properties['left_bases'][result[i-1]]
                else:
                    # prev prediction is deeper; adjust end of slip timing
                    properties['right_bases'][result[i-1]] = properties['right_bases'][result[i]]
    
    t_peaks = t_peaks[ind_valid_peaks]
    for item in properties:
        # a dictionary containing prominence, start, end etc.
        properties[item] = properties[item][ind_valid_peaks]



    return t_peaks, properties


def find_slips(pd_dataframe, bodypart, axis, panel = None, method = 'Baseline', likelihood_threshold = 0.1, depth_threshold = 0.8, window = '', threshold = '', **kwargs): 
        
    if method == 'Deviation':
        '''
        recommended for smooth / noiseless prediction
        '''
        t_peaks, properties = find_peaks(pd_dataframe[f'{bodypart} {axis}'], height=-10000, prominence=(45,100000))
#         print('start filter')
        t_peaks, properties = filter_predictions(t_peaks, properties, pd_dataframe, bodypart, likelihood_threshold, depth_threshold)
        # h_peaks = properties["prominences"]
        start_times = properties['left_bases']
        end_times = properties['right_bases']
        h_peaks = pd_dataframe[f'{bodypart} {axis}'][t_peaks]

    elif method == 'Baseline':
        '''
        recommended for prediction with much jittering / noise
        '''
        index = np.array(pd_dataframe[pd_dataframe[f'{bodypart} likelihood']>=likelihood_threshold].index)
        baseline = baseline_als(pd_dataframe[f'{bodypart} {axis}']\
                                [pd_dataframe[f'{bodypart} likelihood']>=likelihood_threshold], 10**2, 0.1)
        t_peaks, properties = find_peaks(baseline, prominence=(10,100000))
        t_peaks = index[t_peaks]
        properties['left_bases'] = index[properties['left_bases']]
        properties['right_bases'] = index[properties['right_bases']]
        t_peaks, properties = filter_predictions(t_peaks, properties, pd_dataframe, bodypart, likelihood_threshold, depth_threshold)
        if window == '':
            if panel is not None:
                window = panel.frame_rate // 5
            else:
                window = 10
        t_peaks = adjust_times(pd_dataframe[f'{bodypart} {axis}'], t_peaks, window)
        h_peaks = pd_dataframe[f'{bodypart} {axis}'][t_peaks]
        start_times = properties['left_bases']
        end_times = properties['right_bases']

    elif method == 'Threshold':
        '''
        when there is a clear cut off pixel value for the entire video, 
        e.g. location of ladder in frame
        '''
        if threshold == '':  
            threshold = np.mean(pd_dataframe[f'{bodypart} {axis}']) + np.std(pd_dataframe[f'{bodypart} {axis}'])
        adjusted = fit_threshold(pd_dataframe[f'{bodypart} {axis}'], threshold)
        t_peaks, properties = find_peaks(adjusted, prominence=(10,1000))
        t_peaks, properties = filter_predictions(t_peaks, properties, pd_dataframe, bodypart, likelihood_threshold, depth_threshold)
        h_peaks = pd_dataframe[f'{bodypart} {axis}'][t_peaks]
        start_times = properties['left_bases']
        end_times = properties['right_bases']

    else:
        '''
        to make sure t_peaks exists
        '''
        t_peaks = []
        h_peaks = []
        start_times = []
        end_times = []
        h_peaks = []

    n_peaks = len(t_peaks)
    
    return n_peaks, list(h_peaks), list(t_peaks), list(start_times), list(end_times)


def calculate_depths(pd_dataframe, bodypart, starts, ends, t_peaks):
    if ends is []:
        depth = np.array(pd_dataframe[f'{bodypart} y'][t_peaks]) - \
              np.array(pd_dataframe[f'{bodypart} y'][starts])
    else:
        depth = ((np.array(pd_dataframe[f'{bodypart} y'][t_peaks]) - \
                np.array(pd_dataframe[f'{bodypart} y'][starts])) + \
                (np.array(pd_dataframe[f'{bodypart} y'][t_peaks]) - \
                np.array(pd_dataframe[f'{bodypart} y'][ends]))) / 2
    
    return depth

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
        sublist = list(y[int(start):int(end)])
        lowest_timepoint = start + sublist.index(np.max(sublist))
        # DLC and opencv indexing: 
        # lowest point has largest axis value
        t_prediction[i] = lowest_timepoint
    
    return t_prediction


def make_output(pathname, pd_dataframe, t_slips, depth_slips, start_slips, end_slips, bodyparts, frame_rate, confirmed = [], confirmed_only = False):
    duration = []

    if confirmed_only:
        ts = []
        depths = []
        starts = []
        ends = []
        bds = []
        for i, bodypart in enumerate(bodyparts):
            if confirmed[i] == 1:
                ts.append(t_slips[i])
                bds.append(bodyparts[i])
                try:
                    depths.append(calculate_depths(pd_dataframe, bodypart, start_slips[i], end_slips[i], t_slips[i]))
                except TypeError:
                    # missing start and end?
                    depths.append(np.nan)
                try:
                    duration.append(round((end_slips[i] - start_slips[i]) / frame_rate, 3))
                except TypeError:
                    duration.append(np.nan)
                try: 
                    starts.append(start_slips[i])
                except TypeError:
                    starts.append(np.nan)
                try:
                    ends.append(end_slips[i])
                except TypeError:
                    ends.append(np.nan)

        df_output = pd.DataFrame({'time (frame)': ts,
                                'depth (pixel)': depths,
                                'start (frame)': starts,
                                'end (frame)': ends,
                                'duration (s)': duration,
                                'bodypart': bds})

    else:
        for i, bodypart in enumerate(bodyparts):
            depth_slips[i] = calculate_depths(pd_dataframe, bodypart, start_slips[i], end_slips[i], t_slips[i])
            duration.append(round((end_slips[i] - start_slips[i]) / frame_rate, 3))
        
        df_output = pd.DataFrame({'time (frame)': t_slips,
                                'depth (pixel)': depth_slips,
                                'start (frame)': start_slips,
                                'end (frame)': end_slips,
                                'duration (s)': duration,
                                'bodypart': bodyparts})

    df_output.to_csv(pathname, index = False)


def load_video(filename):

    vidcap = cv2.VideoCapture(filename)
    return vidcap


def plot_frame(video_file, n_frame, width, height, frame_rate, pd_dataframe, bodypart):

    try: 
        x_loc = pd_dataframe[f'{bodypart} x'].iloc[n_frame]
        y_loc = pd_dataframe[f'{bodypart} y'].iloc[n_frame]

        # update this with image dimension
        if x_loc < 200:
            x_loc += 200
        if y_loc < 100:
            y_loc += 100
        if x_loc > 1800:
            x_loc -= 200
        if y_loc > 550:
            y_loc -= 100

        figure = mpl.figure.Figure(figsize=(width, height), tight_layout = True)
        axes = figure.add_subplot(111)
        axes.margins(x = 0)

        vidcap = load_video(video_file)
        vidcap.set(1, n_frame)
        _, frame = vidcap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        axes.imshow(image)
        axes.title.set_text(f'frame {n_frame} ({n_frame/frame_rate:.1f} s)')
        axes.set_xlim(x_loc - 200, x_loc + 200)
        axes.set_ylim(y_loc + 100, y_loc - 100)

        return figure
        
    except cv2.error:
        print(f'Frame {n_frame} cannot be displayed! (cv2 error)')
        # plot_frame(video_file, n_frame+2, width, height, frame_rate, baseline)


def plot_labels(pd_dataframe, n_current_frame, method, t_pred, start_pred, end_pred, width, height, bodypart, bodypart_list, selected_bodyparts, axis, likelihood_threshold, confirmed):
    
    figure = mpl.figure.Figure(figsize=(width, height), tight_layout = True)
    axes = figure.add_subplot(111)
    axes.margins(x = 0)
    # figure.tight_layout()
    # low_likelihood = np.array(pd_dataframe[pd_dataframe[bodyparts + ' likelihood'] < likelihood_threshold]['bodyparts coords'])
    axes.xaxis.set_label_position('top') 

    for bp in selected_bodyparts:
        axes.scatter(pd_dataframe['bodyparts coords'], pd_dataframe[f'{bp} {axis}'], s = 0.1, color = 'steelblue')

    axes.scatter(pd_dataframe['bodyparts coords'], pd_dataframe[f'{bodypart} {axis}'], s = 1)
    axes.scatter(pd_dataframe['bodyparts coords'].iloc[n_current_frame], pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame], marker = 'x', c = 'r')

    if n_current_frame in t_pred:
        index = t_pred.index(n_current_frame)
        try:
            if bodypart_list[index] == bodypart:
                axes.scatter(start_pred[index], pd_dataframe[f'{bodypart} {axis}'].iloc[start_pred[index]], s = 2, color = 'r')
                axes.scatter(end_pred[index], pd_dataframe[f'{bodypart} {axis}'].iloc[end_pred[index]], s = 2, color = 'r')
                axes.annotate('Start', (start_pred[index], pd_dataframe[f'{bodypart} {axis}'].iloc[start_pred[index]]))
                axes.annotate('End', (end_pred[index], pd_dataframe[f'{bodypart} {axis}'].iloc[end_pred[index]]))
        except TypeError:
            # user hasn't selected start and end time for added slips!
            pass

    axes.scatter(pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < likelihood_threshold]['bodyparts coords'], \
        pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < likelihood_threshold][f'{bodypart} {axis}'], s = 1, c = '0.7')
    axes.invert_yaxis()
    for i, t in enumerate(t_pred):
        bp = bodypart_list[i]
        if bp == np.nan:
            bp = bodypart
        if confirmed[i]: 
            c = 'g'
        else:
            c = 'r'
        axes.annotate(str(i+1), (t, pd_dataframe[f'{bp} {axis}'][t]), color = c, weight = 'bold')

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

def find_confirmed_neighbors(n_current_frame, t_val, confirmed):
    t_val_confirmed = []
    for i, t in enumerate(t_val):
        if confirmed[i] == 1:
            t_val_confirmed.append(t)
    return find_neighbors(n_current_frame, t_val_confirmed)


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
    else:
        index = panel.t_val.index(panel.n_frame)
        if panel.start_val[index] is np.nan:
            panel.to_start_button.Disable()
        if panel.end_val[index] is np.nan:
            
            panel.to_end_button.Disable()


def ControlPrediction(panel):

    if panel.n_frame in panel.t_val:
        index = panel.t_val.index(panel.n_frame)
        panel.val_check_box.SetValue(panel.confirmed[index])
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


def DisplayPlots(panel, set_bodypart = True):

    if panel.n_frame in panel.t_val and set_bodypart:
        panel.bodypart = panel.bodypart_list_val[panel.t_val.index(panel.n_frame)]
        panel.bodypart_to_plot.SetValue(panel.bodypart)
        
    try:
        frame = plot_frame(panel.video, panel.n_frame, 
        6,3, int(panel.frame_rate), panel.df, panel.bodypart)
        frame_canvas  = FigureCanvas(panel, -1, frame)
        panel.frame_canvas.Hide()
        panel.second_sizer.Replace(panel.frame_canvas, frame_canvas)
        panel.frame_canvas = frame_canvas
        panel.second_sizer_widgets.append(panel.frame_canvas)  
        panel.frame_canvas.Show()

        graph = plot_labels(panel.df, panel.n_frame, panel.method_selection, panel.t_val, panel.start_val, 
            panel.end_val, (panel.window_width-60) // 100, (panel.window_height // 3) // 100, panel.bodypart, panel.bodypart_list_val, panel.selected_bodyparts, 'y', panel.likelihood_threshold, panel.confirmed)
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