import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib as mpl
import cv2

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


def read_file(file):
    
    pd_dataframe = pd.read_csv(file, header=[1,2])
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


def filter_predictions(t_peaks, properties, pd_dataframe, bodypart, likelihood_threshold = 0.1, depth_threshold = 0.5):
    '''
    discard found peaks if the DLC prediction at a certain timepoint is below the set likelihood threshold
    ''' 
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

    for i, t in enumerate(properties['left_bases']):
        if pd_dataframe.iloc[t][f'{bodypart} likelihood']<likelihood_threshold:
            properties['left_bases'][i] = t_peaks[i] - 1
            properties['right_bases'][i] = t_peaks[i] + 1

    return t_peaks, properties


def find_footfalls(pd_dataframe, bodypart, axis, panel = None, method = 'Baseline', likelihood_threshold = 0.1, depth_threshold = 0.8, window = '', threshold = '', **kwargs): 
        
    if method == 'Deviation':
        '''
        recommended for smooth / noiseless prediction
        '''
        t_peaks, properties = find_peaks(pd_dataframe[f'{bodypart} {axis}'], height=-10000, prominence=(45,100000))
        t_peaks, properties = filter_predictions(t_peaks, properties, pd_dataframe, bodypart, likelihood_threshold, depth_threshold)
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
    # Check for NaN values in indices
    if pd.isna(starts) or pd.isna(t_peaks):
        return np.nan
    
    if ends is [] or pd.isna(ends):
        try:
            depth = np.array(pd_dataframe[f'{bodypart} y'][int(t_peaks)]) - \
                  np.array(pd_dataframe[f'{bodypart} y'][int(starts)])
        except (KeyError, ValueError, TypeError):
            return np.nan
    else:
        try:
            depth = ((np.array(pd_dataframe[f'{bodypart} y'][int(t_peaks)]) - \
                    np.array(pd_dataframe[f'{bodypart} y'][int(starts)])) + \
                    (np.array(pd_dataframe[f'{bodypart} y'][int(t_peaks)]) - \
                    np.array(pd_dataframe[f'{bodypart} y'][int(ends)]))) / 2
        except (KeyError, ValueError, TypeError):
            return np.nan
    
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


def make_output(pathname, pd_dataframe, t_footfalls, depth_footfalls, start_footfalls, end_footfalls, bodyparts, slip_falls, frame_rate, confirmed = [], confirmed_only = False):
    duration = []

    if confirmed_only:
        ts = []
        depths = []
        starts = []
        ends = []
        bds = []
        slip_or_falls = []
        for i, bodypart in enumerate(bodyparts):
            if confirmed[i] == 1:
                ts.append(t_footfalls[i])
                bds.append(bodyparts[i])
                slip_or_falls.append(slip_falls[i])
                try:
                    depths.append(calculate_depths(pd_dataframe, bodypart, start_footfalls[i], end_footfalls[i], t_footfalls[i]))
                except (TypeError, KeyError, ValueError):
                    # missing start/end or invalid index
                    depths.append(np.nan)
                try:
                    if pd.isna(start_footfalls[i]) or pd.isna(end_footfalls[i]):
                        duration.append(np.nan)
                    else:
                        duration.append(round((end_footfalls[i] - start_footfalls[i]) / frame_rate, 3))
                except (TypeError, ValueError):
                    duration.append(np.nan)
                try:
                    starts.append(int(start_footfalls[i]) if not pd.isna(start_footfalls[i]) else np.nan)
                except (TypeError, ValueError):
                    starts.append(np.nan)
                try:
                    ends.append(int(end_footfalls[i]) if not pd.isna(end_footfalls[i]) else np.nan)
                except (TypeError, ValueError):
                    ends.append(np.nan)

        df_output = pd.DataFrame({'time (frame)': ts,
                                'depth (pixel)': depths,
                                'start (frame)': starts,
                                'end (frame)': ends,
                                'duration (s)': duration,
                                'bodypart': bds,
                                'slip or fall': slip_or_falls})

    else:
        for i, bodypart in enumerate(bodyparts):
            depth_footfalls[i] = calculate_depths(pd_dataframe, bodypart, start_footfalls[i], end_footfalls[i], t_footfalls[i])
            duration.append(round((end_footfalls[i] - start_footfalls[i]) / frame_rate, 3))
        
        df_output = pd.DataFrame({'time (frame)': t_footfalls,
                                'depth (pixel)': depth_footfalls,
                                'start (frame)': start_footfalls,
                                'end (frame)': end_footfalls,
                                'duration (s)': duration,
                                'bodypart': bodyparts})

    df_output.to_csv(pathname, index = False)


def load_video(filename):

    vidcap = cv2.VideoCapture(filename)
    return vidcap


def plot_frame(video_file, n_frame, width, height, frame_rate, pd_dataframe, bodypart, zoom):

    try: 
        figure = mpl.figure.Figure(figsize=(width, height), tight_layout=True, facecolor='none')
        axes = figure.add_subplot(111)
        axes.margins(x = 0)

        vidcap = load_video(video_file)
        vidcap.set(1, n_frame)
        _, frame = vidcap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if zoom: 
            y_max = np.array(image).shape[0]
            x_max = np.array(image).shape[1]

            # print(x_max, y_max)
            x_loc = pd_dataframe[f'{bodypart} x'].iloc[n_frame]
            y_loc = pd_dataframe[f'{bodypart} y'].iloc[n_frame]

            if x_loc <= 100:
                axes.set_xlim(0, 200)
            elif x_loc >= x_max-100:
                axes.set_xlim(x_max-200, x_max)
            else:
                axes.set_xlim(x_loc-100, x_loc+100)

            if y_loc <= 50:
                axes.set_ylim(100, 0)
            elif y_loc >= y_max-50:
                axes.set_ylim(y_max, y_max-100)
            else:
                axes.set_ylim(y_loc+50, y_loc-50)

        axes.imshow(image)
        axes.title.set_text(f'frame {n_frame} ({n_frame/frame_rate:.1f} s)')

        return figure
        
    except cv2.error:
        print(f'Frame {n_frame} cannot be displayed! (cv2 error)')
        return None


def plot_labels(pd_dataframe, n_current_frame, method, t_pred, start_pred, end_pred, width, height, bodypart, bodypart_list, selected_bodyparts, axis, likelihood_threshold, confirmed, zoom=True):
    if zoom:
        figure = mpl.figure.Figure(figsize=(width, height), facecolor='none')
    else:
        figure = mpl.figure.Figure(figsize=(width, height), tight_layout=True, facecolor='none')
    axes = figure.add_subplot(111)
    axes.margins(x = 0)
    axes.xaxis.set_label_position('top')
    axes.set_xlabel('Frame', fontsize=10)
    axes.set_ylabel('Y Position (pixels)', fontsize=10)

    # Plot all bodyparts in light blue
    for bp in selected_bodyparts:
        axes.scatter(pd_dataframe['bodyparts coords'], pd_dataframe[f'{bp} {axis}'], s = 0.5, color = 'steelblue', alpha=0.3)

    # Plot current bodypart
    axes.plot(pd_dataframe['bodyparts coords'], pd_dataframe[f'{bodypart} {axis}'], linewidth=0.8, color='darkblue', alpha=0.6)
    
    # Highlight current frame with larger marker
    axes.scatter(pd_dataframe['bodyparts coords'].iloc[n_current_frame], 
                pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame], 
                marker = 'o', s=100, c = 'red', zorder=10, edgecolors='darkred', linewidths=2)

    # Find current footfall if on one
    current_footfall_index = None
    if n_current_frame in t_pred:
        current_footfall_index = t_pred.index(n_current_frame)
    
    # Plot all confirmed footfalls with numbers
    count_footfalls = 0
    for i, t in enumerate(t_pred):
        bp = bodypart_list[i]
        if bp == np.nan:
            bp = bodypart
        if confirmed[i]: 
            count_footfalls += 1
            # Mark the footfall peak
            axes.scatter(t, pd_dataframe[f'{bp} {axis}'][t], s=60, color='green', marker='v', zorder=5, edgecolors='darkgreen', linewidths=1)
            axes.annotate(str(count_footfalls), (t, pd_dataframe[f'{bp} {axis}'][t]), 
                         color='green', fontweight='bold', fontsize=11, 
                         xytext=(0, -15), textcoords='offset points', ha='center')
            
            # Draw start and end markers if they exist
            try:
                if not np.isnan(start_pred[i]) and bodypart_list[i] == bodypart:
                    start_frame = int(start_pred[i])
                    axes.scatter(start_frame, pd_dataframe[f'{bodypart} {axis}'].iloc[start_frame], 
                               s=40, color='orange', marker='<', zorder=6, edgecolors='darkorange', linewidths=1)
                    # Draw span line from start to peak
                    axes.plot([start_frame, t], 
                             [pd_dataframe[f'{bodypart} {axis}'].iloc[start_frame], pd_dataframe[f'{bp} {axis}'][t]], 
                             color='orange', linewidth=2, alpha=0.5, linestyle='--')
                    
                if not np.isnan(end_pred[i]) and bodypart_list[i] == bodypart:
                    end_frame = int(end_pred[i])
                    axes.scatter(end_frame, pd_dataframe[f'{bodypart} {axis}'].iloc[end_frame], 
                               s=40, color='purple', marker='>', zorder=6, edgecolors='indigo', linewidths=1)
                    # Draw span line from peak to end
                    axes.plot([t, end_frame], 
                             [pd_dataframe[f'{bp} {axis}'][t], pd_dataframe[f'{bodypart} {axis}'].iloc[end_frame]], 
                             color='purple', linewidth=2, alpha=0.5, linestyle='--')
                    
            except (TypeError, ValueError, IndexError):
                pass
        else:
            # Unconfirmed footfalls in gray
            axes.scatter(t, pd_dataframe[f'{bp} {axis}'][t], s=40, color='gray', marker='v', alpha=0.5, zorder=4)

    # Highlight current footfall's start and end if active
    if current_footfall_index is not None:
        try:
            if bodypart_list[current_footfall_index] == bodypart:
                start_idx = start_pred[current_footfall_index]
                end_idx = end_pred[current_footfall_index]
                
                if not np.isnan(start_idx):
                    start_frame = int(start_idx)
                    axes.axvline(x=start_frame, color='orange', linestyle=':', linewidth=2, alpha=0.7)
                    axes.text(start_frame, axes.get_ylim()[0], 'START', 
                             rotation=90, va='bottom', ha='right', color='orange', fontweight='bold', fontsize=9)
                    
                if not np.isnan(end_idx):
                    end_frame = int(end_idx)
                    axes.axvline(x=end_frame, color='purple', linestyle=':', linewidth=2, alpha=0.7)
                    axes.text(end_frame, axes.get_ylim()[0], 'END', 
                             rotation=90, va='bottom', ha='left', color='purple', fontweight='bold', fontsize=9)
        except (TypeError, ValueError, IndexError):
            pass

    # Plot low likelihood points in light gray
    low_likelihood_mask = pd_dataframe[f'{bodypart} likelihood'] < likelihood_threshold
    axes.scatter(pd_dataframe[low_likelihood_mask]['bodyparts coords'], 
                pd_dataframe[low_likelihood_mask][f'{bodypart} {axis}'], 
                s = 2, c = '0.8', alpha=0.5, zorder=1)
    
    axes.invert_yaxis()

    if zoom:
        x_max = len(pd_dataframe)
        if n_current_frame <= 300:
            axes.set_xlim(0, 600)
        elif n_current_frame >= x_max-300:
            axes.set_xlim(x_max-600, x_max)
        else:
            axes.set_xlim(n_current_frame-300, n_current_frame+300)

        y_current = pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame]
        axes.set_ylim(y_current+100, y_current-100)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Current Frame'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='green', markersize=8, label='Confirmed Footfall'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=8, label='Unconfirmed'),
        Line2D([0], [0], marker='<', color='w', markerfacecolor='orange', markersize=8, label='Start'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='purple', markersize=8, label='End'),
    ]
    axes.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    return figure


def find_neighbors(n_current_frame, t_pred, start = None, end = None):

    if len(t_pred) == 0:
        return 0, 0

    elif len(t_pred) == 1:
        if n_current_frame > t_pred[0]:
            return t_pred[0], 0
        elif n_current_frame < t_pred[0]:
            return 0, t_pred[0]
        elif n_current_frame == t_pred[0]:
            return 0, 0

    elif n_current_frame < t_pred[0] and len(t_pred) > 1:
        return 0, t_pred[0]

    elif n_current_frame > t_pred[-1] and len(t_pred) > 1:
        return t_pred[-1], 0

    elif n_current_frame in t_pred and len(t_pred) > 1:
        if n_current_frame == t_pred[0]:
            return 0, t_pred[1]
        elif n_current_frame == t_pred[-1]:
            return t_pred[-2], 0
        else:
            current_ind = t_pred.index(n_current_frame)
            return t_pred[current_ind - 1], t_pred[current_ind + 1]

    else:
        return find_closest_neighbors(n_current_frame, t_pred, start=start, end=end)


def find_closest_neighbors(n_current_frame, t_pred, start = None, end = None):
    if end is not None:
        for i in range(end-1, -1, -1):
            if t_pred[i] < n_current_frame:
                return t_pred[i], t_pred[i+1]
    elif start is not None:
        for i in range(start, len(t_pred)):
            if t_pred[i] > n_current_frame:
                return t_pred[i-1], t_pred[i]
    else:
        for i, t in enumerate(t_pred):
            if t > n_current_frame:
                return t_pred[i-1], t

def find_confirmed_neighbors(n_current_frame, t_val, confirmed = None, start = None, end = None):

    t_val_confirmed = np.array(t_val)[np.array(confirmed) == 1]
    if start is not None:
        try:
            start = list(t_val_confirmed).index(t_val[start])
            return find_neighbors(n_current_frame, t_val_confirmed, start=start)
        except ValueError:
            # prev closest prediction is not confirmed
            start = None
            return find_neighbors(n_current_frame, t_val_confirmed)
    if end is not None:
        try:
            end = list(t_val_confirmed).index(t_val[end])
            return find_neighbors(n_current_frame, t_val_confirmed, end=end)
        except ValueError:
            # next closest prediction is not confirmed
            end = None
            return find_neighbors(n_current_frame, t_val_confirmed)


def ControlButton(panel):
    """Control button states for PySide6"""
    
    def enable_button(btn):
        btn.setEnabled(True)
    
    def disable_button(btn):
        btn.setEnabled(False)
    
    # Enable all buttons by default
    enable_button(panel.prev_pred_button)
    enable_button(panel.next_pred_button)
    enable_button(panel.prev10_button)
    enable_button(panel.next10_button)
    enable_button(panel.prev_button)
    enable_button(panel.next_button)
    enable_button(panel.to_start_button)
    enable_button(panel.to_end_button)
    
    # Disable based on state
    if panel.n_val == 0:
        disable_button(panel.prev_pred_button)
        disable_button(panel.next_pred_button)
    elif panel.n_frame <= panel.t_val[0]:
        disable_button(panel.prev_pred_button)
    elif panel.n_frame >= panel.t_val[-1]:
        disable_button(panel.next_pred_button)
    
    if panel.n_frame < 10:
        disable_button(panel.prev10_button)
        if panel.n_frame == 0:
            disable_button(panel.prev_button)
    elif panel.n_frame > len(panel.df) - 10:
        disable_button(panel.next10_button)
        if panel.n_frame == len(panel.df):
            disable_button(panel.next_button)

    if panel.n_frame not in panel.t_val:
        disable_button(panel.to_start_button)
        disable_button(panel.to_end_button)
    else:
        index = panel.t_val.index(panel.n_frame)
        if panel.start_val[index] is np.nan:
            disable_button(panel.to_start_button)
        if panel.end_val[index] is np.nan:
            disable_button(panel.to_end_button)


def ControlPrediction(panel):
    """Update checkbox states for PySide6"""
    
    def set_checkbox(checkbox, value):
        checkbox.setChecked(value)
    
    if panel.n_frame in panel.t_val:
        index = panel.t_val.index(panel.n_frame)
        set_checkbox(panel.val_check_box, panel.confirmed[index])
    else:
        set_checkbox(panel.val_check_box, False)

    if panel.n_frame in panel.start_val:
        set_checkbox(panel.start_check_box, True)
    else:
        set_checkbox(panel.start_check_box, False)

    if panel.n_frame in panel.end_val:
        set_checkbox(panel.end_check_box, True)
    else:
        set_checkbox(panel.end_check_box, False)


def DisplayPlots(panel, set_bodypart = True):
    """Update frame and graph displays - works with both wxPython and PySide6"""
    
    if panel.n_frame in panel.t_val and set_bodypart:
        panel.bodypart = panel.bodypart_list_val[panel.t_val.index(panel.n_frame)]
        panel.bodypart_to_plot.setCurrentText(panel.bodypart)
        
    try:
        # Create new frame plot
        frame = plot_frame(panel.video, panel.n_frame, 
                          6, 3, int(panel.frame_rate), panel.df, panel.bodypart, panel.zoom_image)
        
        # Check if using interactive timeline (PyQtGraph) or old matplotlib
        if hasattr(panel, 'interactive_timeline'):
            # Update interactive timeline widget
            panel.interactive_timeline.set_current_bodypart(panel.bodypart)
            panel.interactive_timeline.set_detection_data(panel.t_val, panel.start_val, panel.end_val, 
                                                         panel.bodypart_list_val, panel.confirmed, panel.slip_fall_val)
            panel.interactive_timeline.set_current_frame(panel.n_frame, panel.likelihood_threshold)
            panel.interactive_timeline.update_plot()
            
            # Update frame canvas
            panel.frame_canvas.figure.clear()
            panel.frame_canvas.figure = frame
            for ax in frame.axes:
                panel.frame_canvas.figure._axstack.add(ax)
            panel.frame_canvas.draw()
            
            panel.update()
        else:
            # Fallback to old matplotlib approach
            graph = plot_labels(panel.df, panel.n_frame, panel.method_selection, panel.t_val, panel.start_val, 
                               panel.end_val, (panel.window_width-60) // 100, (panel.window_height // 3) // 100, 
                               panel.bodypart, panel.bodypart_list_val, panel.selected_bodyparts, 'y', 
                               panel.likelihood_threshold, panel.confirmed, panel.zoom)
            
            # PySide6 approach - replace canvas figures
            panel.frame_canvas.figure.clear()
            panel.frame_canvas.figure = frame
            for ax in frame.axes:
                panel.frame_canvas.figure._axstack.add(ax)
            panel.frame_canvas.draw()
            
            panel.graph_canvas.figure.clear()
            panel.graph_canvas.figure = graph
            for ax in graph.axes:
                panel.graph_canvas.figure._axstack.add(ax)
            panel.graph_canvas.draw()
            
            panel.update()
        
        ControlPrediction(panel)
        
    except AttributeError as e:
        print(f"DisplayPlots error: {e}")