import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import cv2


def read_file(file):

    pd_dataframe = pd.read_csv(file, header=[1, 2])
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


def filter_predictions(t_peaks, properties, pd_dataframe, bodypart, likelihood_threshold=0.1, depth_threshold=0.8):
    '''
    discard found peaks if the DLC prediction at a
    certain timepoint is below the set likelihood threshold
    '''

    result = np.where(pd_dataframe.iloc[list(t_peaks)][f'{bodypart} likelihood']>=likelihood_threshold)
    ind_valid_peaks = []
    result = np.array(result[0])
    ind_valid_peaks.append(result[0])

    for i in range(1, len(result)):
        prev_mid = t_peaks[result[i-1]]
        curr_mid = t_peaks[result[i]]

        # highest point between prev and current footfall
        # (recovered y distance)
        max_between_footfall = min(pd_dataframe.iloc[prev_mid : curr_mid][pd_dataframe.iloc[prev_mid : curr_mid][f'{bodypart} likelihood'] >= likelihood_threshold][f'{bodypart} y'])
        max_between = np.where(pd_dataframe.iloc[prev_mid : curr_mid][f'{bodypart} y']==max_between_footfall)[0][0]
        max_between = np.array(pd_dataframe.iloc[prev_mid : curr_mid]['bodyparts coords'])[max_between]
        prev_depth = pd_dataframe.iloc[prev_mid][f'{bodypart} y'] - \
            pd_dataframe.iloc[properties['left_bases'][result[i-1]]][f'{bodypart} y']
        prev_mid_depth = pd_dataframe.iloc[prev_mid][f'{bodypart} y']
        prev_end = properties['right_bases'][result[i-1]]
        curr_mid_depth = pd_dataframe.iloc[t_peaks[result[i]]][f'{bodypart} y']
        curr_start = properties['left_bases'][result[i]]

        # x coordinate location compared to prev footfall
        prev_x = pd_dataframe.iloc[prev_mid][f'{bodypart} x']
        curr_x = pd_dataframe.iloc[curr_mid][f'{bodypart} x']
        x_diff = np.abs(prev_x - curr_x)
        if curr_start > prev_end and x_diff >= 30:
            # separate footfalls
            ind_valid_peaks.append(result[i])
        else:
            # overlapping predictions
            if curr_mid_depth - max_between_footfall >= depth_threshold*prev_depth:
                # recovered a percentage of prev footfall depth
                if x_diff >= 30:
                    # different rung
                    properties['left_bases'][result[i]] = max_between
                    ind_valid_peaks.append(result[i])
                    # not different rung, recovered -> do not count
            else:
                # did not recover, mark as same footfall;
                # correct prev end and depth
                if curr_mid_depth > prev_mid_depth:
                    # current prediction is deeper;
                    # adjust mid and start of footfall timing
                    ind_valid_peaks.pop()
                    ind_valid_peaks.append(result[i])
                    properties['left_bases'][result[i]] = properties['left_bases'][result[i-1]]
                else:
                    # prev prediction is deeper;
                    # adjust end of footfall timing
                    properties['right_bases'][result[i-1]] = properties['right_bases'][result[i]]

    t_peaks = t_peaks[ind_valid_peaks]
    for item in properties:
        # a dictionary containing prominence, start, end etc.
        properties[item] = properties[item][ind_valid_peaks]
    for i, t in enumerate(properties['left_bases']):
        if pd_dataframe.iloc[t][f'{bodypart} likelihood'] < likelihood_threshold:
            properties['left_bases'][i] = t_peaks[i] - 1
            properties['right_bases'][i] = t_peaks[i] + 1

    return t_peaks, properties


def find_footfalls(pd_dataframe, bodypart, axis, panel=None, method='Baseline', likelihood_threshold=0.1, depth_threshold=0.8, window='', threshold='', **kwargs): 

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
        t_peaks, properties = find_peaks(adjusted, prominence=(10, 1000))
        t_peaks, properties = filter_predictions(
            t_peaks, properties,
            pd_dataframe,
            bodypart,
            likelihood_threshold,
            depth_threshold)
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
        depth = (
            (np.array(pd_dataframe[f'{bodypart} y'][t_peaks]) -
                np.array(pd_dataframe[f'{bodypart} y'][starts])) +
            (np.array(pd_dataframe[f'{bodypart} y'][t_peaks]) -
                np.array(pd_dataframe[f'{bodypart} y'][ends]))) / 2
    return depth


def sort_list(list1, list2):
    '''
    sort list2 according to list1
    '''
    result = [x for _, x in sorted(zip(list1, list2))] 
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
    and 10^2 ≤ λ ≤ 10^9,
    but exceptions may occur.
    In any case one should vary λ on a grid that is
    approximately linear for log λ"
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


def make_output(pathname, pd_dataframe, t_footfalls, depth_footfalls, start_footfalls, end_footfalls, bodyparts, slip_falls, frame_rate, confirmed=[], confirmed_only=False):
    
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
                    depths.append(
                        calculate_depths(
                            pd_dataframe,
                            bodypart,
                            start_footfalls[i],
                            end_footfalls[i],
                            t_footfalls[i]))
                except TypeError:
                    # missing start and end?
                    depths.append(np.nan)
                try:
                    duration.append(
                        round((end_footfalls[i] - start_footfalls[i]) / frame_rate, 3))
                except TypeError:
                    duration.append(np.nan)
                try:
                    starts.append(start_footfalls[i])
                except TypeError:
                    starts.append(np.nan)
                try:
                    ends.append(end_footfalls[i])
                except TypeError:
                    ends.append(np.nan)

        df_output = pd.DataFrame({
            'time (frame)': ts,
            'depth (pixel)': depths,
            'start (frame)': starts,
            'end (frame)': ends,
            'duration (s)': duration,
            'bodypart': bds,
            'slip or fall': slip_or_falls})

    else:
        for i, bodypart in enumerate(bodyparts):
            depth_footfalls[i] = calculate_depths(
                pd_dataframe,
                bodypart,
                start_footfalls[i],
                end_footfalls[i],
                t_footfalls[i])
            duration.append(round((end_footfalls[i] - start_footfalls[i]) / frame_rate, 3))

        df_output = pd.DataFrame({
            'time (frame)': t_footfalls,
            'depth (pixel)': depth_footfalls,
            'start (frame)': start_footfalls,
            'end (frame)': end_footfalls,
            'duration (s)': duration,
            'bodypart': bodyparts})

    df_output.to_csv(pathname, index=False)


def load_video(filename):

    vidcap = cv2.VideoCapture(filename)
    return vidcap


def plot_frame(video_file, n_frame, width, height, frame_rate, pd_dataframe, bodypart, zoom):

    try:
        figure = mpl.figure.Figure(
            figsize=(width, height), tight_layout=True, facecolor='none')
        axes = figure.add_subplot(111)
        axes.margins(x=0)

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


def plot_labels(pd_dataframe, n_current_frame, method, t_pred, start_pred, end_pred, width, height, bodypart, bodypart_list, selected_bodyparts, axis, likelihood_threshold, confirmed, zoom=True):
    
    if zoom:
        figure = mpl.figure.Figure(figsize=(width, height), facecolor='none')
    else:
        figure = mpl.figure.Figure(
            figsize=(width, height), tight_layout=True, facecolor='none')
    axes = figure.add_subplot(111)
    axes.margins(x=0)
    axes.xaxis.set_label_position('top')

    for bp in selected_bodyparts:
        axes.scatter(
            pd_dataframe['bodyparts coords'],
            pd_dataframe[f'{bp} {axis}'],
            s=0.1, color='steelblue')

    axes.scatter(
        pd_dataframe['bodyparts coords'],
        pd_dataframe[f'{bodypart} {axis}'],
        s=1)
    axes.scatter(
        pd_dataframe['bodyparts coords'].iloc[n_current_frame],
        pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame],
        marker='x', c='r')

    if n_current_frame in t_pred:
        index = t_pred.index(n_current_frame)
        try:
            if bodypart_list[index] == bodypart:
                axes.scatter(
                    start_pred[index],
                    pd_dataframe[f'{bodypart} {axis}'].iloc[start_pred[index]],
                    s=5, color='r')
                axes.scatter(
                    end_pred[index],
                    pd_dataframe[f'{bodypart} {axis}'].iloc[end_pred[index]],
                    s=5, color='r')
                axes.annotate(
                    'Start',
                    (start_pred[index],
                        pd_dataframe[f'{bodypart} {axis}'].iloc[start_pred[index]]))
                axes.annotate(
                    'End',
                    (end_pred[index],
                        pd_dataframe[f'{bodypart} {axis}'].iloc[end_pred[index]]))
        except TypeError:
            # user hasn't selected start and end time for added footfalls!
            pass

    axes.scatter(
        pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < likelihood_threshold]['bodyparts coords'], \
        pd_dataframe[pd_dataframe[f'{bodypart} likelihood'] < likelihood_threshold][f'{bodypart} {axis}'],
        s=1, c='0.7')
    axes.invert_yaxis()

    count_footfalls = 0
    for i, t in enumerate(t_pred):
        bp = bodypart_list[i]
        if bp == np.nan:
            bp = bodypart
        if confirmed[i]:
            count_footfalls += 1
            c = 'g'
            axes.annotate(
                str(count_footfalls),
                (t, pd_dataframe[f'{bp} {axis}'][t]), color=c)
        else:
            pass

    if zoom:
        x_max = len(pd_dataframe)
        if n_current_frame <= 300:
            axes.set_xlim(0, 600)
        elif n_current_frame >= x_max-300:
            axes.set_xlim(x_max-600, x_max)
        else:
            axes.set_xlim(n_current_frame-300, n_current_frame+300)

        axes.set_ylim(
            pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame]+100,
            pd_dataframe[f'{bodypart} {axis}'].iloc[n_current_frame]-100)

    return figure


def find_neighbors(n_current_frame, t_pred, start=None, end=None):

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
        return find_closest_neighbors(
            n_current_frame, t_pred, start=start, end=end)


def find_closest_neighbors(n_current_frame, t_pred, start=None, end=None):
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


def find_confirmed_neighbors(n_current_frame, t_val, confirmed=None, start=None, end=None):

    t_val_confirmed = np.array(t_val)[np.array(confirmed) == 1]
    if start is not None:
        try:
            start = list(
                t_val_confirmed).index(t_val[start])
            return find_neighbors(
                n_current_frame, t_val_confirmed, start=start)
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

    panel.prev_pred_button.Enable()
    panel.next_pred_button.Enable()
    panel.prev10_button.Enable()
    panel.next10_button.Enable()
    panel.prev_button.Enable()
    panel.next_button.Enable()
    panel.to_start_button.Enable()
    panel.to_end_button.Enable()

    if panel.n_frame <= panel.t_val[0]:
        panel.prev_pred_button.Disable()
    elif panel.n_frame >= panel.t_val[-1]:
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


def DisplayPlots(panel, set_bodypart=True):

    if panel.n_frame in panel.t_val and set_bodypart:
        panel.bodypart = panel.bodypart_list_val[panel.t_val.index(panel.n_frame)]
        panel.bodypart_to_plot.SetValue(panel.bodypart)

    try:
        frame = plot_frame(
            panel.video, panel.n_frame, 
            6, 3,
            int(panel.frame_rate),
            panel.df, panel.bodypart, panel.zoom_image)
        frame_canvas = FigureCanvas(panel, -1, frame)
        panel.frame_canvas.Hide()
        panel.sizer_2.Replace(panel.frame_canvas, frame_canvas)
        panel.frame_canvas = frame_canvas
        panel.sizer_2_widgets.append(panel.frame_canvas)
        panel.frame_canvas.Show()

        graph = plot_labels(
            panel.df, panel.n_frame,
            panel.method_selection, panel.t_val, panel.start_val,
            panel.end_val,
            (panel.window_width-60) // 100,
            (panel.window_height // 3) // 100,
            panel.bodypart, panel.bodypart_list_val,
            panel.selected_bodyparts, 'y',
            panel.likelihood_threshold, panel.confirmed, panel.zoom)
        graph_canvas = FigureCanvas(panel, -1, graph)
        panel.graph_canvas.Hide()
        panel.sizer_2.Replace(panel.graph_canvas, graph_canvas)
        panel.graph_canvas = graph_canvas
        panel.sizer_2_widgets.append(panel.graph_canvas)
        panel.graph_canvas.Show()
        panel.Fit()

        panel.SetSizer(panel.sizer_2)
        ControlPrediction(panel)
        panel.GetParent().Layout()

    except AttributeError:
        pass
