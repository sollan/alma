from ast import parse
import pandas as pd
from scipy.signal import peak_widths, find_peaks, butter, filtfilt
from scipy.spatial.distance import euclidean
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import math
import fastdtw
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
        if column.endswith(' y'):
            bodyparts.append(column.strip(' y'))
        
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
                 change_threshold = 0, treadmill_speed = None, frame_rate = 119.88, threshold = 540, **kwargs):

    start_times = []
    end_times = []
    durations = []

    if method == "Rate of change":
#         axis = 'x'

#         mean_bodypart_x = pd_dataframe[f'{bodypart} {axis}'].rolling(rolling_window).mean() if rolling_window is not None else pd_dataframe[f'{bodypart} {axis}']
#         mean_bodypart_x_change = np.diff(mean_bodypart_x)

#         is_stance = [i <= change_threshold for i in mean_bodypart_x_change]

#         if not is_stance[0]:
#             start_times.append(0)

#         for i in range(1, len(is_stance)):
#             if is_stance[i] != is_stance[i-1]: # change of stance / swing status
#                 if not is_stance[i]: # from stance to not stance
#                     end_times.append(i-1)
#                     start_times.append(i)
#                 # else: # from not stance to stance
#                     # end_times.append(i)
#                     # pass
#         if is_stance[-1]:
#             end_times.append(len(is_stance))
        axis = 'x'

#         mean_bodypart_x = pd_dataframe[f'{bodypart} {axis}'].rolling(rolling_window).mean() if rolling_window is not None else pd_dataframe[f'{bodypart} {axis}']
        fc = 6  # Cut-off frequency of the filter
        w = fc / (frame_rate / 2) # Normalize the frequency
        b, a = butter(5, w, 'low')
        filtered_bodypart_x = filtfilt(b, a, pd_dataframe[f'{bodypart} x'])
        filtered_bodypart_x_change = np.diff(filtered_bodypart_x)
        # movement direction: x loc decreasing; treadmill direction: x loc increasing
        # x loc change > 0 -> treadmill movement dominates (limb is stable)
        is_stance = [i >= change_threshold for i in filtered_bodypart_x_change]
#         print(is_stance, filtered_bodypart_x_change)
        for i in range(1, len(is_stance)):
            if is_stance[i] != is_stance[i-1]: # change of stance / swing status
                if is_stance[i] and pd_dataframe[f'{bodypart} y'].iloc[i] >= threshold: 
                # from "not stance" to "stance", and y loc also lower than threshold (opencv reverses axis)
                # i.e., "touching" treadmill
                    start_times.append(i)
                elif is_stance[i] and pd_dataframe[f'{bodypart} y'].iloc[i] < threshold: 
                    is_stance[i] = False
                    
        for i in range(1,len(start_times)):
            end_times.append(start_times[i]-1)
            
        end_times.append(len(is_stance))
        

    elif method == 'Threshold':
        '''
        when there is a clear cut off pixel value for the entire video, 
        e.g. location of treadmill in frame
        '''
        axis = 'y'

        if threshold is None:  
            threshold = np.percentile(pd_dataframe[f'{bodypart} {axis}'], 50)
        # print(threshold)
        # pd_dataframe[f'{bodypart} {axis}'] = pd_dataframe[f'{bodypart} {axis}'].rolling(rolling_window).mean() if rolling_window is not None else pd_dataframe[f'{bodypart} {axis}']

        # y axis loc larger than threshold == limb touching treadmill == end of stride        
        on_treadmill = [i >= threshold for i in pd_dataframe[f'{bodypart} {axis}']]
        # print(on_treadmill)
#         if not on_treadmill[0]:
#             start_times.append(0)
#         for i in range(1, len(on_treadmill)):
#             if on_treadmill[i] != on_treadmill[i-1]: # change of stance / swing status
#                 if not on_treadmill[i]: # from not stance to stance
#                     # end_times.append(i)
#                     # pass
#                 # else: # from stance to not stance
#                     end_times.append(i-1)
#                     start_times.append(i)
#         if on_treadmill[-1]:
#             end_times.append(len(on_treadmill))

        for i in range(1, len(on_treadmill)):
            if on_treadmill[i] != on_treadmill[i-1]: # change of stance / swing status
                if on_treadmill[i]: # from stance to not stance
                    start_times.append(i)
                    
        for i in range(1,len(start_times)):
            end_times.append(start_times[i]-1)
            
        end_times.append(len(on_treadmill))
        print(start_times, end_times)
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




'''
Functions to calculate kinematic parameters
'''

def angle(vector1, vector2):
    length1 = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1])
    length2 = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1])
    return math.acos((vector1[0] * vector2[0] + vector1[1] * vector2[1])/ (length1 * length2))


def find_angles(x1, y1, x_center, y_center, x2, y2):
    x1 = x1-x_center
    y1 = y1-y_center
    x2 = x2-x_center
    y2 = y2-y_center
    return [np.rad2deg(angle([x1[i], y1[i]], [x2[i], y2[i]])) for i in range(len(x1))]


def butterworth_filter(bodypart_loc, frame_rate, cutoff_f):
    w = cutoff_f / (frame_rate / 2) # rormalize the frequency
    b, a = butter(5, w, 'low')
    filtered_bodypart_loc = filtfilt(b, a, bodypart_loc)
    return filtered_bodypart_loc


def find_stride_len(toe_x, start, end):
    '''
    difference between the starting and ending x-axis position of toe
    (in pixels)
    '''
    return toe_x[end] - toe_x[start]


def find_swing_stance(elevation_angle_change, start, end):
    '''
    return stance duration in frames, swing duration, stance duration
    '''
    try:
        # only when swing phase is detected (neg-to-pos zero crossing exists in elevation angle delta)
        # stride start to swing onset = stance duration
        # swing onset to stride end = swing duration
        # since stride is marked based on the consequtive stance onsets (toe y axis movement)
        stance_dur = np.where(np.diff(np.sign(elevation_angle_change[start:end])) > 0)[0][0]
    except IndexError:
        # might be a false stride or only drag
        print(f'No swing phase found in this step cycle! Marking full cycle as stance. (Stride starting at frame {start})')
        stance_dur = end-start

    return stance_dur, (end-start-stance_dur)/(end-start), stance_dur/(end-start)


def find_limb_length(x1, y1, x2, y2):
    '''
    calculate the Euclidean distance between two endpoints of a limb
    using 4 lists
    '''
    return [math.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2) for i in range(len(x1))]


def find_drag(toe_y, stance_dur, threshold, start, end):
    # threshold: % threshold for y axis loc (treadmill y level), below which the toe is considered to be in contact with treadmill
    is_drag = toe_y[(start+stance_dur):end] > threshold
    return sum(np.where(is_drag, [1], [0])), sum(np.where(is_drag, [1], [0])) / (end - stance_dur)


def find_euclidean_speed(x, y, frame_rate):
    loc_changes = []
    for i in range(1, len(x)):
        loc_changes.append(np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) / frame_rate)
    return loc_changes


def parse_stride_lists(x, starts, ends, first_stride, n):
    '''
    reshapes a continuous time series of bodypart location x into a list of n strides, 
    each starting at starts[n] and ending at ends[n], with the first stride in the new list
    starting at starts[first_stride]
    starts / ends: index of the stride start and endpoints to use for parsing continuous time series x
    first_stride: index of the first stride to extract, based on the starts and ends lists
    n: number of strides to extract from x
    returns a list of strides
    '''
    return [x[starts[i]:ends[i]] for i in range(first_stride, first_stride+n)]


def pairwise_dtw(iterable):
    '''
    calculates the dynamic time warping distance between every two strides
    iterable: a list / array of time series (endpoint paths of a number of strides)
    returns: 
    dtw_result: a list of DTW distance values
    e.g., iterable = [stride_a, stride_b, stride_c]
    -> dtw_result = [distance_ab, distance_ac, distance_bc]
    '''
    dtw_result = []
    for i in range(len(iterable)-1):
        for j in range(i+1, len(iterable)):
            distance, _ = fastdtw.fastdtw(iterable[i], iterable[j], dist=euclidean)
            dtw_result.append(distance)
    return dtw_result


def extract_parameters(frame_rate, pixels_per_cm, pd_dataframe, stance_threshold, treadmill_y, cutoff_f, starts, ends):
    '''
    stance_threshold: y-axis threshold, above which--in reality below 
    which (bottom pixels have large y axis values in opencv)--the motion 
    is considered stance

    treadmill_y: y axis level of the treadmill (or the runway) height, for 
    step height calculation only
    '''
    # edit find_drag to make use of treadmill_y_level if provided
    cycle_dur_secs = []
    cycle_dur_frames = []
    cycle_vs = []
    
    stride_lens = []
    stance_dur_secs = []
    swing_dur_secs = []
    swing_percentages = []
    stance_percentages = []

    limb_len_means = []
    limb_len_maxs = []
    limb_len_mins = []
    limb_len_sds = []
    
    step_heights = []
    max_v_during_swings = []
    
    mtp_joint_extensions = []
    ankle_joint_extensions = []
    knee_joint_extensions = []
    hip_joint_extensions = []
    
    mtp_joint_flexions = []
    ankle_joint_flexions = []
    knee_joint_flexions = []
    hip_joint_flexions = []
    
    mtp_joint_amplitudes = []
    ankle_joint_amplitudes = []
    knee_joint_amplitudes = []
    hip_joint_amplitudes = []
    
    drag_ts = []
    drag_percentages = []

    dtw_x_plane_5_means = [] # dynamic time warping: measures spatial variability
    dtw_x_plane_5_sds = []
    dtw_y_plane_5_means = []
    dtw_y_plane_5_sds = []
    dtw_xy_plane_5_means = []
    dtw_xy_plane_5_sds = []

    dtw_x_plane_10_means = []
    dtw_x_plane_10_sds = []
    dtw_y_plane_10_means = []
    dtw_y_plane_10_sds = []
    dtw_xy_plane_10_means = []
    dtw_xy_plane_10_sds = []

    # should use a dictionary here
    smooth_toe_x = butterworth_filter(pd_dataframe['toe x'], frame_rate, cutoff_f)
    smooth_mtp_x = butterworth_filter(pd_dataframe['mtp x'], frame_rate, cutoff_f)
    smooth_ankle_x = butterworth_filter(pd_dataframe['ankle x'], frame_rate, cutoff_f)
    smooth_knee_x = butterworth_filter(pd_dataframe['knee x'], frame_rate, cutoff_f)
    smooth_hip_x = butterworth_filter(pd_dataframe['hip x'], frame_rate, cutoff_f)
    smooth_crest_x = butterworth_filter(pd_dataframe['iliac crest x'], frame_rate, cutoff_f)

    smooth_toe_y = butterworth_filter(pd_dataframe['toe y'], frame_rate, cutoff_f)
    smooth_mtp_y = butterworth_filter(pd_dataframe['mtp y'], frame_rate, cutoff_f)
    smooth_ankle_y = butterworth_filter(pd_dataframe['ankle y'], frame_rate, cutoff_f)
    smooth_knee_y = butterworth_filter(pd_dataframe['knee y'], frame_rate, cutoff_f)
    smooth_hip_y = butterworth_filter(pd_dataframe['hip y'], frame_rate, cutoff_f)
    smooth_crest_y = butterworth_filter(pd_dataframe['iliac crest y'], frame_rate, cutoff_f)

    angles_toe_mtp_ankle = find_angles(
            smooth_toe_x, smooth_toe_y, 
            smooth_mtp_x, smooth_mtp_y, 
            smooth_ankle_x, smooth_ankle_y
        )
    angles_mtp_ankle_knee = find_angles(
            smooth_mtp_x, smooth_mtp_y, 
            smooth_ankle_x, smooth_ankle_y,
            smooth_knee_x, smooth_knee_y
        )
    angles_ankle_knee_hip = find_angles(
            smooth_ankle_x, smooth_ankle_y,
            smooth_knee_x, smooth_knee_y,
            smooth_hip_x, smooth_hip_y
        )
    angles_knee_hip_crest = find_angles(
            smooth_knee_x, smooth_knee_y,
            smooth_hip_x, smooth_hip_y,
            smooth_crest_x, smooth_crest_y
        )
    elevation_angles = find_angles(
            smooth_mtp_x, smooth_mtp_y, 
            smooth_crest_x, smooth_crest_y, 
            smooth_crest_x, np.zeros(len(smooth_crest_y))
        )
    elevation_angle_change = np.diff(elevation_angles)

    limb_lens = find_limb_length(smooth_toe_x, smooth_toe_y, smooth_hip_x, smooth_hip_y)

    velocities = find_euclidean_speed(smooth_toe_x, smooth_toe_y, frame_rate)
    
    for i in range(len(starts)):
        
        cycle_dur_frame = ends[i] - starts[i]
        cycle_dur_secs.append(cycle_dur_frame / frame_rate)
        cycle_dur_frames.append(cycle_dur_frame)

        stride_len = find_stride_len(smooth_toe_x, starts[i], ends[i])
        stride_lens.append(stride_len/pixels_per_cm)
        
        cycle_v = stride_len / cycle_dur_frame / frame_rate
        cycle_vs.append(cycle_v / pixels_per_cm)
        # or use averaged v from euclidean distance between every two frames? 
        # (that would consider the whole trail travelled, not just the distance between start and destination)

        stance_dur_frame, swing_perc, stance_perc = find_swing_stance(elevation_angle_change, starts[i], ends[i])
        stance_dur_secs.append(stance_dur_frame / frame_rate)
        swing_dur_secs.append((cycle_dur_frame - stance_dur_frame) / frame_rate)
        
        swing_percentages.append(swing_perc)
        stance_percentages.append(stance_perc)
        
        limb_len_means.append(np.mean(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
        limb_len_maxs.append(max(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
        limb_len_mins.append(min(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
        limb_len_sds.append(np.std(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)

        step_heights.append((treadmill_y - min(smooth_toe_y[starts[i]:ends[i]])) / pixels_per_cm) 

        # max step height (opencv reverses y axis, therefore the MIN())
        if stance_dur_frame == cycle_dur_frame:
            max_v_during_swings.append(0)
        else:
            max_v_during_swings.append(max(velocities[(starts[i]+stance_dur_frame):ends[i]]) / pixels_per_cm)
        
        mtp_joint_extension = max(angles_toe_mtp_ankle[starts[i] : ends[i]])
        ankle_joint_extension = max(angles_mtp_ankle_knee[starts[i] : ends[i]])
        knee_joint_extension = max(angles_ankle_knee_hip[starts[i] : ends[i]])
        hip_joint_extension = max(angles_knee_hip_crest[starts[i] : ends[i]])
        
        mtp_joint_flexion = min(angles_toe_mtp_ankle[starts[i] : ends[i]])
        ankle_joint_flexion = min(angles_mtp_ankle_knee[starts[i] : ends[i]])
        knee_joint_flexion = min(angles_ankle_knee_hip[starts[i] : ends[i]])
        hip_joint_flexion = min(angles_knee_hip_crest[starts[i] : ends[i]])
        
        mtp_joint_amplitude = mtp_joint_extension - mtp_joint_flexion
        ankle_joint_amplitude = ankle_joint_extension - ankle_joint_flexion
        knee_joint_amplitude = knee_joint_extension - knee_joint_flexion
        hip_joint_amplitude = hip_joint_extension - hip_joint_flexion
        
        mtp_joint_extensions.append(mtp_joint_extension)
        mtp_joint_flexions.append(mtp_joint_flexion)
        mtp_joint_amplitudes.append(mtp_joint_amplitude)

        ankle_joint_extensions.append(ankle_joint_extension)
        ankle_joint_flexions.append(ankle_joint_flexion)
        ankle_joint_amplitudes.append(ankle_joint_amplitude)
        
        knee_joint_extensions.append(knee_joint_extension)
        knee_joint_flexions.append(knee_joint_flexion)
        knee_joint_amplitudes.append(knee_joint_amplitude)
        
        hip_joint_extensions.append(hip_joint_extension)
        hip_joint_flexions.append(hip_joint_flexion)
        hip_joint_amplitudes.append(hip_joint_amplitude)
        
        drag, drag_percent = find_drag(smooth_toe_x, stance_dur_frame, stance_threshold, starts[i], ends[i])
        drag_ts.append(drag/frame_rate)
        drag_percentages.append(drag_percent)

        if i <= len(starts)-5:
            toe_xs = parse_stride_lists(smooth_toe_x, starts, ends, i, 5)
            pairwise_dtw_res_x_5 = pairwise_dtw(toe_xs)
            dtw_x_plane_5_mean = np.mean(pairwise_dtw_res_x_5)
            dtw_x_plane_5_sd = np.std(pairwise_dtw_res_x_5)

            toe_ys = parse_stride_lists(smooth_toe_y, starts, ends, i, 5)
            pairwise_dtw_res_y_5 = pairwise_dtw(toe_ys)
            dtw_y_plane_5_mean = np.mean(pairwise_dtw_res_y_5)
            dtw_y_plane_5_sd = np.std(pairwise_dtw_res_y_5)

            toe_xys = [np.append(toe_xs[j], toe_ys[j]).reshape(2,-1).T for j in range(5)]
            pairwise_dtw_res_xy_5 = pairwise_dtw(toe_xys)
            dtw_xy_plane_5_mean = np.mean(pairwise_dtw_res_xy_5)
            dtw_xy_plane_5_sd = np.std(pairwise_dtw_res_xy_5)

            if i <= len(starts)-10:
                toe_xs = parse_stride_lists(smooth_toe_x, starts, ends, i, 10)
                pairwise_dtw_res_x_10 = pairwise_dtw(toe_xs)
                dtw_x_plane_10_mean = np.mean(pairwise_dtw_res_x_10)
                dtw_x_plane_10_sd = np.std(pairwise_dtw_res_x_10)

                toe_ys = parse_stride_lists(smooth_toe_y, starts, ends, i, 10)
                pairwise_dtw_res_y_10 = pairwise_dtw(toe_ys)
                dtw_y_plane_10_mean = np.mean(pairwise_dtw_res_y_10)
                dtw_y_plane_10_sd = np.std(pairwise_dtw_res_y_10)

                toe_xys = [np.append(toe_xs[j], toe_ys[j]).reshape(2,-1).T for j in range(10)]
                pairwise_dtw_res_xy_10 = pairwise_dtw(toe_xys)
                dtw_xy_plane_10_mean = np.mean(pairwise_dtw_res_xy_10)
                dtw_xy_plane_10_sd = np.std(pairwise_dtw_res_xy_10)

                dtw_x_plane_10_means.append(dtw_x_plane_10_mean)
                dtw_x_plane_10_sds.append(dtw_x_plane_10_sd)
                dtw_y_plane_10_means.append(dtw_y_plane_10_mean)
                dtw_y_plane_10_sds.append(dtw_y_plane_10_sd)
                dtw_xy_plane_10_means.append(dtw_xy_plane_10_mean)
                dtw_xy_plane_10_sds.append(dtw_xy_plane_10_sd)
            else:
                dtw_x_plane_10_means.append(np.nan)
                dtw_x_plane_10_sds.append(np.nan)
                dtw_y_plane_10_means.append(np.nan)
                dtw_y_plane_10_sds.append(np.nan)
                dtw_xy_plane_10_means.append(np.nan)
                dtw_xy_plane_10_sds.append(np.nan)
                
            dtw_x_plane_5_means.append(dtw_x_plane_5_mean)
            dtw_x_plane_5_sds.append(dtw_x_plane_5_sd)
            dtw_y_plane_5_means.append(dtw_y_plane_5_mean)
            dtw_y_plane_5_sds.append(dtw_y_plane_5_sd)
            dtw_xy_plane_5_means.append(dtw_xy_plane_5_mean)
            dtw_xy_plane_5_sds.append(dtw_xy_plane_5_sd)
        else:
            dtw_x_plane_5_means.append(np.nan)
            dtw_x_plane_5_sds.append(np.nan)
            dtw_y_plane_5_means.append(np.nan)
            dtw_y_plane_5_sds.append(np.nan)
            dtw_xy_plane_5_means.append(np.nan)
            dtw_xy_plane_5_sds.append(np.nan)
            dtw_x_plane_10_means.append(np.nan)
            dtw_x_plane_10_sds.append(np.nan)
            dtw_y_plane_10_means.append(np.nan)
            dtw_y_plane_10_sds.append(np.nan)
            dtw_xy_plane_10_means.append(np.nan)
            dtw_xy_plane_10_sds.append(np.nan)
        
    return pd.DataFrame(data=np.array([cycle_dur_secs,
            cycle_dur_frames,
            cycle_vs,
            stride_lens, 
            stance_dur_secs, 
            swing_dur_secs,
            swing_percentages, 
            stance_percentages, 
            limb_len_means,
            limb_len_maxs,
            limb_len_mins,
            limb_len_sds,
            step_heights,
            max_v_during_swings,
            mtp_joint_extensions,
            mtp_joint_flexions,
            mtp_joint_amplitudes,
            ankle_joint_extensions,
            ankle_joint_flexions,
            ankle_joint_amplitudes,
            knee_joint_extensions,
            knee_joint_flexions,
            knee_joint_amplitudes,
            hip_joint_extensions,
            hip_joint_flexions,
            hip_joint_amplitudes,            
            drag_ts, 
            drag_percentages,
            dtw_x_plane_5_means,
            dtw_x_plane_5_sds,
            dtw_y_plane_5_means,
            dtw_y_plane_5_sds,
            dtw_xy_plane_5_means,
            dtw_xy_plane_5_sds,
            dtw_x_plane_10_means,
            dtw_x_plane_10_sds,
            dtw_y_plane_10_means,
            dtw_y_plane_10_sds,
            dtw_xy_plane_10_means,
            dtw_xy_plane_10_sds            
            ]).T, 
                        columns=[
                            'cycle duration (s)',
                            'cycle duration (no. frames)',
                            'cycle velocity (cm/s)',
                            'stride length (cm)', 
                            'stance duration (s)', 
                            'swing duration (s)',
                            'swing percentage (%)', 
                            'stance percentage (%)', 
                            'mean toe-to-crest distance (cm)',
                            'max toe-to-crest distance (cm)',
                            'min toe-to-crest distance (cm)',
                            'toe-to-crest distance SD (cm)',
                            'step height (cm)',
                            'max velocity during swing (cm/s)',
                            'mtp joint extension (deg)',
                            'mtp joint flexion (deg)',
                            'mtp joint amplitude (deg)',
                            'ankle joint extension (deg)',
                            'ankle joint flexion (deg)',
                            'ankle joint amplitude (deg)',
                            'knee joint extension (deg)',
                            'knee joint flexion (deg)',
                            'knee joint amplitude (deg)',
                            'hip joint extension (deg)',
                            'hip joint flexion (deg)',
                            'hip joint amplitude (deg)',            
                            'drag duration (s)', 
                            'drag percentage (%)',
                            'DTW distance x plane 5 frames mean',
                            'DTW distance x plane 5 frames SD',
                            'DTW distance y plane 5 frames mean',
                            'DTW distance y plane 5 frames SD',
                            'DTW distance xy plane 5 frames mean',
                            'DTW distance xy plane 5 frames SD',
                            'DTW distance x plane 10 frames mean',
                            'DTW distance x plane 10 frames SD',
                            'DTW distance y plane 10 frames mean',
                            'DTW distance y plane 10 frames SD',
                            'DTW distance xy plane 10 frames mean',
                            'DTW distance xy plane 10 frames SD',
                        ])


def make_parameters_output(pathname, parameters):
    parameters.to_csv(pathname)