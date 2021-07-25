import pandas as pd
import scipy
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean
import numpy as np
import math
import fastdtw
import statistics as stat
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

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
        if column.endswith(' y'):
            bodyparts.append(column.strip(' y'))
        
    pd_dataframe.columns = col_names
    
    return pd_dataframe, bodyparts


def treadmill_correction(pd_dataframe, bodyparts, px_speed = 8.09):

    correction = np.arange(0, len(pd_dataframe), 1)
    correction = correction * px_speed

    if type(bodyparts) is list and len(bodyparts) > 1:
        for bodypart in bodyparts:
            pd_dataframe[f'{bodypart} x'] = -(pd_dataframe[f'{bodypart} x'] - correction)
    
    elif type(bodyparts) is list and len(bodyparts) == 1:
        pd_dataframe[f'{bodyparts[0]} x'] = -(pd_dataframe[f'{bodyparts[0]} x'] - correction)
    
    elif type(bodyparts) is str:
        pd_dataframe[f'{bodyparts} x'] = -(pd_dataframe[f'{bodyparts} x'] - correction)

    return pd_dataframe


def estimate_speed(pd_dataframe, bodypart, cm_speed, px_to_cm_speed_ratio, frame_rate):

    x_change = np.diff(pd_dataframe[f'{bodypart} x'][pd_dataframe[f'{bodypart} likelihood']>0.5])
    x_change_filt = x_change[(x_change < np.mean(x_change) + 1*np.std(x_change)) & (x_change > np.mean(x_change) - 1*np.std(x_change))]

    px_speed, _ = scipy.stats.norm.fit(x_change_filt[x_change_filt>0])

    if cm_speed is None: 
        cm_speed = px_speed / px_to_cm_speed_ratio
    else:
        px_to_cm_speed_ratio = px_speed / cm_speed

    pixels_per_cm = 1 / (cm_speed / px_speed / frame_rate)

    return cm_speed, px_speed, pixels_per_cm, px_to_cm_speed_ratio


def make_output(pathname, start_times, end_times, durations):

    df_output = pd.DataFrame({'stride_start': start_times,
                            'stride_end': end_times,
                            'stride_duration': durations})

    df_output.to_csv(pathname, index = False)



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
    w = cutoff_f / (frame_rate / 2) # normalize the frequency
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
        # print(f'No swing phase found in this step cycle! Marking full cycle as stance. (Stride starting at frame {start})')
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
    '''
    calculate speed in pixel/s
    '''
    loc_changes = []
    for i in range(1, len(x)):
        loc_changes.append(np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) * frame_rate)
    return loc_changes


def parse_stride_lists(x, starts, ends, first_stride, n, adjust = False):
    '''
    reshapes a continuous time series of bodypart location x into a list of n strides, 
    each starting at starts[n] and ending at ends[n], with the first stride in the new list
    starting at starts[first_stride]
    starts / ends: index of the stride start and endpoints to use for parsing continuous time series x
    first_stride: index of the first stride to extract, based on the starts and ends lists
    n: number of strides to extract from x
    returns a list of strides
    '''
    results = [x[starts[i]:ends[i]+1] for i in range(first_stride, first_stride+n)]
    if adjust:
        for n_sublist in range(len(results)):
            centers = [results[n_sublist][0]]*len(results[n_sublist])
            results[n_sublist] = [a - b for a, b in zip(results[n_sublist], centers)]
    return results



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


def make_parameters_output(pathname, parameters):

    parameters.to_csv(pathname)


def make_averaged_output(pathname, truncated=False):

    files = []
    for file in os.listdir(pathname):
        if truncated:
            if file.endswith('.csv') and file.startswith('10_continuous_strides_parameters_'):
                files.append(file)
        else:
            if file.endswith('.csv') and file.startswith('parameters_'):
                files.append(file)

    dfs = []
    for file in files:

        if truncated:
            input_name = file.split('10_continuous_strides_parameters_')[1].split('.')[0]
        else:
            input_name = file.split('parameters_')[1].split('.')[0]

        path = os.path.join(pathname, file)
        df = pd.read_csv(path)
        df['id'] = input_name

        dfs.append(df)

    res = pd.concat(dfs).groupby('id').agg(['mean','std'])
    res = res.drop(['Unnamed: 0','stride_start (frame)','stride_end (frame)'], axis=1, errors='ignore')
    if truncated:  
        res.to_csv(os.path.join(pathname, 'averaged_truncated_results.csv'))
    else:
        res.to_csv(os.path.join(pathname, 'averaged_results.csv'))
    
def convert_to_binary(A):
    
    list_valid = [0 if np.isnan(i) else 1 for i in A]
    
    return list_valid


def findLongestSequence(A, k):
    '''
    Function to find the maximum sequence of continuous 1's by replacing
    at most `k` zeroes by 1 using sliding window technique
    '''
    left = 0        # represents the current window's starting index
    count = 0       # stores the total number of zeros in the current window
    window = 0      # stores the maximum number of continuous 1's found
                    # so far (including `k` zeroes)
    leftIndex = 0   # stores the left index of maximum window found so far

    for right in range(len(A)):
        if A[right] == 0:
            count = count + 1
        while count > k:
            if A[left] == 0:
                count = count - 1
            left = left + 1
            
        if right - left + 1 > window:
            window = right - left + 1
            leftIndex = left
    
    return leftIndex, leftIndex + window - 1


def return_ten_central(pd_dataframe_parameters, plot=False, pd_dataframe_coords=None, bodyparts=None, is_stance=[], filename=''):
    valid_list = convert_to_binary(pd_dataframe_parameters['cycle duration (s)'])
    start, end = findLongestSequence(valid_list, 0)
    start_stride = int(np.mean([start, end]))-5
    end_stride = start_stride+10
    if plot:
        x_coords, y_coords = collect_filtered_coords(pd_dataframe_coords, bodyparts)
        start_frame = int(pd_dataframe_parameters.iloc[start_stride]['stride_start (frame)'])
        end_frame = int(pd_dataframe_parameters.iloc[end_stride]['stride_end (frame)'])
        continuous_stickplot(pd_dataframe_coords, bodyparts, is_stance, x_coords, y_coords, start_frame, end_frame, filename)

    return pd_dataframe_parameters.iloc[start_stride:end_stride, :]


def extract_parameters(frame_rate, pd_dataframe, cutoff_f, bodypart, cm_speed = None, px_to_cm_speed_ratio = 0.4806, plot = False):
    '''
    pd_dataframe: contains raw coordinates (not adjusted for treadmill movement, unfiltered)
    bodypart: which bodypart to use for stride estimation

    '''
    starts = []
    ends = []
    durations = []
        
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
    
    mtp_joint_sds = []
    ankle_joint_sds = []
    knee_joint_sds = []
    hip_joint_sds = []
    
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
    
    x_change = np.diff(pd_dataframe[f'{bodypart} x'][pd_dataframe[f'{bodypart} likelihood']>0.5])
    x_change_filt = x_change[(x_change < np.mean(x_change) + 1*np.std(x_change)) & (x_change > np.mean(x_change) - 1*np.std(x_change))]

    px_speed, _ = scipy.stats.norm.fit(x_change_filt[x_change_filt>0])

    if cm_speed is None: 
        cm_speed = px_speed / px_to_cm_speed_ratio

    pixels_per_cm = 1 / (cm_speed / px_speed / frame_rate)

    bodyparts = ['toe', 'mtp', 'ankle', 'knee', 'hip', 'iliac crest']
    pd_dataframe = treadmill_correction(pd_dataframe, bodyparts, px_speed)

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
    
    # does gait appear normal? -> multimodal distribution from swing (body movement) and stance (treadmill)
    if np.abs(stat.median(x_change)) < 0.5 and len(np.where(np.abs(x_change) < 2)[0])/len(x_change) > 0.68:
        
        # bodypart x coordinate rate of change does not meet assumption of multimodal distribution
        # instead it is centered around 0
        # step cycle cannot be detected properly
        
        print('Too much dragging? Please check raw data.')
        print(f'Median of selected bodypart x coordinate change per frame: {stat.median(x_change)}')                           
        print('Calculating a subset of parameters independent of step cycles...')
        
        starts.append(np.nan)
        ends.append(np.nan)
        durations.append(np.nan)
        cycle_dur_secs.append(np.nan)
        cycle_dur_frames.append(np.nan)
        stride_lens.append(np.nan)
        cycle_vs.append(np.nan)
        stance_dur_secs.append(np.nan)
        swing_dur_secs.append(np.nan)
        swing_percentages.append(np.nan)
        stance_percentages.append(np.nan)
        limb_len_means.append(np.mean(limb_lens) / pixels_per_cm)
        limb_len_maxs.append(np.nan)
        limb_len_mins.append(np.nan)
        limb_len_sds.append(np.std(limb_lens) / pixels_per_cm)
        step_heights.append(0) 
        max_v_during_swings.append(np.nan)

        mtp_joint_extensions.append(np.nan)
        mtp_joint_flexions.append(np.nan)
        mtp_joint_amplitudes.append(np.nan)

        ankle_joint_extensions.append(np.nan)
        ankle_joint_flexions.append(np.nan)
        ankle_joint_amplitudes.append(np.nan)

        knee_joint_extensions.append(np.nan)
        knee_joint_flexions.append(np.nan)
        knee_joint_amplitudes.append(np.nan)

        hip_joint_extensions.append(np.nan)
        hip_joint_flexions.append(np.nan)
        hip_joint_amplitudes.append(np.nan)

        mtp_joint_sds.append(np.nan)
        ankle_joint_sds.append(np.nan)
        knee_joint_sds.append(np.nan)
        hip_joint_sds.append(np.nan)

        drag_ts.append(np.nan)
        drag_percentages.append(1)
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

    else:
        
        starts = []
        ends = []
        
        filtered_bodypart_x = butterworth_filter(pd_dataframe[f'{bodypart} x'], frame_rate, 5)
        filtered_bodypart_x_change = np.diff(filtered_bodypart_x)

        # movement direction: x loc decreasing; treadmill direction: x loc increasing
        # x loc change > 0 -> treadmill movement dominates (limb is stable)
        is_stance = [i >= 0 for i in filtered_bodypart_x_change]
        for i in range(1, len(is_stance)):
            if is_stance[i] != is_stance[i-1]:
                if is_stance[i]:
                    starts.append(i)
        for i in range(1,len(starts)):
            ends.append(starts[i]-1)
        ends.append(len(is_stance))

        y = pd_dataframe['toe y'][pd_dataframe['toe likelihood'] > 0.5]
        y_filt = y[(y < np.mean(y) + 1*np.std(y)) & (y > np.mean(y) - 1*np.std(y))]
        _, b = np.histogram(y_filt, bins=100, density=True)
        # fit 2 Gaussians to the pdf of toe y coord
        gm = GaussianMixture(n_components=2, random_state=0).fit(np.array(b).reshape(-1,1))
        # the Gaussian with larger mean corresponds to y coord during stance phase
        # use this mean as threshold to detect dragging during swing phase
        stance_threshold = max(gm.means_)
        # use this plus SD (i.e., lower in space) as treadmill y coord, to calculate step height
        treadmill_y = float(stance_threshold + np.sqrt(gm.covariances_[np.where(gm.means_ == stance_threshold)]))

        starts_included = []
        ends_included = []
        
        for i in range(len(starts)):
            
            stride_len = find_stride_len(smooth_toe_x, starts[i], ends[i]) # in pixel
            stride_len_cm = stride_len / pixels_per_cm
            limb_len_max = max(limb_lens[starts[i] : ends[i]]) / pixels_per_cm
            cycle_dur_frame = ends[i] - starts[i]
            cycle_v = stride_len / cycle_dur_frame * frame_rate # (px / frames) * (frames/s) = px/s

            stance_dur_frame, swing_perc, stance_perc = find_swing_stance(elevation_angle_change, starts[i], ends[i])
            swing_dur_sec = (cycle_dur_frame - stance_dur_frame) / frame_rate
                
            if swing_dur_sec and stance_dur_frame > 0:
                # min: image y axis starts from top
                step_height = -(min(smooth_toe_y[starts[i]+stance_dur_frame:ends[i]]) - \
                                max(smooth_toe_y[starts[i]:starts[i]+stance_dur_frame])) / pixels_per_cm
            elif swing_dur_sec and stance_dur_frame == 0:
                step_height = -(min(smooth_toe_y[starts[i]+stance_dur_frame:ends[i]]) - \
                                smooth_toe_y[starts[i]]) / pixels_per_cm
            else:
                # no swing phase!
                step_height = 0
                
        
            # does extracted stride seem accurate? (e.g., due to mislabeled bodypart position)
            # extreme limb length
            # extreme stride length
            # single frame "cycle" duration
            # extreme step height
            if limb_len_max < 15 and stride_len_cm < 8 and stride_len > 0 and \
                cycle_dur_frame > 1 and step_height < 1.5 and step_height > 0:
                
                starts_included.append(starts[i])
                ends_included.append(ends[i])
                
                limb_len_means.append(np.mean(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
                limb_len_maxs.append(limb_len_max)
                limb_len_mins.append(min(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
                limb_len_sds.append(np.std(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)

                cycle_dur_secs.append(cycle_dur_frame / frame_rate)
                cycle_dur_frames.append(cycle_dur_frame)

                stride_lens.append(stride_len_cm)
                
                step_heights.append(step_height)

                cycle_vs.append(cycle_v / pixels_per_cm) # (px/s) / (px/cm) = cm/s
                # or use averaged v from euclidean distance between every two frames? 
                # (that would consider the entire trajectory of a stride, including vertical movement / lift,
                # not just the distance between start and end point)

                stance_dur_secs.append(stance_dur_frame / frame_rate)
                swing_dur_secs.append(swing_dur_sec)
                swing_percentages.append(swing_perc)
                stance_percentages.append(stance_perc)
                
                if stance_dur_frame == cycle_dur_frame:
                    max_v_during_swings.append(np.nan)
                else:
                    max_v_during_swings.append(max(velocities[starts[i]:ends[i]]) / pixels_per_cm)

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

                mtp_joint_sds.append(np.std(angles_toe_mtp_ankle[starts[i] : ends[i]]))
                ankle_joint_sds.append(np.std(angles_mtp_ankle_knee[starts[i] : ends[i]]))
                knee_joint_sds.append(np.std(angles_ankle_knee_hip[starts[i] : ends[i]]))
                hip_joint_sds.append(np.std(angles_knee_hip_crest[starts[i] : ends[i]]))

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

                drag, drag_percent = find_drag(smooth_toe_y, stance_dur_frame, treadmill_y, starts[i], ends[i])
                drag_ts.append(drag/frame_rate)
                if swing_dur_sec and stance_dur_frame > 0:
                    drag_percentages.append((drag/frame_rate) / swing_dur_sec)    
                elif swing_dur_sec and stance_dur_frame == 0:
                    drag_percentages.append(np.nan)
                else:
                    # no swing phase detected!
                    drag_percentages.append(1)
                
            else:
                
                cycle_dur_secs.append(np.nan)
                cycle_dur_frames.append(np.nan)
                stride_lens.append(np.nan)
                cycle_vs.append(np.nan)
                stance_dur_secs.append(np.nan)
                swing_dur_secs.append(np.nan)
                swing_percentages.append(np.nan)
                stance_percentages.append(np.nan)
                limb_len_means.append(np.nan)
                limb_len_maxs.append(np.nan)
                limb_len_mins.append(np.nan)
                limb_len_sds.append(np.nan)
                step_heights.append(np.nan) 
                max_v_during_swings.append(np.nan)
                mtp_joint_extensions.append(np.nan)
                mtp_joint_flexions.append(np.nan)
                mtp_joint_amplitudes.append(np.nan)
                ankle_joint_extensions.append(np.nan)
                ankle_joint_flexions.append(np.nan)
                ankle_joint_amplitudes.append(np.nan)
                knee_joint_extensions.append(np.nan)
                knee_joint_flexions.append(np.nan)
                knee_joint_amplitudes.append(np.nan)
                hip_joint_extensions.append(np.nan)
                hip_joint_flexions.append(np.nan)
                hip_joint_amplitudes.append(np.nan)
                mtp_joint_sds.append(np.nan)
                ankle_joint_sds.append(np.nan)
                knee_joint_sds.append(np.nan)
                hip_joint_sds.append(np.nan)
                drag_ts.append(np.nan)
                drag_percentages.append(np.nan)
            
        starts_included = np.array(starts_included)
        ends_included = np.array(ends_included)
        
        for i in range(len(starts)):
            
            if starts[i] not in starts_included:
                
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
                
            else:
                included_index = int(np.where(starts[i] == starts_included)[0])                
                
                if included_index <= len(starts_included)-5:
                    
                    toe_xs = parse_stride_lists(smooth_toe_x, starts_included, ends_included, included_index, 5, True)
                    pairwise_dtw_res_x_5 = pairwise_dtw(toe_xs)
                    dtw_x_plane_5_mean = np.mean(pairwise_dtw_res_x_5)
                    dtw_x_plane_5_sd = np.std(pairwise_dtw_res_x_5)

                    toe_ys = parse_stride_lists(smooth_toe_y, starts_included, ends_included, included_index, 5, True)
                    pairwise_dtw_res_y_5 = pairwise_dtw(toe_ys)
                    dtw_y_plane_5_mean = np.mean(pairwise_dtw_res_y_5)
                    dtw_y_plane_5_sd = np.std(pairwise_dtw_res_y_5)

                    toe_xys = [np.append(toe_xs[j], toe_ys[j]).reshape(2,-1).T for j in range(5)]
                    pairwise_dtw_res_xy_5 = pairwise_dtw(toe_xys)
                    dtw_xy_plane_5_mean = np.mean(pairwise_dtw_res_xy_5)
                    dtw_xy_plane_5_sd = np.std(pairwise_dtw_res_xy_5)

                    if included_index <= len(starts_included)-10:
                        toe_xs = parse_stride_lists(smooth_toe_x, starts_included, ends_included, included_index, 10, True)
                        pairwise_dtw_res_x_10 = pairwise_dtw(toe_xs)
                        dtw_x_plane_10_mean = np.mean(pairwise_dtw_res_x_10)
                        dtw_x_plane_10_sd = np.std(pairwise_dtw_res_x_10)

                        toe_ys = parse_stride_lists(smooth_toe_y, starts_included, ends_included, included_index, 10, True)
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
                
    for bodypart in bodyparts:
        for coord in ['x', 'y']:
            pd_dataframe[f'{bodypart} {coord}'] = butterworth_filter(pd_dataframe[f'{bodypart} {coord}'], frame_rate, cutoff_f)
        
    return pd.DataFrame(data=np.array([
                            starts,
                            ends,
                            cycle_dur_secs,
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
                            mtp_joint_sds,
                            ankle_joint_extensions,
                            ankle_joint_flexions,
                            ankle_joint_amplitudes,
                            ankle_joint_sds,
                            knee_joint_extensions,
                            knee_joint_flexions,
                            knee_joint_amplitudes,
                            knee_joint_sds,
                            hip_joint_extensions,
                            hip_joint_flexions,
                            hip_joint_amplitudes,   
                            hip_joint_sds,
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
                                'stride_start (frame)',
                                'stride_end (frame)',
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
                                'mtp joint SD (deg)',
                                'ankle joint extension (deg)',
                                'ankle joint flexion (deg)',
                                'ankle joint amplitude (deg)',
                                'ankle joint SD (deg)',
                                'knee joint extension (deg)',
                                'knee joint flexion (deg)',
                                'knee joint amplitude (deg)',
                                'knee joint SD (deg)',
                                'hip joint extension (deg)',
                                'hip joint flexion (deg)',
                                'hip joint amplitude (deg)', 
                                'hip joint SD (deg)', 
                                'drag duration (s)', 
                                'drag percentage (%)',
                                'DTW distance x plane 5 strides mean',
                                'DTW distance x plane 5 strides SD',
                                'DTW distance y plane 5 strides mean',
                                'DTW distance y plane 5 strides SD',
                                'DTW distance xy plane 5 strides mean',
                                'DTW distance xy plane 5 strides SD',
                                'DTW distance x plane 10 strides mean',
                                'DTW distance x plane 10 strides SD',
                                'DTW distance y plane 10 strides mean',
                                'DTW distance y plane 10 strides SD',
                                'DTW distance xy plane 10 strides mean',
                                'DTW distance xy plane 10 strides SD',
                                ]), pd_dataframe, is_stance, bodyparts


def collect_filtered_coords(filt_corrected_df, bodyparts):

    x_coords = filt_corrected_df[[f'{bodyparts[0]} x', f'{bodyparts[1]} x', f'{bodyparts[2]} x', f'{bodyparts[3]} x', f'{bodyparts[4]} x', f'{bodyparts[5]} x']]
    y_coords = filt_corrected_df[[f'{bodyparts[0]} y', f'{bodyparts[1]} y', f'{bodyparts[2]} y', f'{bodyparts[3]} y', f'{bodyparts[4]} y', f'{bodyparts[5]} y']]

    x_coords.columns = bodyparts
    y_coords.columns = bodyparts

    x_coords = x_coords.T
    y_coords = y_coords.T

    return x_coords, y_coords


def continuous_stickplot(filt_corrected_df, bodyparts, is_stance, x_coords, y_coords, start, end, filename='truncated_stickplot'):

    x_min = min(x_coords.loc[:, start:end].min())
    x_max = max(x_coords.loc[:, start:end].max())
    x_range = x_max - x_min
    y_min = min(y_coords.loc[:, start:end].min())
    y_max = max(y_coords.loc[:, start:end].max())
    y_range = y_max - y_min
    plt.figure(figsize = (x_range // y_range * 5, 5))

    for t in filt_corrected_df.index[start:end]:
        if t%2 == 0:
            # plotting every two frames to reduce clutter
            toe_mtp = pd.DataFrame(pd.concat([x_coords[t].iloc[0:2], y_coords[t].iloc[0:2]], axis=1))
            mtp_ankle = pd.DataFrame(pd.concat([x_coords[t].iloc[1:3], y_coords[t].iloc[1:3]], axis=1))
            ankle_knee = pd.DataFrame(pd.concat([x_coords[t].iloc[2:4], y_coords[t].iloc[2:4]], axis=1))
            knee_hip = pd.DataFrame(pd.concat([x_coords[t].iloc[3:5], y_coords[t].iloc[3:5]], axis=1))
            hip_crest = pd.DataFrame(pd.concat([x_coords[t].iloc[4:6], y_coords[t].iloc[4:6]], axis=1))

            toe_mtp.columns = ['coord x', 'coord y']
            mtp_ankle.columns = ['coord x', 'coord y']
            ankle_knee.columns = ['coord x', 'coord y']
            knee_hip.columns = ['coord x', 'coord y']
            hip_crest.columns = ['coord x', 'coord y']
            
            if is_stance[t] == 1:
                plt.plot('coord x', 'coord y', data=toe_mtp, c='#999999')
                plt.plot('coord x', 'coord y', data=mtp_ankle, c='#999999')
                plt.plot('coord x', 'coord y', data=ankle_knee, c='#999999')
                plt.plot('coord x', 'coord y', data=knee_hip, c='#999999')
                plt.plot('coord x', 'coord y', data=hip_crest, c='#999999')
            else:
                plt.plot('coord x', 'coord y', data=toe_mtp, c='#FAAF40')
                plt.plot('coord x', 'coord y', data=mtp_ankle, c='#FAAF40')
                plt.plot('coord x', 'coord y', data=ankle_knee, c='#FAAF40')
                plt.plot('coord x', 'coord y', data=knee_hip, c='#FAAF40')
                plt.plot('coord x', 'coord y', data=hip_crest, c='#FAAF40')

    plt.xlabel('x coordinate (pixel)')
    plt.ylabel('y coordinate (pixel)')
    plt.legend([f'{bodyparts[0]}-{bodyparts[1]}', 
                f'{bodyparts[1]}-{bodyparts[2]}', 
                f'{bodyparts[2]}-{bodyparts[3]}', 
                f'{bodyparts[3]}-{bodyparts[4]}', 
                f'{bodyparts[4]}-{bodyparts[5]}'])

    plt.ylim(y_min, y_max)
    plt.gca().invert_yaxis()
    plt.savefig(f'./{filename}.svg', format='svg', dpi=1200)
