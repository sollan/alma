import pandas as pd
import scipy
from scipy.signal import butter, filtfilt
import numpy as np
import math
import statistics as stat
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

def read_file(file):
    
    pd_dataframe = pd.read_csv(file, header=[1,2])
    filename = file.split('/')[-1]
    return pd_dataframe, filename


def detect_bodypart_mapping(bodyparts_detected):
    '''
    Auto-detect bodypart naming convention and create mapping to standard names.
    Handles: 'toe'/'toeR'/'toeL', 'iliac crest'/'crestR', etc.
    Returns: dict mapping detected names to standard names, and list of standard bodyparts
    '''
    standard_bodyparts = ['toe', 'mtp', 'ankle', 'knee', 'hip', 'iliac crest']
    bodypart_aliases = {
        'toe': ['toe', 'toer', 'toel', 'toe_r', 'toe_l'],
        'mtp': ['mtp', 'mtpr', 'mtpl', 'mtp_r', 'mtp_l'],
        'ankle': ['ankle', 'ankler', 'anklel', 'ankle_r', 'ankle_l'],
        'knee': ['knee', 'kneer', 'kneel', 'knee_r', 'knee_l'],
        'hip': ['hip', 'hipr', 'hipl', 'hip_r', 'hip_l'],
        'iliac crest': ['iliac crest', 'crest', 'crestr', 'crestl', 'crest_r', 'crest_l', 
                        'iliac crestr', 'iliac crestl', 'iliacr', 'iliacl']
    }
    
    mapping = {}
    found_bodyparts = []
    
    for detected in bodyparts_detected:
        detected_lower = detected.lower()
        for standard, aliases in bodypart_aliases.items():
            if detected_lower in aliases:
                mapping[detected] = standard
                if standard not in found_bodyparts:
                    found_bodyparts.append(standard)
                break
    
    # If no mapping found, keep original names
    if not mapping:
        return {bp: bp for bp in bodyparts_detected}, bodyparts_detected
    
    return mapping, found_bodyparts


def fix_column_names(pd_dataframe, custom_mapping=None):
    
    header_1 = pd_dataframe.columns.get_level_values(0)
    header_2 = pd_dataframe.columns.get_level_values(1)
    col_names = []
    bodyparts_raw = []

    for i in range(len(header_1)):
        col_names.append(' '.join([header_1[i], header_2[i]]))

    for column in col_names:
        if column.endswith(' y'):
            bodyparts_raw.append(column.strip(' y'))
    
    if custom_mapping is not None:
        bodypart_mapping = custom_mapping
        bodyparts_standard = list(set(custom_mapping.values()))
    else:
        bodypart_mapping, bodyparts_standard = detect_bodypart_mapping(bodyparts_raw)
    
    col_names_standard = []
    for col in col_names:
        parts = col.rsplit(' ', 1)
        if len(parts) == 2:
            bodypart_name, coord = parts
            standard_name = bodypart_mapping.get(bodypart_name, bodypart_name)
            col_names_standard.append(f'{standard_name} {coord}')
        else:
            col_names_standard.append(col)
    
    pd_dataframe.columns = col_names_standard
    
    return pd_dataframe, bodyparts_standard, bodyparts_raw


def treadmill_correction(pd_dataframe, bodyparts, px_speed = 8.09, right_to_left = True):
    '''
    Correct for treadmill movement in x coordinates
    
    Parameters:
    -----------
    pd_dataframe : DataFrame with bodypart coordinates
    bodyparts : list or str of bodypart names
    px_speed : treadmill speed in pixels/frame
    right_to_left : bool
        If True: mouse walks right-to-left, treadmill moves left-to-right
                 correction = -(x - treadmill_shift) 
        If False: mouse walks left-to-right, treadmill moves right-to-left
                 correction = x + treadmill_shift
    
    Explanation:
    - Right-to-left (typical): Mouse at position 100px, after 10 frames treadmill moved 80px right
      Without correction: mouse appears at 180px (moved by treadmill)
      With correction: -(180 - 80) = -100px (removes treadmill effect)
    
    - Left-to-right: Opposite signs needed
    '''

    correction = np.arange(0, len(pd_dataframe), 1)
    correction = correction * px_speed

    if type(bodyparts) is list and len(bodyparts) > 1:
        for bodypart in bodyparts:
            if right_to_left:
                # Mouse walks right-to-left (decreasing x), treadmill pushes left-to-right (increasing x)
                pd_dataframe[f'{bodypart} x'] = -(pd_dataframe[f'{bodypart} x'] - correction)
            else:
                # Mouse walks left-to-right (increasing x), treadmill pushes right-to-left (decreasing x)
                pd_dataframe[f'{bodypart} x'] = pd_dataframe[f'{bodypart} x'] + correction
    
    elif type(bodyparts) is list and len(bodyparts) == 1:
        if right_to_left:
            pd_dataframe[f'{bodyparts[0]} x'] = -(pd_dataframe[f'{bodyparts[0]} x'] - correction)
        else:
            pd_dataframe[f'{bodyparts[0]} x'] = pd_dataframe[f'{bodyparts[0]} x'] + correction
    
    elif type(bodyparts) is str:
        if right_to_left:
            pd_dataframe[f'{bodyparts} x'] = -(pd_dataframe[f'{bodyparts} x'] - correction)
        else:
            pd_dataframe[f'{bodyparts} x'] = pd_dataframe[f'{bodyparts} x'] + correction

    return pd_dataframe


def estimate_pixels_per_cm_from_bodyparts(pd_dataframe, bodypart1, bodypart2, known_distance_cm=None):
    '''
    Camera-distance independent calibration using anatomical reference
    Estimates pixels_per_cm by measuring a body segment with known real-world length
    
    bodypart1, bodypart2: two bodyparts defining a reference segment (e.g., 'hip' to 'knee')
    known_distance_cm: known real-world distance between bodyparts in cm
                       If None, uses typical rodent measurements:
                       - hip to knee (femur): ~2.5 cm for mouse, ~4.5 cm for rat
                       - knee to ankle (tibia): ~2.0 cm for mouse, ~3.5 cm for rat
    '''
    # Normalize bodypart names (handle variations)
    bodypart1_variants = [bodypart1, bodypart1.lower(), bodypart1.replace('_', ' ')]
    bodypart2_variants = [bodypart2, bodypart2.lower(), bodypart2.replace('_', ' ')]
    
    # Try to find the bodypart columns
    def find_bodypart_column(bodypart_variants, coord):
        for variant in bodypart_variants:
            col_name = f'{variant} {coord}'
            if col_name in pd_dataframe.columns:
                return col_name
        # If not found, return first variant (will raise error later if truly missing)
        return f'{bodypart_variants[0]} {coord}'
    
    try:
        # Calculate distance in pixels across all frames
        x1 = pd_dataframe[find_bodypart_column(bodypart1_variants, 'x')]
        y1 = pd_dataframe[find_bodypart_column(bodypart1_variants, 'y')]
        x2 = pd_dataframe[find_bodypart_column(bodypart2_variants, 'x')]
        y2 = pd_dataframe[find_bodypart_column(bodypart2_variants, 'y')]
        
        # Use high-confidence frames only
        likelihood1 = pd_dataframe[find_bodypart_column(bodypart1_variants, 'likelihood')]
        likelihood2 = pd_dataframe[find_bodypart_column(bodypart2_variants, 'likelihood')]
        
        valid_frames = (likelihood1 > 0.9) & (likelihood2 > 0.9)
        n_frames_high = valid_frames.sum()
        
        if n_frames_high == 0:
            valid_frames = (likelihood1 > 0.7) & (likelihood2 > 0.7)
            n_frames_medium = valid_frames.sum()
            
            if n_frames_medium == 0:
                valid_frames = (likelihood1 > 0.5) & (likelihood2 > 0.5)
                n_frames_low = valid_frames.sum()
                
                if n_frames_low < 10:
                    print(f"  ✗ Insufficient tracking for {bodypart1}-{bodypart2}:")
                    print(f"    Frames with likelihood > 0.5: {n_frames_low}")
                    print(f"    Need at least 10 frames for reliable calibration")
                    return None
                else:
                    print(f"  ⚠ Using {n_frames_low} frames with lower confidence (>0.5)")
            else:
                print(f"  ⚠ Using {n_frames_medium} frames with medium confidence (>0.7)")
        else:
            print(f"  ✓ Using {n_frames_high} high-confidence frames (>0.9)")
        
        distances_px = np.sqrt((x1[valid_frames] - x2[valid_frames])**2 + 
                               (y1[valid_frames] - y2[valid_frames])**2)
        
        # Calculate statistics to assess quality
        median_distance_px = np.median(distances_px)
        std_distance_px = np.std(distances_px)
        cv = (std_distance_px / median_distance_px) * 100  # coefficient of variation
        
        # Check if measurement is reliable (low variability)
        if cv > 15:  # More than 15% variation
            print(f"  ⚠ High variability detected: CV = {cv:.1f}%")
            print(f"    Segment length varies significantly across frames")
            print(f"    This might indicate poor tracking or actual limb flexion")
            if cv > 30:  # Very high variation
                print(f"  ✗ Calibration unreliable (CV > 30%), rejecting this segment")
                return None
        
        # If known distance not provided, use default based on segment
        if known_distance_cm is None:
            # Attempt to auto-detect based on segment name
            segment_key = f"{bodypart1}_{bodypart2}".lower().replace(' ', '_')
            
            # Default values for adult mouse (adjust based on your species)
            defaults = {
                'hip_knee': 2.5,      # femur
                'knee_ankle': 2.0,    # tibia
                'ankle_mtp': 0.8,     # metatarsus
                'ankle_toe': 1.5,     # ankle to toe (often well-tracked)
                'mtp_toe': 0.7,       # mtp to toe
                'hip_ankle': 4.5,     # femur + tibia
                'iliac_crest_hip': 1.5,  # pelvis segment
                'crest_hip': 1.5      # alias for iliac crest
            }
            
            known_distance_cm = defaults.get(segment_key, 3.0)  # default 3cm if unknown
            print(f"  Using default distance for {bodypart1}-{bodypart2}: {known_distance_cm} cm")
        
        pixels_per_cm = median_distance_px / known_distance_cm
        
        # Report calibration quality
        print(f"  Segment: {median_distance_px:.1f} ± {std_distance_px:.1f} pixels (CV: {cv:.1f}%)")
        
        return pixels_per_cm
        
    except KeyError as e:
        print(f"Warning: Could not find bodypart columns for calibration: {e}")
        print(f"Available columns: {list(pd_dataframe.columns[:10])}...")
        return None


def estimate_speed(pd_dataframe, bodypart, cm_speed, frame_rate, right_to_left = True, 
                   auto_calibrate_spatial=False, reference_bodyparts=None, reference_distance_cm=None):
    '''
    Estimate speed parameters for treadmill analysis
    
    auto_calibrate_spatial: if True, uses intrinsic body scaling for camera-independent calibration
    reference_bodyparts: tuple of (bodypart1, bodypart2) for spatial calibration, e.g., ('hip', 'knee')
    reference_distance_cm: known distance between reference bodyparts in cm
    '''

    x_change = np.diff(pd_dataframe[f'{bodypart} x'][pd_dataframe[f'{bodypart} likelihood']>0.5])
    x_change_filt = x_change[(x_change < np.mean(x_change) + 1*np.std(x_change)) & (x_change > np.mean(x_change) - 1*np.std(x_change))]

    if right_to_left:
        px_speed, _ = scipy.stats.norm.fit(x_change_filt[x_change_filt>0])
    else:
        px_speed, _ = scipy.stats.norm.fit(x_change_filt[x_change_filt<0])
        px_speed = - px_speed

    if cm_speed is None: 
        # cm_speed must be provided by user for treadmill analysis
        cm_speed = None
    else:
        px_to_cm_speed_ratio = px_speed / cm_speed

    # Camera-independent spatial calibration
    if auto_calibrate_spatial and reference_bodyparts is not None:
        pixels_per_cm = estimate_pixels_per_cm_from_bodyparts(
            pd_dataframe, 
            reference_bodyparts[0], 
            reference_bodyparts[1], 
            reference_distance_cm
        )
        
        # If primary segment failed, try fallback segments
        if pixels_per_cm is None:
            print(f"Primary segment {reference_bodyparts[0]}-{reference_bodyparts[1]} failed.")
            print("Trying fallback segments...")
            
            # Fallback order: ankle-toe, mtp-toe, knee-ankle, hip-knee
            fallback_segments = [
                (('ankle', 'toe'), 1.5),
                (('mtp', 'toe'), 0.7),
                (('knee', 'ankle'), 2.0),
                (('hip', 'knee'), 2.5),
            ]
            
            for fallback_pair, fallback_length in fallback_segments:
                # Skip if it's the same as what we just tried
                if fallback_pair == reference_bodyparts:
                    continue
                
                print(f"  Trying {fallback_pair[0]}-{fallback_pair[1]}...")
                pixels_per_cm = estimate_pixels_per_cm_from_bodyparts(
                    pd_dataframe,
                    fallback_pair[0],
                    fallback_pair[1],
                    fallback_length
                )
                
                if pixels_per_cm is not None:
                    print(f"  ✓ Success with {fallback_pair[0]}-{fallback_pair[1]}")
                    break
        
        if pixels_per_cm is not None:
            print(f"Auto-calibrated: {pixels_per_cm:.2f} pixels/cm (camera-distance independent)")
        else:
            # Fall back to original method if all auto-calibration failed
            print("⚠ All segments failed, falling back to traditional method")
            pixels_per_cm = 1 / (cm_speed / px_speed / frame_rate)
    else:
        # Original method
        pixels_per_cm = 1 / (cm_speed / px_speed / frame_rate)

    return cm_speed, px_speed, pixels_per_cm, px_to_cm_speed_ratio





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


def _detect_dragging_movement_cycles(pd_dataframe, frame_rate, cutoff_f, right_to_left):
    """
    Alternative stride detection for dragging mice based on forward movement cycles.
    Instead of detecting stance/swing phases, detect forward movement cycles.
    """
    bodypart = 'toe'
    
    # Get forward movement (x-coordinate changes)
    x_coords = pd_dataframe[f'{bodypart} x'][pd_dataframe[f'{bodypart} likelihood'] > 0.5]
    x_change = np.diff(x_coords)
    
    # Smooth the movement signal
    x_change_smooth = butterworth_filter(np.concatenate([[0], x_change]), frame_rate, cutoff_f)
    
    # Find movement cycles using zero-crossings of velocity
    # Positive values = forward movement, negative = backward movement
    zero_crossings = []
    for i in range(1, len(x_change_smooth)):
        if (x_change_smooth[i-1] <= 0 and x_change_smooth[i] > 0) or \
           (x_change_smooth[i-1] >= 0 and x_change_smooth[i] < 0):
            zero_crossings.append(i)
    
    # Group zero crossings into movement cycles
    starts = []
    ends = []
    is_stance = [0] * len(pd_dataframe)
    
    if len(zero_crossings) >= 2:
        for i in range(0, len(zero_crossings)-1, 2):
            if i+1 < len(zero_crossings):
                start_idx = zero_crossings[i]
                end_idx = zero_crossings[i+1]
                
                # Only include cycles with meaningful forward movement
                cycle_movement = np.sum(x_change_smooth[start_idx:end_idx])
                if cycle_movement > 5:  # Minimum forward movement threshold
                    starts.append(start_idx)
                    ends.append(end_idx)
                    
                    # Mark all frames in this cycle as "stance" (dragging)
                    for j in range(start_idx, min(end_idx, len(is_stance))):
                        is_stance[j] = 1
    
    # Use median y-coordinate as ground level
    y_coords = pd_dataframe[f'{bodypart} y'][pd_dataframe[f'{bodypart} likelihood'] > 0.5]
    treadmill_y = np.median(y_coords) if len(y_coords) > 0 else 0
    
    # Detected movement cycles for dragging mouse
    return is_stance, starts, ends, treadmill_y


def _robust_stance_segments_treadmill(pd_dataframe, frame_rate, cutoff_f, right_to_left):
    toe_x_raw = pd_dataframe['toe x']
    toe_y_raw = pd_dataframe['toe y']
    toe_lik = pd_dataframe['toe likelihood']

    smooth_toe_x = butterworth_filter(toe_x_raw, frame_rate, cutoff_f)
    smooth_toe_y = butterworth_filter(toe_y_raw, frame_rate, cutoff_f)

    x_change = np.diff(smooth_toe_x)
    if right_to_left:
        dom = x_change >= 0
    else:
        dom = x_change <= 0

    # Toe speed (px/s), align to diff length
    speed = np.array(find_euclidean_speed(smooth_toe_x, smooth_toe_y, frame_rate))

    # Likelihood and y arrays aligned to speed/x_change length
    lik = np.array(toe_lik[1:])
    toe_y = np.array(smooth_toe_y[1:])

    # Toe y stance threshold via GMM (use raw likelihood mask for stability)
    y_valid = pd_dataframe['toe y'][pd_dataframe['toe likelihood'] > 0.5]
    y_filt = y_valid[(y_valid < np.mean(y_valid) + 1*np.std(y_valid)) & (y_valid > np.mean(y_valid) - 1*np.std(y_valid))]
    _, b = np.histogram(y_filt, bins=100, density=True)
    gm = GaussianMixture(n_components=2, random_state=0).fit(np.array(b).reshape(-1,1))
    stance_threshold = max(gm.means_).item()
    ind_stance_gaussian = np.argmax(gm.means_)
    treadmill_y = float(stance_threshold + np.sqrt(gm.covariances_[ind_stance_gaussian].item()))

    y_contact = toe_y > treadmill_y
    lik_good = lik > 0.6

    # Hysteresis thresholds from distribution of speed
    p35 = np.percentile(speed, 35)
    p65 = np.percentile(speed, 65)
    low_th = min(p35, p65)
    high_th = max(p35, p65)
    if high_th <= low_th:
        high_th = low_th * 1.2 + 1e-3

    n = len(speed)
    is_stance = [False] * n
    prev = dom[0] and (speed[0] <= high_th) and (not lik_good[0] or y_contact[0])
    is_stance[0] = prev
    for t in range(1, n):
        if not lik_good[t]:
            # keep previous state when likelihood is low
            is_stance[t] = prev
        elif speed[t] <= low_th and dom[t] and (y_contact[t] or speed[t] <= low_th*0.8):
            is_stance[t] = True
        elif speed[t] >= high_th:
            is_stance[t] = False
        else:
            is_stance[t] = prev
        prev = is_stance[t]

    # Remove micro-segments shorter than 3 frames
    cleaned = is_stance[:]
    run_start = 0
    for t in range(1, n+1):
        if t == n or is_stance[t] != is_stance[t-1]:
            run_len = t - run_start
            if run_len < 3:
                fill = is_stance[t] if t < n else is_stance[run_start-1] if run_start > 0 else False
                for k in range(run_start, t):
                    cleaned[k] = fill
            run_start = t

    is_stance = cleaned

    # Build starts/ends based on transitions into stance
    starts = []
    ends = []
    for i in range(1, len(is_stance)):
        if is_stance[i] and not is_stance[i-1]:
            starts.append(i)
    for i in range(1, len(starts)):
        ends.append(starts[i]-1)
    ends.append(len(is_stance))

    return is_stance, starts, ends, treadmill_y


def find_limb_length(x1, y1, x2, y2):
    '''
    calculate the Euclidean distance between two endpoints of a limb
    using 4 lists
    '''
    return [math.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2) for i in range(len(x1))]


def find_drag(toe_y, stance_dur, threshold, start, end, clearance_px=0, min_consecutive_frames=4):
    """
    Detect dragging during swing phase with improved robustness.
    
    Args:
        toe_y: toe y-coordinates (vertical position)
        stance_dur: stance duration in frames
        threshold: fallback threshold (treadmill y level)
        start: stride start frame
        end: stride end frame
        clearance_px: minimum clearance above ground in pixels to not count as drag
        min_consecutive_frames: minimum consecutive frames of ground contact to count as dragging
    
    Returns:
        drag_frames: number of frames with dragging
        drag_fraction: fraction of swing phase with dragging
        drag_mask: boolean array indicating drag frames (for visualization)
    """
    swing_start = start + stance_dur
    swing_end = end
    swing_length = swing_end - swing_start
    
    if swing_length <= 0:
        return 0, 0, np.array([])
    
    # Per-stride ground detection from stance phase
    if stance_dur > 0:
        ground_level = np.percentile(toe_y[start:swing_start], 90)
    else:
        ground_level = threshold
    
    # Detect dragging: toe is near/below ground during swing
    is_drag_raw = toe_y[swing_start:swing_end] >= (ground_level - clearance_px)
    
    # Require at least min_consecutive_frames consecutive frames to reduce noise
    is_drag = is_drag_raw.copy()
    if len(is_drag) >= min_consecutive_frames:
        # Check for runs of at least min_consecutive_frames consecutive dragging frames
        for i in range(len(is_drag) - min_consecutive_frames + 1):
            if all(is_drag_raw[i:i+min_consecutive_frames]):
                # Mark this window as dragging
                is_drag[i:i+min_consecutive_frames] = True
            else:
                # If not all frames in window are dragging, mark as not dragging
                is_drag[i] = False
        # Handle remaining frames
        for i in range(len(is_drag) - min_consecutive_frames + 1, len(is_drag)):
            is_drag[i] = False
    
    drag_frames = np.sum(is_drag)
    drag_fraction = drag_frames / swing_length if swing_length > 0 else 0
    
    return drag_frames, drag_fraction, is_drag


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
    results = [np.array(x[starts[i]:ends[i]+1]) for i in range(first_stride, first_stride+n)]
    if adjust:
        for n_sublist in range(len(results)):
            centers = results[n_sublist][0]
            results[n_sublist] = np.array(results[n_sublist] - centers).flatten()
    return results



def pairwise_dtw(iterable):
    '''
    calculates stride-to-stride variability using mean absolute difference
    iterable: a list / array of time series (endpoint paths of a number of strides)
    returns: 
    variability_result: a list of pairwise normalized differences
    e.g., iterable = [stride_a, stride_b, stride_c]
    -> variability_result = [diff_ab, diff_ac, diff_bc]
    
    Note: Replaced DTW with simpler normalized difference metric for better compatibility
    '''
    variability_result = []
    for i in range(len(iterable)-1):
        for j in range(i+1, len(iterable)):
            stride_i = np.array(iterable[i])
            stride_j = np.array(iterable[j])
            
            # Check if 1D or 2D trajectories
            is_2d = len(stride_i.shape) > 1 and stride_i.shape[1] > 1
            
            if is_2d:
                # For 2D trajectories (xy plane)
                len_i = stride_i.shape[0]
                len_j = stride_j.shape[0]
                
                if len_i == len_j:
                    # Euclidean distance between corresponding points
                    diff = np.mean(np.sqrt(np.sum((stride_i - stride_j)**2, axis=1)))
                else:
                    # Interpolate to same length for each dimension
                    common_len = max(len_i, len_j)
                    stride_i_interp = np.zeros((common_len, stride_i.shape[1]))
                    stride_j_interp = np.zeros((common_len, stride_j.shape[1]))
                    
                    for dim in range(stride_i.shape[1]):
                        stride_i_interp[:, dim] = np.interp(np.linspace(0, 1, common_len), 
                                                            np.linspace(0, 1, len_i), 
                                                            stride_i[:, dim])
                        stride_j_interp[:, dim] = np.interp(np.linspace(0, 1, common_len), 
                                                            np.linspace(0, 1, len_j), 
                                                            stride_j[:, dim])
                    
                    diff = np.mean(np.sqrt(np.sum((stride_i_interp - stride_j_interp)**2, axis=1)))
            else:
                # For 1D trajectories (x or y plane only)
                len_i = len(stride_i)
                len_j = len(stride_j)
                
                if len_i == len_j:
                    diff = np.mean(np.abs(stride_i - stride_j))
                else:
                    # Interpolate to same length
                    common_len = max(len_i, len_j)
                    stride_i_interp = np.interp(np.linspace(0, 1, common_len), 
                                               np.linspace(0, 1, len_i), 
                                               stride_i.flatten())
                    stride_j_interp = np.interp(np.linspace(0, 1, common_len), 
                                               np.linspace(0, 1, len_j), 
                                               stride_j.flatten())
                    diff = np.mean(np.abs(stride_i_interp - stride_j_interp))
            
            variability_result.append(diff)
    return variability_result


def make_parameters_output(pathname, parameters):

    parameters.to_csv(pathname)


def make_averaged_output(pathname, truncated=False):

    files = []
    for file in os.listdir(pathname):
        if truncated:
            if file.endswith('.csv') and file.startswith('continuous_strides_parameters_'):
                files.append(file)
        else:
            if file.endswith('.csv') and file.startswith('parameters_'):
                files.append(file)

    dfs = []
    for file in files:

        if truncated:
            input_name = file.split('continuous_strides_parameters_')[1].split('.')[0]
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


def return_continuous(pd_dataframe_parameters, n_continuous=10, plot=False, pd_dataframe_coords=None, bodyparts=None, is_stance=[], filename='', drag_masks=None, starts=None, drag_min_consecutive_frames=4):
    # Check if parameters DataFrame is empty or contains only NaN values
    if pd_dataframe_parameters.empty or pd_dataframe_parameters.isnull().all().all():
        # Parameters DataFrame is empty or contains only NaN values - skipping continuous analysis
        return None
    
    # Debug: cycle duration data
    # print(pd_dataframe_parameters['cycle duration (s)'])
    valid_list = convert_to_binary(np.array(pd_dataframe_parameters['cycle duration (s)'], dtype='float32'))
    start, end = findLongestSequence(valid_list, 0)
    start_stride = int(np.mean([start, end]))-n_continuous//2
    end_stride = start_stride+n_continuous
    if end_stride >= end or end-start <= n_continuous: # not enough continuous
        start_stride = start
        end_stride = end
    if end_stride==start_stride:
        return
    else:
        try:
            if plot:
                x_coords, y_coords = collect_filtered_coords(pd_dataframe_coords, bodyparts)
                start_frame_val = pd_dataframe_parameters.iloc[start_stride]['stride_start (frame)']
                end_frame_val = pd_dataframe_parameters.iloc[end_stride]['stride_end (frame)']
                
                # Check for NaN values (both actual NaN and string 'nan')
                if pd.isna(start_frame_val) or pd.isna(end_frame_val) or \
                   str(start_frame_val).lower() == 'nan' or str(end_frame_val).lower() == 'nan':
                    # Skipping plot generation due to NaN values
                    return pd_dataframe_parameters.iloc[start_stride:end_stride, :]
                
                start_frame = int(float(start_frame_val))
                end_frame = int(float(end_frame_val))
                
                # Build full is_drag array from drag_masks and starts
                is_drag = None
                if drag_masks is not None and starts is not None and len(drag_masks) > 0:
                    is_drag = np.zeros(len(is_stance), dtype=bool)
                    for i, drag_mask in enumerate(drag_masks):
                        if len(drag_mask) > 0 and i < len(starts):
                            stride_start = starts[i]
                            # Find the end of this stride to determine stance duration
                            stride_end = len(is_stance)  # Default to end of data
                            if i + 1 < len(starts):
                                stride_end = starts[i + 1]
                            
                            # Find stance duration for this stride
                            stance_dur = 0
                            for frame in range(stride_start, min(stride_end, len(is_stance))):
                                if is_stance[frame] == 1:
                                    stance_dur += 1
                                else:
                                    break
                            
                            # Apply drag mask only during swing phase (after stance)
                            swing_start = stride_start + stance_dur
                            for j, is_dragging in enumerate(drag_mask):
                                frame_idx = swing_start + j
                                if frame_idx < len(is_drag) and frame_idx < stride_end:
                                    # Only mark as dragging if it's during swing phase (not stance)
                                    if is_stance[frame_idx] == 0:  # Swing phase
                                        is_drag[frame_idx] = is_dragging
                
                continuous_stickplot(pd_dataframe_coords, bodyparts, is_stance, x_coords, y_coords, start_frame, end_frame, filename, is_drag, drag_min_consecutive_frames)
        except (IndexError, ValueError) as e:
            # Error in return_continuous - returning partial results
            pass
        return pd_dataframe_parameters.iloc[start_stride:end_stride, :]


def compute_limb_joint_angles(smooth_toe_x, smooth_toe_y, \
                            smooth_mtp_x, smooth_mtp_y, \
                            smooth_ankle_x, smooth_ankle_y, \
                            smooth_knee_x, smooth_knee_y, \
                            smooth_hip_x, smooth_hip_y, \
                            smooth_crest_x, smooth_crest_y):

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

    return angles_toe_mtp_ankle, angles_mtp_ankle_knee, angles_ankle_knee_hip, angles_knee_hip_crest, elevation_angles, elevation_angle_change




def extract_parameters(frame_rate, pd_dataframe, cutoff_f, bodypart, cm_speed=None, right_to_left=True,
                      step_height_min_cm=0.0, step_height_max_cm=1.5, stride_length_min_cm=0.0, stride_length_max_cm=8.0, drag_clearance_cm=0.3, drag_min_consecutive_frames=4):

    '''
    pd_dataframe: contains raw coordinates (not adjusted for treadmill movement, unfiltered)
    bodypart: which bodypart to use for stride estimation

    '''
    bodyparts = ['toe', 'mtp', 'ankle', 'knee', 'hip', 'iliac crest'] # modify this
    starts = []
    ends = []

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
    drag_masks = []  # For visualization in stickplot
    is_stance = []  # Initialize early for excessive dragging case

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

    # Auto-detect walking direction if not specified
    if right_to_left == "auto" or right_to_left is None:
        # Check which direction has more movement (stance phase shows treadmill movement)
        positive_changes = x_change_filt[x_change_filt > 0]
        negative_changes = x_change_filt[x_change_filt < 0]
        
        if len(positive_changes) > len(negative_changes):
            right_to_left = True
            # Auto-detected walking direction: right-to-left
        else:
            right_to_left = False
            # Auto-detected walking direction: left-to-right
    
    # Detect treadmill speed based on walking direction
    # Right-to-left: mouse moves left (negative), treadmill moves right (positive during stance)
    # Left-to-right: mouse moves right (positive), treadmill moves left (negative during stance)
    if right_to_left:
        px_speed, _ = scipy.stats.norm.fit(x_change_filt[x_change_filt>0])
    else:
        px_speed, _ = scipy.stats.norm.fit(x_change_filt[x_change_filt<0])
        px_speed = -px_speed  # Make positive for consistency

    if cm_speed is None: 
        # cm_speed must be provided by user for treadmill analysis
        raise ValueError("cm_speed must be provided for treadmill analysis")

    pixels_per_cm = 1 / (cm_speed / px_speed / frame_rate)
    pd_dataframe_corrected = treadmill_correction(pd_dataframe, bodyparts, px_speed, right_to_left)

    smooth_toe_x = butterworth_filter(pd_dataframe_corrected['toe x'], frame_rate, cutoff_f)
    smooth_mtp_x = butterworth_filter(pd_dataframe_corrected['mtp x'], frame_rate, cutoff_f)
    smooth_ankle_x = butterworth_filter(pd_dataframe_corrected['ankle x'], frame_rate, cutoff_f)
    smooth_knee_x = butterworth_filter(pd_dataframe_corrected['knee x'], frame_rate, cutoff_f)
    smooth_hip_x = butterworth_filter(pd_dataframe_corrected['hip x'], frame_rate, cutoff_f)
    smooth_crest_x = butterworth_filter(pd_dataframe_corrected['iliac crest x'], frame_rate, cutoff_f)

    smooth_toe_y = butterworth_filter(pd_dataframe['toe y'], frame_rate, cutoff_f)
    smooth_mtp_y = butterworth_filter(pd_dataframe['mtp y'], frame_rate, cutoff_f)
    smooth_ankle_y = butterworth_filter(pd_dataframe['ankle y'], frame_rate, cutoff_f)
    smooth_knee_y = butterworth_filter(pd_dataframe['knee y'], frame_rate, cutoff_f)
    smooth_hip_y = butterworth_filter(pd_dataframe['hip y'], frame_rate, cutoff_f)
    smooth_crest_y = butterworth_filter(pd_dataframe['iliac crest y'], frame_rate, cutoff_f)
    
    angles_toe_mtp_ankle, \
        angles_mtp_ankle_knee, \
        angles_ankle_knee_hip, \
        angles_knee_hip_crest, \
        elevation_angles, \
        elevation_angle_change = compute_limb_joint_angles(
            smooth_toe_x,
            smooth_toe_y, 
            smooth_mtp_x, 
            smooth_mtp_y,
            smooth_ankle_x,
            smooth_ankle_y,
            smooth_knee_x,
            smooth_knee_y,
            smooth_hip_x,
            smooth_hip_y,
            smooth_crest_x,
            smooth_crest_y)

    limb_lens = find_limb_length(smooth_toe_x, smooth_toe_y, smooth_hip_x, smooth_hip_y)
    velocities = find_euclidean_speed(smooth_toe_x, smooth_toe_y, frame_rate)

    # Check if there's any meaningful forward movement at all
    if np.abs(stat.median(x_change)) < 0.1 and len(np.where(np.abs(x_change) < 0.5)[0])/len(x_change) > 0.95:
        
        # No meaningful forward movement detected - mouse may be stationary

        starts.append(np.nan)
        ends.append(np.nan)
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
        drag_masks.append(np.array([]))
        is_stance = [0] * len(pd_dataframe)  # All frames considered as invalid/unknown
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

        # Return early for excessive dragging case
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
                                    'Variability x plane 5 strides mean',
                                    'Variability x plane 5 strides SD',
                                    'Variability y plane 5 strides mean',
                                    'Variability y plane 5 strides SD',
                                    'Variability xy plane 5 strides mean',
                                    'Variability xy plane 5 strides SD',
                                    'Variability x plane 10 strides mean',
                                    'Variability x plane 10 strides SD',
                                    'Variability y plane 10 strides mean',
                                    'Variability y plane 10 strides SD',
                                    'Variability xy plane 10 strides mean',
                                    'Variability xy plane 10 strides SD',
                                    ]), pd_dataframe, is_stance, bodyparts, drag_masks, starts

    else:
        
        # Try normal stance detection first
        is_stance, starts, ends, treadmill_y = _robust_stance_segments_treadmill(pd_dataframe, frame_rate, cutoff_f, right_to_left)
        
        # If no strides detected, try alternative detection for dragging mice
        if len(starts) == 0:
            # No strides detected with normal method - trying alternative detection for dragging mice
            # Use forward movement cycles instead of stance/swing phases
            is_stance, starts, ends, treadmill_y = _detect_dragging_movement_cycles(pd_dataframe, frame_rate, cutoff_f, right_to_left)

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
        
            # For dragging mice, be more lenient with step height (they can't lift their feet)
            # But still require some forward movement
            include_stride = (
                (stride_len_cm >= stride_length_min_cm) and (stride_len_cm <= stride_length_max_cm) and
                (step_height >= 0) and (step_height <= step_height_max_cm)  # Allow 0 step height for dragging
            )
            if include_stride:
                
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

                drag_clearance_px = drag_clearance_cm * pixels_per_cm
                drag, drag_percent, drag_mask = find_drag(smooth_toe_y, stance_dur_frame, treadmill_y, starts[i], ends[i], drag_clearance_px, drag_min_consecutive_frames)
                drag_ts.append(drag/frame_rate)
                drag_masks.append(drag_mask)
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
                drag_masks.append(np.array([]))
            
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
                included_index = int(np.array(np.where(starts[i] == starts_included)[0]).item())                
                
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
                                    'Variability x plane 5 strides mean',
                                    'Variability x plane 5 strides SD',
                                    'Variability y plane 5 strides mean',
                                    'Variability y plane 5 strides SD',
                                    'Variability xy plane 5 strides mean',
                                    'Variability xy plane 5 strides SD',
                                    'Variability x plane 10 strides mean',
                                    'Variability x plane 10 strides SD',
                                    'Variability y plane 10 strides mean',
                                    'Variability y plane 10 strides SD',
                                    'Variability xy plane 10 strides mean',
                                    'Variability xy plane 10 strides SD',
                                    ]), pd_dataframe, is_stance, bodyparts, drag_masks, starts


def extract_spontaneous_parameters(frame_rate, pd_dataframe, cutoff_f, pixels_per_cm=49.143, no_outlier_filter=False, dragging_filter=False,
                                   step_height_min_cm=0.0, step_height_max_cm=1.5, stride_length_min_cm=0.0, stride_length_max_cm=8.0, drag_clearance_cm=0.3, drag_min_consecutive_frames=4):
    '''
    pd_dataframe: contains raw coordinates (not adjusted for treadmill movement, unfiltered)
    bodypart: which bodypart to use for stride estimation
    '''

    bodyparts = ['toe', 'mtp', 'ankle', 'knee', 'hip', 'iliac crest'] # should be modified to detect names / change this manually on error!
    starts_all = []
    ends_all = []
    limbs = []

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

    is_stances = []

    for direction in ['L', 'R', '']:
        if f'toe{direction} x' in pd_dataframe.columns:
            x_change = np.diff(pd_dataframe[f'toe{direction} x'][pd_dataframe[f'toe{direction} likelihood']>0.5])
            # print('found', direction)
            smooth_toe_x = butterworth_filter(pd_dataframe[f'toe{direction} x'], frame_rate, cutoff_f)
            smooth_mtp_x = butterworth_filter(pd_dataframe[f'mtp{direction} x'], frame_rate, cutoff_f)
            smooth_ankle_x = butterworth_filter(pd_dataframe[f'ankle{direction} x'], frame_rate, cutoff_f)
            smooth_knee_x = butterworth_filter(pd_dataframe[f'knee{direction} x'], frame_rate, cutoff_f)
            smooth_hip_x = butterworth_filter(pd_dataframe[f'hip{direction} x'], frame_rate, cutoff_f)
            smooth_crest_x = butterworth_filter(pd_dataframe[f'iliac crest{direction} x'], frame_rate, cutoff_f)

            smooth_toe_y = butterworth_filter(pd_dataframe[f'toe{direction} y'], frame_rate, cutoff_f)
            smooth_mtp_y = butterworth_filter(pd_dataframe[f'mtp{direction} y'], frame_rate, cutoff_f)
            smooth_ankle_y = butterworth_filter(pd_dataframe[f'ankle{direction} y'], frame_rate, cutoff_f)
            smooth_knee_y = butterworth_filter(pd_dataframe[f'knee{direction} y'], frame_rate, cutoff_f)
            smooth_hip_y = butterworth_filter(pd_dataframe[f'hip{direction} y'], frame_rate, cutoff_f)
            smooth_crest_y = butterworth_filter(pd_dataframe[f'iliac crest{direction} y'], frame_rate, cutoff_f)

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

            # Check if there's any meaningful forward movement at all
            if dragging_filter and \
                len(np.where(np.abs(elevation_angle_change) < 1)[0])/len(elevation_angle_change) > 0.95:

                # No meaningful elevation angle change - mouse may be stationary or severely impaired
                limbs.append(direction)
                starts_all.append(np.nan)
                ends_all.append(np.nan)
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

                is_stances.append(np.nan)

            else:
                starts = []
                ends = []

                # Try normal stride detection first
                bodypart_x_change = np.diff(pd_dataframe[f'toe{direction} x'])
                is_stance = [abs(i) <= 0.001 for i in bodypart_x_change]

                # if direction == 'L': # toe L: direction right to left
                #     is_stance = [i <= 0 for i in bodypart_x_change]
                # elif direction == 'R': # toeR: direction left to right
                #     is_stance = [i >= 0 for i in bodypart_x_change]
                # else: # default direction left to right: right side visible
                #     is_stance = [i >= 0 for i in bodypart_x_change]
                for i in range(1, len(is_stance)):
                    if is_stance[i] != is_stance[i-1]:
                        if is_stance[i] and pd_dataframe[f'toe{direction} likelihood'][i] > 0.6:
                            if len(starts) == 0:
                                starts.append(i)
                            elif abs(pd_dataframe[f'toe{direction} x'][starts[-1]] - pd_dataframe[f'toe{direction} x'][i])>20:
                                starts.append(i)
                for i in range(1, len(starts)):
                    ends.append(starts[i]-1)

                ends.append(len(is_stance))

                starts_all.extend(starts)
                ends_all.extend(ends)

                is_stances.append(is_stance)

                y = pd_dataframe[f'toe{direction} y'][pd_dataframe[f'toe{direction} likelihood'] > 0.5]
                y_filt = y[(y < np.mean(y) + 1*np.std(y)) & (y > np.mean(y) - 1*np.std(y))]
                _, b = np.histogram(y_filt, bins=100, density=True)
                # fit 2 Gaussians to the pdf of toe y coord
                gm = GaussianMixture(n_components=2, random_state=0).fit(np.array(b).reshape(-1,1))
                # the Gaussian with larger mean corresponds to y coord during stance phase
                # use this mean as threshold to detect dragging during swing phase
                stance_threshold = max(gm.means_).item()
                # use this plus SD (i.e., lower in space) as treadmill y coord, to calculate step height
                ind_stance_gaussian = np.argmax(gm.means_)
                treadmill_y = float(stance_threshold + np.sqrt(gm.covariances_[ind_stance_gaussian].item()))

                starts_included = []
                ends_included = []

                for i in range(len(starts)):
                    limbs.append(direction)

                    stride_len = abs(find_stride_len(smooth_toe_x, starts[i], ends[i])) # in pixel
                    stride_len_cm = stride_len / pixels_per_cm
                    limb_len_max = max(limb_lens[starts[i] : ends[i]]) / pixels_per_cm
                    # limb_len_max = max(limb_lens[starts[i] : ends[i]])
                    cycle_dur_frame = ends[i] - starts[i]
                    cycle_v = stride_len_cm / cycle_dur_frame * frame_rate # (px / frames) * (frames/s) = px/s

                    stance_dur_frame, swing_perc, stance_perc = find_swing_stance(elevation_angle_change, starts[i], ends[i])
                    swing_dur_sec = (cycle_dur_frame - stance_dur_frame) / frame_rate

                    if swing_dur_sec and stance_dur_frame > 0:
                    # min: image y axis starts from top
                    # step_height = -(min(smooth_toe_y[starts[i]+stance_dur_frame:ends[i]]) - \
                    # max(smooth_toe_y[starts[i]:starts[i]+stance_dur_frame])) / pixels_per_cm
                        step_height = -(min(smooth_toe_y[starts[i]:ends[i]]) - \
                                        max(smooth_toe_y[starts[i]:ends[i]])) / pixels_per_cm
                    elif swing_dur_sec and stance_dur_frame == 0:
                        step_height = -(min(smooth_toe_y[starts[i]+stance_dur_frame:ends[i]]) - \
                                        smooth_toe_y[starts[i]]) / pixels_per_cm
                    else:
                        # no swing phase!
                        step_height = 0

                    if no_outlier_filter or (
                        (stride_len_cm >= stride_length_min_cm) and (stride_len_cm <= stride_length_max_cm) and
                        (step_height >= step_height_min_cm) and (step_height <= step_height_max_cm)
                    ):
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

                        cycle_vs.append(cycle_v) # (px/s) / (px/cm) = cm/s
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

                        drag_clearance_px = drag_clearance_cm * pixels_per_cm
                        drag, drag_percent, drag_mask = find_drag(smooth_toe_y, stance_dur_frame, treadmill_y, starts[i], ends[i], drag_clearance_px, drag_min_consecutive_frames)
                        drag_ts.append(drag/frame_rate)
                        if direction == 'L':
                            drag_masks_left.append(drag_mask)
                        elif direction == 'R':
                            drag_masks_right.append(drag_mask)
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
                    included_index = int(np.where(starts[i] == starts_included)[0].item())                
                    
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
        # except KeyError:
        else:
            pass

    # Debug: cycle detection summary
    # print(f"starts_all: {len(starts_all)}, ends_all: {len(ends_all)}, cycle_dur_secs: {len(cycle_dur_secs)}")
    
    # Check if any gait cycles were detected
    if len(starts_all) == 0 and len(ends_all) == 0:
        # No gait cycles detected - returning empty DataFrame
        # Return empty DataFrame with proper structure
        empty_df = pd.DataFrame(columns=[
            'limb (hind left / right)', 'stride_start (frame)', 'stride_end (frame)',
            'cycle duration (s)', 'cycle duration (no. frames)', 'cycle velocity (cm/s)',
            'stride length (cm)', 'stance duration (s)', 'swing duration (s)',
            'swing percentage (%)', 'stance percentage (%)', 'mean toe-to-crest distance (cm)',
            'max toe-to-crest distance (cm)', 'min toe-to-crest distance (cm)',
            'std toe-to-crest distance (cm)', 'step height (cm)', 'max velocity during swing (cm/s)',
            'mtp joint extension (degrees)', 'mtp joint flexion (degrees)', 'mtp joint amplitude (degrees)',
            'mtp joint std (degrees)', 'ankle joint extension (degrees)', 'ankle joint flexion (degrees)',
            'ankle joint amplitude (degrees)', 'ankle joint std (degrees)', 'knee joint extension (degrees)',
            'knee joint flexion (degrees)', 'knee joint amplitude (degrees)', 'knee joint std (degrees)',
            'hip joint extension (degrees)', 'hip joint flexion (degrees)', 'hip joint amplitude (degrees)',
            'hip joint std (degrees)', 'drag time (s)', 'drag percentage (%)',
            'dtw x plane 5 means', 'dtw x plane 5 sds', 'dtw y plane 5 means', 'dtw y plane 5 sds',
            'dtw xy plane 5 means', 'dtw xy plane 5 sds', 'dtw x plane 10 means', 'dtw x plane 10 sds',
            'dtw y plane 10 means', 'dtw y plane 10 sds', 'dtw xy plane 10 means', 'dtw xy plane 10 sds'
        ])
        return empty_df, pd.DataFrame(), [], bodyparts
    
    # Ensure all lists have the same length
    all_lists = [starts_all, ends_all, cycle_dur_secs, cycle_dur_frames, cycle_vs, stride_lens, 
                 stance_dur_secs, swing_dur_secs, swing_percentages, stance_percentages,
                 limb_len_means, limb_len_maxs, limb_len_mins, limb_len_sds, step_heights,
                 max_v_during_swings, mtp_joint_extensions, mtp_joint_flexions, mtp_joint_amplitudes,
                 mtp_joint_sds, ankle_joint_extensions, ankle_joint_flexions, ankle_joint_amplitudes,
                 ankle_joint_sds, knee_joint_extensions, knee_joint_flexions, knee_joint_amplitudes,
                 knee_joint_sds, hip_joint_extensions, hip_joint_flexions, hip_joint_amplitudes,
                 hip_joint_sds, drag_ts, drag_percentages, dtw_x_plane_5_means, dtw_x_plane_5_sds,
                 dtw_y_plane_5_means, dtw_y_plane_5_sds, dtw_xy_plane_5_means, dtw_xy_plane_5_sds,
                 dtw_x_plane_10_means, dtw_x_plane_10_sds, dtw_y_plane_10_means, dtw_y_plane_10_sds,
                 dtw_xy_plane_10_means, dtw_xy_plane_10_sds]
    
    max_len = max(len(lst) for lst in all_lists)
    
    # Pad shorter lists with appropriate values
    def pad_list(lst, target_len, pad_value=np.nan):
        return lst + [pad_value] * (target_len - len(lst))
    
    # Pad limbs list with empty strings
    limbs = pad_list(limbs, max_len, "")
    
    # Pad all lists
    starts_all = pad_list(starts_all, max_len)
    ends_all = pad_list(ends_all, max_len)
    cycle_dur_secs = pad_list(cycle_dur_secs, max_len)
    cycle_dur_frames = pad_list(cycle_dur_frames, max_len)
    cycle_vs = pad_list(cycle_vs, max_len)
    stride_lens = pad_list(stride_lens, max_len)
    stance_dur_secs = pad_list(stance_dur_secs, max_len)
    swing_dur_secs = pad_list(swing_dur_secs, max_len)
    swing_percentages = pad_list(swing_percentages, max_len)
    stance_percentages = pad_list(stance_percentages, max_len)
    limb_len_means = pad_list(limb_len_means, max_len)
    limb_len_maxs = pad_list(limb_len_maxs, max_len)
    limb_len_mins = pad_list(limb_len_mins, max_len)
    limb_len_sds = pad_list(limb_len_sds, max_len)
    step_heights = pad_list(step_heights, max_len)
    max_v_during_swings = pad_list(max_v_during_swings, max_len)
    mtp_joint_extensions = pad_list(mtp_joint_extensions, max_len)
    mtp_joint_flexions = pad_list(mtp_joint_flexions, max_len)
    mtp_joint_amplitudes = pad_list(mtp_joint_amplitudes, max_len)
    mtp_joint_sds = pad_list(mtp_joint_sds, max_len)
    ankle_joint_extensions = pad_list(ankle_joint_extensions, max_len)
    ankle_joint_flexions = pad_list(ankle_joint_flexions, max_len)
    ankle_joint_amplitudes = pad_list(ankle_joint_amplitudes, max_len)
    ankle_joint_sds = pad_list(ankle_joint_sds, max_len)
    knee_joint_extensions = pad_list(knee_joint_extensions, max_len)
    knee_joint_flexions = pad_list(knee_joint_flexions, max_len)
    knee_joint_amplitudes = pad_list(knee_joint_amplitudes, max_len)
    knee_joint_sds = pad_list(knee_joint_sds, max_len)
    hip_joint_extensions = pad_list(hip_joint_extensions, max_len)
    hip_joint_flexions = pad_list(hip_joint_flexions, max_len)
    hip_joint_amplitudes = pad_list(hip_joint_amplitudes, max_len)
    hip_joint_sds = pad_list(hip_joint_sds, max_len)
    drag_ts = pad_list(drag_ts, max_len)
    drag_percentages = pad_list(drag_percentages, max_len)
    dtw_x_plane_5_means = pad_list(dtw_x_plane_5_means, max_len)
    dtw_x_plane_5_sds = pad_list(dtw_x_plane_5_sds, max_len)
    dtw_y_plane_5_means = pad_list(dtw_y_plane_5_means, max_len)
    dtw_y_plane_5_sds = pad_list(dtw_y_plane_5_sds, max_len)
    dtw_xy_plane_5_means = pad_list(dtw_xy_plane_5_means, max_len)
    dtw_xy_plane_5_sds = pad_list(dtw_xy_plane_5_sds, max_len)
    dtw_x_plane_10_means = pad_list(dtw_x_plane_10_means, max_len)
    dtw_x_plane_10_sds = pad_list(dtw_x_plane_10_sds, max_len)
    dtw_y_plane_10_means = pad_list(dtw_y_plane_10_means, max_len)
    dtw_y_plane_10_sds = pad_list(dtw_y_plane_10_sds, max_len)
    dtw_xy_plane_10_means = pad_list(dtw_xy_plane_10_means, max_len)
    dtw_xy_plane_10_sds = pad_list(dtw_xy_plane_10_sds, max_len)

    return pd.DataFrame(data=np.array([
                            limbs,
                            starts_all,
                            ends_all,
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
                                'limb (hind left / right)',
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
                                'Variability x plane 5 strides mean',
                                'Variability x plane 5 strides SD',
                                'Variability y plane 5 strides mean',
                                'Variability y plane 5 strides SD',
                                'Variability xy plane 5 strides mean',
                                'Variability xy plane 5 strides SD',
                                'Variability x plane 10 strides mean',
                                'Variability x plane 10 strides SD',
                                'Variability y plane 10 strides mean',
                                'Variability y plane 10 strides SD',
                                'Variability xy plane 10 strides mean',
                                'Variability xy plane 10 strides SD',
                                ]), pd_dataframe, np.array(is_stances).reshape(-1), bodyparts, None, None

def collect_filtered_coords(filt_corrected_df, bodyparts):

    x_coords = filt_corrected_df[[f'{bodyparts[0]} x', f'{bodyparts[1]} x', f'{bodyparts[2]} x', f'{bodyparts[3]} x', f'{bodyparts[4]} x', f'{bodyparts[5]} x']]
    y_coords = filt_corrected_df[[f'{bodyparts[0]} y', f'{bodyparts[1]} y', f'{bodyparts[2]} y', f'{bodyparts[3]} y', f'{bodyparts[4]} y', f'{bodyparts[5]} y']]

    x_coords.columns = bodyparts
    y_coords.columns = bodyparts

    x_coords = x_coords.T
    y_coords = y_coords.T

    return x_coords, y_coords


def continuous_stickplot(filt_corrected_df, bodyparts, is_stance, x_coords, y_coords, start, end, filename='truncated_stickplot', is_drag=None, drag_min_consecutive_frames=4):

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
            
            # Determine color: red for dragging, gray for stance, orange for clean swing
            # Only show red if there are at least drag_min_consecutive_frames consecutive dragging frames
            is_dragging_frame = False
            if is_drag is not None and len(is_drag) > t and is_drag[t]:
                # Check if this dragging frame is part of a sequence with minimum consecutive frames
                drag_start = t
                drag_end = t
                # Find start of dragging sequence
                while drag_start > 0 and is_drag[drag_start - 1]:
                    drag_start -= 1
                # Find end of dragging sequence  
                while drag_end < len(is_drag) - 1 and is_drag[drag_end + 1]:
                    drag_end += 1
                # Only show red if sequence is at least drag_min_consecutive_frames
                if drag_end - drag_start + 1 >= drag_min_consecutive_frames:
                    is_dragging_frame = True
            
            if is_dragging_frame:
                color = '#FF0000'  # Red for dragging (only if minimum consecutive frames met)
            elif is_stance[t] == 1:
                color = '#999999'  # Gray for stance
            else:
                color = '#FAAF40'  # Orange for swing
            
            plt.plot('coord x', 'coord y', data=toe_mtp, c=color)
            plt.plot('coord x', 'coord y', data=mtp_ankle, c=color)
            plt.plot('coord x', 'coord y', data=ankle_knee, c=color)
            plt.plot('coord x', 'coord y', data=knee_hip, c=color)
            plt.plot('coord x', 'coord y', data=hip_crest, c=color)

    plt.xlabel('x coordinate (pixel)')
    plt.ylabel('y coordinate (pixel)')
    plt.legend([f'{bodyparts[0]}-{bodyparts[1]}', 
                f'{bodyparts[1]}-{bodyparts[2]}', 
                f'{bodyparts[2]}-{bodyparts[3]}', 
                f'{bodyparts[3]}-{bodyparts[4]}', 
                f'{bodyparts[4]}-{bodyparts[5]}'], 
                loc='upper right')

    plt.ylim(y_min, y_max)
    plt.gca().invert_yaxis()
    plt.savefig(filename, dpi=1200)
    plt.close()