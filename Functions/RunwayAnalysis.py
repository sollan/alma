"""Confidence-aware overground gait analysis with frame-preserving gaps.

Only complete contact-to-contact cycles inside a moving bout are scored.
Detection may bridge explicitly bounded short gaps. Optional reconstruction
for joint summaries has separate time, geometry and per-interval limits.
"""
import math

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

from Functions.TreadmillValidation import BODY_PARTS, PARAMETER_COLUMNS, _positive_number
from Functions.JointInterpolation import reconstruct_short_gaps, interpolation_context

COLUMNS = ['limb (hind left / right)', *PARAMETER_COLUMNS]
ANGLE_PARTS = {'mtp': ('toe', 'mtp', 'ankle'), 'ankle': ('mtp', 'ankle', 'knee'),
               'knee': ('ankle', 'knee', 'hip'), 'hip': ('knee', 'hip', 'iliac crest')}


def runs(mask):
    """Return half-open runs of True without compressing the frame axis."""
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def fill_short_gaps(values, max_frames):
    result = np.asarray(values, dtype=float).copy()
    filled = np.zeros(len(result), dtype=bool)
    for start, end in runs(~np.isfinite(result)):
        if start > 0 and end < len(result) and end - start <= max_frames:
            result[start:end] = np.linspace(result[start - 1], result[end], end - start + 2)[1:-1]
            filled[start:end] = True
    return result, filled


def filter_segments(values, frame_rate, cutoff):
    """Filter finite runs independently; short runs remain unavailable."""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    b, a = butter(5, cutoff / (frame_rate / 2), 'low')
    padding = 3 * max(len(a), len(b))
    for start, end in runs(np.isfinite(values)):
        if end - start > padding:
            result[start:end] = filtfilt(b, a, values[start:end])
    return result


def _noise(values):
    values = np.asarray(values, dtype=float)
    estimates = []
    for start, end in runs(np.isfinite(values)):
        if end - start >= 5:
            difference = np.diff(values[start:end], n=2)
            estimates.extend(difference.tolist())
    if not estimates:
        return 0.
    estimates = np.asarray(estimates)
    return float(1.4826 * np.median(np.abs(estimates - np.median(estimates))) / np.sqrt(6))


def _derivative(values, frame_rate):
    result = np.full(len(values), np.nan)
    for start, end in runs(np.isfinite(values)):
        if end - start >= 3:
            result[start:end] = np.gradient(values[start:end]) * frame_rate
    return result


def _event_filter(values, frame_rate, window_s):
    result = np.full_like(values, np.nan, dtype=float)
    window = max(3, int(round(window_s * frame_rate)) | 1)
    for start,end in runs(np.isfinite(values).all(axis=1)):
        if end-start >= window:
            result[start:end] = savgol_filter(values[start:end],window,2,axis=0)
    return result


def _clean_runs(mask, minimum, join_gap, allowed):
    result = np.asarray(mask, dtype=bool).copy()
    if join_gap:
        for start, end in runs(~result):
            if start > 0 and end < len(result) and end - start <= join_gap and allowed[start:end].all():
                result[start:end] = True
    for start, end in runs(result):
        if end - start < minimum:
            result[start:end] = False
    return result


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


def _angles(coords, first, center, last):
    a = coords[first] - coords[center]
    b = coords[last] - coords[center]
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.full(len(a), np.nan)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (denominator > 0)
    cosine[valid] = np.sum(a[valid] * b[valid], axis=1) / denominator[valid]
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def _metric_quality(parts, start, end, observed, filled, context, values, maximum):
    reconstructed = np.logical_or.reduce([filled[bp][start:end] for bp in parts])
    valid_observations = np.logical_and.reduce([observed[bp][start:end] for bp in parts])
    fraction = float(np.mean(reconstructed))
    reason = None
    if not np.isfinite(values).all():
        reason = 'incomplete valid coordinates or degenerate geometry'
    elif fraction > maximum + 1e-12:
        reason = 'joint interpolation fraction exceeds per-metric interval limit'
    influenced = bool(any(context[bp][start:end].any() for bp in parts))
    return {'body_parts': list(parts), 'observed_frames': int(valid_observations.sum()),
            'interpolated_frames': int(reconstructed.sum()), 'interpolated_fraction': fraction,
            'filter_context_has_interpolation': influenced,
            'status': 'unavailable' if reason else ('estimated' if influenced else 'observed'),
            'unavailable_reason': reason}, reconstructed


def analyze_runway(frame_rate, dataframe, cutoff_f, pixels_per_cm,
                   no_outlier_filter=False, dragging_filter=False,
                   step_height_min_cm=0., step_height_max_cm=1.5,
                   stride_length_min_cm=0., stride_length_max_cm=8.,
                   drag_clearance_cm=.3, drag_min_consecutive_frames=4,
                   confidence_threshold=.6, max_gap_s=.02,
                   min_body_speed=.25, min_contact_s=.03, min_swing_s=.02,
                   right_to_left='auto', contact_smoothing_s=.025, min_stride_progress=.05,
                   joint_interpolation='none', joint_max_gap_s=.02,
                   joint_max_interpolated_fraction=.1, joint_max_segment_change=.35,
                   joint_cutoff_f=None):
    from Functions import KinematicsFunctions as helpers

    for name, value in [('Frame rate', frame_rate), ('Filter cutoff', cutoff_f),
                        ('Measured pixels per cm', pixels_per_cm),
                        ('Minimum contact duration', min_contact_s), ('Minimum swing duration', min_swing_s),
                        ('Contact smoothing duration',contact_smoothing_s)]:
        _positive_number(name, value)
    if cutoff_f >= frame_rate / 2:
        raise ValueError('Filter cutoff must be less than half the frame rate.')
    joint_cutoff = cutoff_f if joint_cutoff_f is None else joint_cutoff_f
    _positive_number('Joint filter cutoff', joint_cutoff)
    if joint_cutoff >= frame_rate/2:
        raise ValueError('Joint filter cutoff must be less than half the frame rate.')
    if not np.isfinite(confidence_threshold) or not 0 < confidence_threshold < 1:
        raise ValueError('Tracking confidence threshold must be between 0 and 1.')
    if not np.isfinite([max_gap_s, min_body_speed, drag_clearance_cm, min_stride_progress]).all() or min(max_gap_s, min_body_speed, drag_clearance_cm, min_stride_progress) < 0:
        raise ValueError('Maximum gap, minimum body speed and drag clearance must be finite and non-negative.')
    if not isinstance(drag_min_consecutive_frames, (int, np.integer)) or isinstance(drag_min_consecutive_frames, bool) or drag_min_consecutive_frames < 1:
        raise ValueError('Consecutive drag frames must be a positive integer.')
    for name, low, high in [('Step height', step_height_min_cm, step_height_max_cm),
                            ('Stride length', stride_length_min_cm, stride_length_max_cm)]:
        if not np.isfinite([low, high]).all() or low < 0 or high < low:
            raise ValueError(f'{name} limits must be finite, non-negative and minimum <= maximum.')
    if right_to_left not in (True, False, 'auto', None):
        raise ValueError('Walking direction must be True, False, "auto", or None.')
    if joint_interpolation not in ('none', 'linear', 'pchip'):
        raise ValueError('Joint interpolation must be none, linear or pchip.')
    for name, value, upper in [('Joint maximum gap', joint_max_gap_s, None),
                              ('Joint maximum interpolated fraction', joint_max_interpolated_fraction, 1.),
                              ('Joint maximum segment change', joint_max_segment_change, 1.)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value < 0 or (upper is not None and value > upper):
            raise ValueError(f'{name} must be finite, non-negative' + (f' and <= {upper}.' if upper else '.'))
    if not isinstance(dataframe, pd.DataFrame) or dataframe.columns.has_duplicates:
        raise ValueError('Provide a DataFrame with unique mapped columns for one limb. Analyze limbs separately.')
    data = dataframe.copy(deep=True).reset_index(drop=True)
    data.attrs = {}
    n = len(data)
    required = [f'toe {coord}' for coord in ('x', 'y', 'likelihood')]
    missing = [col for col in required if col not in data]
    if missing:
        raise ValueError('Missing required foot-tracking columns: ' + ', '.join(missing))
    index_column = 'bodyparts coords'
    if index_column in data and n > 1:
        labels = pd.to_numeric(data[index_column], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.allclose(np.diff(labels), 1.):
            raise ValueError('Source frame labels must be consecutive. Preserve missing frames as rows rather than compressing recording time.')
    metric = {}
    observed_metric = {}
    observations = {}
    detection = {}
    filled_coordinates = {}
    original_valid = {}
    imputed = {}
    quality = {}
    absent = []
    max_gap_frames = int(math.floor(max_gap_s * frame_rate + 1e-9))
    for bp in BODY_PARTS:
        names = [f'{bp} {c}' for c in ('x', 'y', 'likelihood')]
        if any(col not in data for col in names):
            absent.append(bp)
            for col in names:
                data[col] = np.nan
        try:
            x, y, likelihood = (pd.to_numeric(data[col], errors='raise').to_numpy(dtype=float) for col in names)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{bp} tracking columns must be numeric.') from exc
        finite_likelihood = likelihood[np.isfinite(likelihood)]
        if ((finite_likelihood < 0) | (finite_likelihood > 1)).any():
            raise ValueError(f'{bp} likelihood must be between 0 and 1.')
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= confidence_threshold)
        original_valid[bp] = valid
        masked_x, masked_y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
        observations[bp] = np.column_stack([masked_x, masked_y])
        metric[bp] = np.column_stack([filter_segments(masked_x, frame_rate, joint_cutoff),
                                      filter_segments(masked_y, frame_rate, joint_cutoff)])
        observed_metric[bp] = metric[bp].copy()
        if bp in ('toe','hip','iliac crest'):
            filled_x, fx = fill_short_gaps(masked_x, max_gap_frames)
            filled_y, fy = fill_short_gaps(masked_y, max_gap_frames)
        else:
            filled_x, filled_y = masked_x, masked_y
            fx = fy = np.zeros(n,dtype=bool)
        filled_coordinates[bp] = np.column_stack([filled_x,filled_y])
        detection[bp] = np.column_stack([filter_segments(filled_x, frame_rate, cutoff_f),
                                         filter_segments(filled_y, frame_rate, cutoff_f)])
        imputed[bp] = fx | fy
        quality[bp] = {'valid_observation_frames': int(valid.sum()), 'unavailable_observation_frames': int((~valid).sum()),
                       'short_gap_frames_interpolated_for_detection': int(imputed[bp].sum()),
                       'missing_coordinate_frames': int((~(np.isfinite(x) & np.isfinite(y))).sum())}
        # Preserve both raw samples and the observed-only filtered comparison.
        data[f'{bp} raw x'], data[f'{bp} raw y'] = x, y
        data[f'{bp} observed filtered x'], data[f'{bp} observed filtered y'] = metric[bp][:, 0], metric[bp][:, 1]
        data[f'{bp} x'], data[f'{bp} y'] = metric[bp][:, 0], metric[bp][:, 1]
        data[f'{bp} observed valid'] = valid
        data[f'{bp} metric valid'] = np.isfinite(metric[bp]).all(axis=1)
        data[f'{bp} detection interpolated'] = imputed[bp]
    data = data.copy()  # consolidate added coordinate/quality columns
    joint_gap_frames = int(math.floor(joint_max_gap_s*frame_rate + 1e-9))
    joint_filled = {bp: np.zeros(n, dtype=bool) for bp in BODY_PARTS}
    joint_context = {bp: np.zeros(n, dtype=bool) for bp in BODY_PARTS}
    joint_reports = {bp: {'filled_gaps': [], 'rejected_gaps': []} for bp in BODY_PARTS}
    if joint_interpolation != 'none':
        reconstructed, joint_filled, joint_reports = reconstruct_short_gaps(
            observations, joint_gap_frames, joint_interpolation, joint_max_segment_change,
            context_frames=max(6, int(round(.1*frame_rate))))
        for bp in BODY_PARTS:
            metric[bp] = np.column_stack([filter_segments(reconstructed[bp][:, axis], frame_rate, joint_cutoff) for axis in (0, 1)])
            joint_context[bp] = interpolation_context(reconstructed[bp], joint_filled[bp])
            data[f'{bp} x'], data[f'{bp} y'] = metric[bp][:, 0], metric[bp][:, 1]
            data[f'{bp} metric valid'] = np.isfinite(metric[bp]).all(axis=1)
    for bp in BODY_PARTS:
        data[f'{bp} metric interpolated'] = joint_filled[bp]
        data[f'{bp} metric interpolation context'] = joint_context[bp]
        quality[bp]['short_gap_frames_interpolated_for_joint_metrics'] = int(joint_filled[bp].sum())
    anchors = [bp for bp in ('hip', 'iliac crest') if np.isfinite(detection[bp]).all(axis=1).sum() > 18]
    if not anchors and not any(bp not in absent for bp in ('hip', 'iliac crest')):
        raise ValueError('Runway detection needs hip or iliac crest tracking as a body reference; a knee is not required.')
    anchor = max(anchors, key=lambda bp: np.isfinite(detection[bp]).all(axis=1).sum()) if anchors else 'hip'
    toe, body = detection['toe'], detection[anchor]
    data['body event x'],data['body event y'] = body[:,0],body[:,1]
    event_toe = _event_filter(filled_coordinates['toe'],frame_rate,contact_smoothing_s)
    data['toe event x'],data['toe event y'] = event_toe[:,0],event_toe[:,1]
    data['toe analysis x'],data['toe analysis y'] = toe[:,0],toe[:,1]
    valid = np.isfinite(toe).all(axis=1) & np.isfinite(event_toe).all(axis=1) & np.isfinite(body).all(axis=1)
    distances = np.linalg.norm(toe[valid] - body[valid], axis=1)
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else np.nan
    phase = np.full(n, -1, dtype=np.int8)
    report = {'report_version': 1, 'analysis_type': 'spontaneous', 'detector': 'bout_contact_v1',
        'frames': n, 'frame_number_basis': 'zero-based CSV row position', 'interval_convention': '[start_frame, end_frame)',
        'settings': dict(frame_rate=float(frame_rate), filter_cutoff_hz=float(cutoff_f),
            joint_filter_cutoff_hz=float(joint_cutoff), joint_filter_inherits_main_cutoff=joint_cutoff_f is None,
            pixels_per_cm=float(pixels_per_cm),
            confidence_threshold=float(confidence_threshold), max_gap_s=float(max_gap_s),
            min_body_speed_reference_lengths_s=float(min_body_speed), min_contact_s=float(min_contact_s), min_swing_s=float(min_swing_s),
            contact_smoothing_s=float(contact_smoothing_s), min_stride_progress_reference_lengths=float(min_stride_progress),
            effective_contact_smoothing_frames=max(3,int(round(contact_smoothing_s*frame_rate)) | 1),
            max_gap_frames=max_gap_frames,
            joint_interpolation=joint_interpolation, joint_max_gap_s=float(joint_max_gap_s),
            joint_max_gap_frames=joint_gap_frames, joint_max_interpolated_fraction=float(joint_max_interpolated_fraction),
            joint_max_segment_change=float(joint_max_segment_change),
            requested_right_to_left=right_to_left, step_height_min_cm=float(step_height_min_cm), step_height_max_cm=float(step_height_max_cm),
            stride_length_min_cm=float(stride_length_min_cm), stride_length_max_cm=float(stride_length_max_cm),
            drag_clearance_cm=float(drag_clearance_cm), drag_min_consecutive_frames=int(drag_min_consecutive_frames),
            no_outlier_filter=bool(no_outlier_filter), dragging_filter=bool(dragging_filter)),
        'calibration': {'method': 'manual', 'pixels_per_cm': float(pixels_per_cm)},
        'tracking_confidence': quality, 'body_reference': anchor, 'body_scale_pixels': _finite_number(scale),
        'reference_length_definition': 'median valid toe-to-body-reference separation; not an assumed anatomical body length',
        'bouts': [], 'intervals': [], 'warnings': [], 'candidate_intervals': 0, 'retained_intervals': 0,
        'rejected_intervals': 0, 'discarded_edge_contacts': 0, 'status': 'no_usable_strides'}
    report['variability_method'] = 'pairwise mean trajectory difference after temporal interpolation; pixel-coordinate units'
    report['joint_interpolation'] = {'method': joint_interpolation, 'body_parts': joint_reports,
        'geometry_reference': 'local endpoint segment lengths and observed adjacent joints; no assumed anatomical lengths',
        'motion_reference': 'nearby observed steps; limit is max(3*95th percentile, median+6*MAD)',
        'fraction_definition': 'union of directly filled frames for the body parts of each metric, divided by interval frames',
        'filter_context_definition': 'entire finite filter run touched by a filled gap; can affect surrounding observed samples',
        'accuracy_note': 'Reconstructed coordinates and peaks are estimates; confidence scores do not establish anatomical truth.'}
    report['coordinate_sources'] = {'joint_metrics': 'finite-run Butterworth coordinates in x/y columns; optional bounded joint reconstruction' if joint_interpolation != 'none' else 'observed-only finite-run Butterworth coordinates in x/y columns',
                                    'observed_comparison': 'raw x/y preserve input; observed filtered x/y preserve the confidence-masked, unfilled comparison',
                                    'foot_metrics':'Butterworth foot coordinates in toe analysis x/y, with bounded short-gap interpolation',
                                    'event_detection':'short-window Savitzky-Golay foot coordinates in toe event x/y'}
    if absent:
        report['warnings'].append('Unavailable body parts: ' + ', '.join(absent) + '. Dependent metrics will be unavailable.')
    if any(q['unavailable_observation_frames'] for q in quality.values()):
        report['warnings'].append('Tracking gaps and low-confidence observations are masked. Event/foot interpolation and optional joint reconstruction have separate settings and flags.')
    if any(mask.any() for mask in joint_filled.values()):
        report['warnings'].append('Joint reconstruction is enabled. Estimated coordinates can influence filtered observations and angle peaks. Review per-metric coverage, peak flags and observed-only comparisons.')
    rows, masks, starts = [], [], []
    if not np.isfinite(scale) or scale <= 0 or n < 19:
        report['warnings'].append('Insufficient valid foot/body tracking for complete-cycle detection.')
        return _result(rows, data, phase, masks, starts, report)
    body_velocity = _derivative(body[:, 0], frame_rate)
    toe_velocity = _derivative(event_toe[:, 0], frame_rate)
    noise_x = _noise(np.where(original_valid['toe'], dataframe['toe x'].to_numpy(dtype=float), np.nan))
    body_noise = _noise(np.where(original_valid[anchor], dataframe[f'{anchor} x'].to_numpy(dtype=float), np.nan))
    velocity_noise = 3 * noise_x * frame_rate / max(3, round(.03 * frame_rate))
    speed_floor = max(min_body_speed * scale, 3 * body_noise * frame_rate / max(3, round(.10 * frame_rate)))
    report['thresholds'] = {'minimum_body_speed_pixels_s': float(speed_floor),
                            'toe_velocity_noise_floor_pixels_s': float(velocity_noise),
                            'relative_contact_speed_enter': .35, 'relative_contact_speed_exit': .65,
                            'relative_contact_height': .10}
    min_contact = max(1, int(math.ceil(min_contact_s * frame_rate)))
    min_swing = max(1, int(math.ceil(min_swing_s * frame_rate)))
    report['settings'].update(min_contact_frames=min_contact,min_swing_frames=min_swing)
    min_bout = max(19, int(math.ceil(.15 * frame_rate)))
    joint_angles = {joint: _angles(metric, *triplet) for joint, triplet in ANGLE_PARTS.items()}
    observed_angles = {joint: _angles(observed_metric, *triplet) for joint, triplet in ANGLE_PARTS.items()}
    for sign in (1, -1):
        moving = valid & (body_velocity * sign > speed_floor)
        moving = _clean_runs(moving, min_bout, int(round(.08 * frame_rate)), valid & (body_velocity * sign > -speed_floor))
        for lo, hi in runs(moving):
            if right_to_left in (True, False) and sign != (-1 if right_to_left else 1):
                report['warnings'].append(f'Frames {lo}-{hi-1}: body motion disagrees with the requested direction; bout not scored.')
                continue
            bout_id = len(report['bouts'])
            window = max(3, int(round(.4 * frame_rate)) | 1)
            ground = pd.Series(event_toe[lo:hi, 1]).rolling(window, center=True, min_periods=1).quantile(.85).to_numpy()
            height = ground - event_toe[lo:hi, 1]
            metric_ground = pd.Series(toe[lo:hi,1]).rolling(window,center=True,min_periods=1).quantile(.85).to_numpy()
            metric_height = metric_ground-toe[lo:hi,1]
            local_speed = np.abs(toe_velocity[lo:hi])
            enter = np.maximum(velocity_noise, .35 * np.abs(body_velocity[lo:hi]))
            leave = np.maximum(velocity_noise * 1.5, .65 * np.abs(body_velocity[lo:hi]))
            contact = np.zeros(hi-lo, dtype=bool)
            state = False
            for t in range(hi-lo):
                if local_speed[t] <= enter[t] and height[t] <= .10 * scale:
                    state = True
                elif local_speed[t] > leave[t] or height[t] > .15 * scale:
                    state = False
                contact[t] = state
            contact = _clean_runs(contact, min_contact, min_swing-1, np.ones(hi-lo, dtype=bool))
            phase[lo:hi] = contact.astype(np.int8)
            contacts = runs(contact)
            # An initial supported foot is a partial observation, not a new contact.
            onsets = [(lo+s, lo+e) for s,e in contacts if s >= min_swing]
            report['discarded_edge_contacts'] += int(bool(contacts and contacts[0][0] < min_swing))
            report['discarded_edge_contacts'] += int(bool(onsets))  # final onset has no following contact
            report['bouts'].append({'id': bout_id, 'start_frame': int(lo), 'end_frame_exclusive': int(hi),
                                     'right_to_left': bool(sign < 0), 'contact_onsets': [int(s) for s,_ in onsets]})
            for (start, lift), (end, _) in zip(onsets, onsets[1:]):
                if end-lift < min_swing:
                    continue
                row = {col: np.nan for col in COLUMNS}
                row['limb (hind left / right)'] = ''
                row['stride_start (frame)'], row['stride_end (frame)'] = start, end
                length = abs(toe[end, 0] - toe[start, 0]) / pixels_per_cm
                step = max(0., float(np.max(metric_height[start-lo:end-lo]))) / pixels_per_cm
                duration = (end-start) / frame_rate
                swing_duration = (end-lift) / frame_rate
                relative_y = toe[:, 1].copy()
                relative_y[lo:hi] -= metric_ground
                drag_frames, drag_fraction, drag_mask = helpers.find_drag(relative_y, 0, 0., lift, end,
                                                                         drag_clearance_cm*pixels_per_cm, drag_min_consecutive_frames)
                reasons = []
                progress_floor = max(min_stride_progress*scale,6*noise_x)
                if sign*(event_toe[end,0]-event_toe[start,0]) < progress_floor:
                    reasons.append('insufficient_foot_progress_for_cycle')
                if sign*(body[end,0]-body[start,0]) < max(min_stride_progress*scale,6*body_noise):
                    reasons.append('insufficient_body_progress_for_cycle')
                if not no_outlier_filter:
                    for value, minimum, maximum, prefix in [(length,stride_length_min_cm,stride_length_max_cm,'stride_length'),
                                                            (step,step_height_min_cm,step_height_max_cm,'step_height')]:
                        if value < minimum: reasons.append(prefix+'_below_minimum')
                        if value > maximum: reasons.append(prefix+'_above_maximum')
                if dragging_filter and drag_fraction >= .95:
                    reasons.append('near_continuous_dragging_filter')
                interval = {'start_frame': int(start), 'end_frame': int(end), 'stance_end_frame': int(lift),
                    'bout_id': bout_id, 'retained': not reasons, 'rejection_reasons': reasons,
                    'stride_length_cm': float(length), 'step_height_cm': float(step), 'unavailable_metrics': {},
                    'metric_quality': {}, 'observed_only_metrics': {},
                    'detection_interpolated_frames': int((imputed['toe'][start:end] | imputed[anchor][start:end]).sum())}
                interval['detection_context_has_interpolation'] = bool(imputed['toe'][lo:hi].any() or imputed[anchor][lo:hi].any())
                if not reasons:
                    row.update({'cycle duration (s)': duration, 'cycle duration (no. frames)': end-start,
                        'cycle velocity (cm/s)': length/duration, 'stride length (cm)': length,
                        'stance duration (s)': (lift-start)/frame_rate, 'swing duration (s)': swing_duration,
                        'swing percentage (%)': (end-lift)/(end-start), 'stance percentage (%)': (lift-start)/(end-start),
                        'step height (cm)': step, 'drag duration (s)': float(drag_frames/frame_rate),
                        'drag percentage (%)': float(drag_fraction)})
                    # Joint reconstruction is independent of event/foot interpolation.
                    distance = np.linalg.norm(metric['toe'][start:end] - metric['iliac crest'][start:end], axis=1)/pixels_per_cm
                    observed_distance = np.linalg.norm(observed_metric['toe'][start:end] - observed_metric['iliac crest'][start:end], axis=1)/pixels_per_cm
                    distance_columns = ['mean toe-to-crest distance (cm)','max toe-to-crest distance (cm)',
                                        'min toe-to-crest distance (cm)','toe-to-crest distance SD (cm)']
                    distance_quality, _ = _metric_quality(('toe', 'iliac crest'), start, end,
                        original_valid, joint_filled, joint_context, distance, joint_max_interpolated_fraction)
                    interval['metric_quality']['toe_to_crest_distance'] = distance_quality
                    observed_values = [np.mean(observed_distance), np.max(observed_distance), np.min(observed_distance), np.std(observed_distance)] if np.isfinite(observed_distance).all() else [np.nan]*4
                    interval['observed_only_metrics'].update({col: _finite_number(value) for col, value in zip(distance_columns, observed_values)})
                    if distance_quality['status'] != 'unavailable':
                        for col,value in zip(distance_columns,[np.mean(distance),np.max(distance),np.min(distance),np.std(distance)]):
                            row[col] = float(value)
                    else:
                        for col in distance_columns: interval['unavailable_metrics'][col] = distance_quality['unavailable_reason']
                    speed = np.hypot(np.gradient(toe[lift:end,0]),np.gradient(toe[lift:end,1]))*frame_rate/pixels_per_cm if end-lift>1 else np.array([])
                    row['max velocity during swing (cm/s)'] = float(np.max(speed)) if len(speed) else np.nan
                    if not len(speed):
                        interval['unavailable_metrics']['max velocity during swing (cm/s)'] = 'at least two movement-phase samples are required'
                    for joint,all_angles in joint_angles.items():
                        angle = all_angles[start:end]
                        columns = [f'{joint} joint {stat}' for stat in ('extension (deg)','flexion (deg)','amplitude (deg)','SD (deg)')]
                        quality_item, estimated_frames = _metric_quality(ANGLE_PARTS[joint], start, end,
                            original_valid, joint_filled, joint_context, angle, joint_max_interpolated_fraction)
                        interval['metric_quality'][joint+'_angle'] = quality_item
                        original_angle = observed_angles[joint][start:end]
                        original_values = [np.max(original_angle), np.min(original_angle), np.ptp(original_angle), np.std(original_angle)] if np.isfinite(original_angle).all() else [np.nan]*4
                        interval['observed_only_metrics'].update({col: _finite_number(value) for col, value in zip(columns, original_values)})
                        if quality_item['status'] != 'unavailable':
                            for col,value in zip(columns,[np.max(angle),np.min(angle),np.ptp(angle),np.std(angle)]): row[col] = float(value)
                            for peak, offset in [('extension', int(np.argmax(angle))), ('flexion', int(np.argmin(angle)))]:
                                quality_item[peak+'_frame'] = int(start+offset)
                                quality_item[peak+'_at_interpolated_frame'] = bool(estimated_frames[offset])
                        else:
                            for col in columns: interval['unavailable_metrics'][col] = quality_item['unavailable_reason']
                    masks.append(drag_mask)
                else:
                    masks.append(np.array([], dtype=bool))
                starts.append(int(start))
                rows.append(row)
                report['intervals'].append(interval)
    # Direction groups are processed separately; restore chronological row order.
    order = sorted(range(len(rows)),key=lambda i: starts[i])
    rows, masks, starts = [rows[i] for i in order], [masks[i] for i in order], [starts[i] for i in order]
    report['intervals'] = [report['intervals'][i] for i in order]
    _variability(rows, report['intervals'], toe, helpers)
    return _result(rows, data, phase, masks, starts, report)


def _variability(rows, intervals, toe, helpers):
    for i,row in enumerate(rows):
        if not intervals[i]['retained']:
            continue
        for count in (5,10):
            group = intervals[i:i+count]
            available = len(group) == count and all(r['retained'] and r['bout_id']==group[0]['bout_id'] for r in group)
            available = available and all(a['end_frame']==b['start_frame'] for a,b in zip(group,group[1:]))
            paths = [toe[r['start_frame']:r['end_frame']+1] for r in group] if available else []
            if not available or not all(np.isfinite(path).all() for path in paths):
                for axis in ('x','y','xy'):
                    for statistic in ('mean','SD'):
                        intervals[i]['unavailable_metrics'][f'Variability {axis} plane {count} strides {statistic}'] = f'requires {count} consecutive retained cycles with valid foot trajectories in one bout'
                continue
            centered = [path-path[0] for path in paths]
            for name,trajectories in [('x',[p[:,0] for p in centered]),('y',[p[:,1] for p in centered]),('xy',centered)]:
                differences = helpers.pairwise_dtw(trajectories)
                row[f'Variability {name} plane {count} strides mean'] = float(np.mean(differences))
                row[f'Variability {name} plane {count} strides SD'] = float(np.std(differences))


def _result(rows, data, phase, masks, starts, report):
    parameters = pd.DataFrame(rows, columns=COLUMNS)
    candidate = np.full(len(data),np.nan)
    retained = np.full(len(data),np.nan)
    drag = np.full(len(data),np.nan)
    for i,(interval,mask) in enumerate(zip(report['intervals'],masks)):
        start,end = interval['start_frame'],interval['end_frame']
        candidate[start:end] = i+1
        retained[start:end] = int(interval['retained'])
        if interval['retained']:
            drag[start:end] = 0
            lift = interval['stance_end_frame']
            drag[lift:lift+len(mask)] = np.asarray(mask,dtype=float)
    data['runway phase'] = phase
    data['runway candidate interval'] = candidate
    data['runway retained interval'] = retained
    data['runway drag'] = drag
    report['phase_codes'] = {'-1':'unscored','0':'foot movement','1':'support'}
    report['candidate_intervals'] = len(rows)
    report['retained_intervals'] = sum(interval['retained'] for interval in report['intervals'])
    report['rejected_intervals'] = len(rows)-report['retained_intervals']
    if report['retained_intervals']:
        if any(any(not metric.startswith('Variability ') for metric in interval['unavailable_metrics'])
               for interval in report['intervals'] if interval['retained']):
            report['warnings'].append('Some joint/distance metrics are unavailable on retained intervals; timing remains scored. See per-interval reasons.')
        if any(interval['detection_context_has_interpolation'] for interval in report['intervals']):
            report['warnings'].append('Some event timing uses bounded short-gap interpolation. Review flagged intervals against video.')
        report['status'] = 'completed_with_warnings' if report['warnings'] else 'completed'
    else:
        report['no_usable_reason'] = 'all_complete_cycles_rejected' if rows else ('no_complete_cycles' if report['bouts'] else 'no_valid_moving_bouts')
        if rows:
            report['warnings'].append('All complete cycles were rejected. Inspect per-interval rejection reasons, spatial calibration and acceptance limits.')
        else:
            report['warnings'].append('No complete contact-to-contact cycles were found inside valid moving bouts; no partial or stationary cycle was manufactured.')
    report['variability_windows_available'] = {str(count): int(parameters[f'Variability x plane {count} strides mean'].notna().sum()) for count in (5,10)}
    if any(count==0 for count in report['variability_windows_available'].values()):
        report['warnings'].append('Five/ten-stride variability requires enough consecutive retained cycles in the same bout; unavailable windows remain NaN.')
        if report['status'] == 'completed':
            report['status'] = 'completed_with_warnings'
    parameters.attrs['analysis_report'] = report
    data.attrs['analysis_report'] = report
    return parameters, data, phase, list(BODY_PARTS), masks, starts


def plot_continuous(parameters, coordinates, phase, drag_masks, count, filename, plot, coordinate_source='analysis'):
    """Select one contiguous retained bout; never draw across a rejected/gap interval."""
    intervals = parameters.attrs['analysis_report']['intervals']
    groups = []
    current = []
    for i,item in enumerate(intervals):
        if not item['retained']:
            if current: groups.append(current)
            current = []
        elif current and (item['bout_id'] != intervals[current[-1]]['bout_id'] or
                          item['start_frame'] != intervals[current[-1]]['end_frame']):
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    if current: groups.append(current)
    if not groups:
        return parameters.iloc[:0]
    selected = max(groups,key=len)[:max(1,int(count))]
    subset = parameters.iloc[selected]
    if not plot:
        return subset
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from Functions.SkeletonVisualization import skeleton_coordinates, LOW_CONFIDENCE, RECONSTRUCTED
    xy, states = skeleton_coordinates(coordinates, coordinate_source,
        parameters.attrs['analysis_report']['settings']['confidence_threshold'])
    fig,ax = plt.subplots(figsize=(12,5))
    for i in selected:
        item = intervals[i]
        for frame in range(item['start_frame'],item['end_frame'],2):
            color = '#999999' if phase[frame] == 1 else '#FAAF40'
            offset = frame-item['stance_end_frame']
            if 0 <= offset < len(drag_masks[i]) and drag_masks[i][offset]:
                color = '#d62762'
            for joint in range(len(BODY_PARTS)-1):
                points = xy[frame, joint:joint+2]
                status = states[frame, joint:joint+2]
                low = bool((status == LOW_CONFIDENCE).any())
                estimated = bool((status == RECONSTRUCTED).any())
                if np.isfinite(points).all():
                    ax.plot(points[:,0],points[:,1],color='#b0b0b0' if low else color,
                            linewidth=.8, linestyle=':' if low else ('--' if estimated else '-'))
    ax.set_aspect('equal',adjustable='datalim')
    ax.invert_yaxis()
    label = 'tracked keypoints (unfiltered)' if coordinate_source == 'tracked' else 'analysis coordinates (filtered)'
    ax.set(xlabel='x coordinate (pixel)',ylabel='y coordinate (pixel)',title=f'{len(selected)} consecutive retained cycles | {label}')
    ax.legend(handles=[Line2D([0],[0],color=color,label=label) for color,label in
                        [('#999999','Support'),('#FAAF40','Foot movement'),('#d62762','Near-ground movement')]] +
                      [Line2D([0],[0],color='#b0b0b0',linestyle=':',label='Low-confidence tracking'),
                       Line2D([0],[0],color='#555555',linestyle='--',label='Reconstructed endpoint')],
              loc='best',fontsize=8)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return subset
