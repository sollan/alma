"""Validation and diagnostic reports for the existing treadmill calculation.

Coordinates and thresholds are never repaired or changed implicitly here.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

BODY_PARTS = ('toe', 'mtp', 'ankle', 'knee', 'hip', 'iliac crest')
PARAMETER_COLUMNS = [
    'stride_start (frame)', 'stride_end (frame)', 'cycle duration (s)',
    'cycle duration (no. frames)', 'cycle velocity (cm/s)', 'stride length (cm)',
    'stance duration (s)', 'swing duration (s)', 'swing percentage (%)',
    'stance percentage (%)', 'mean toe-to-crest distance (cm)',
    'max toe-to-crest distance (cm)', 'min toe-to-crest distance (cm)',
    'toe-to-crest distance SD (cm)', 'step height (cm)',
    'max velocity during swing (cm/s)',
    'mtp joint extension (deg)', 'mtp joint flexion (deg)',
    'mtp joint amplitude (deg)', 'mtp joint SD (deg)',
    'ankle joint extension (deg)', 'ankle joint flexion (deg)',
    'ankle joint amplitude (deg)', 'ankle joint SD (deg)',
    'knee joint extension (deg)', 'knee joint flexion (deg)',
    'knee joint amplitude (deg)', 'knee joint SD (deg)',
    'hip joint extension (deg)', 'hip joint flexion (deg)',
    'hip joint amplitude (deg)', 'hip joint SD (deg)',
    'drag duration (s)', 'drag percentage (%)',
    'Variability x plane 5 strides mean', 'Variability x plane 5 strides SD',
    'Variability y plane 5 strides mean', 'Variability y plane 5 strides SD',
    'Variability xy plane 5 strides mean', 'Variability xy plane 5 strides SD',
    'Variability x plane 10 strides mean', 'Variability x plane 10 strides SD',
    'Variability y plane 10 strides mean', 'Variability y plane 10 strides SD',
    'Variability xy plane 10 strides mean', 'Variability xy plane 10 strides SD',
]


def _positive_number(name, value):
    try:
        valid = np.isscalar(value) and np.isfinite(value) and value > 0
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a finite number greater than zero.')


def resolve_calibration(dataframe, method, pixels_per_cm=None,
                        reference_segment=None, reference_length_cm=None):
    """Resolve only the explicitly selected spatial scale, without fallback.

    Belt-motion correction remains in pixel coordinates for all three modes.
    Reference lengths are user-supplied measurements, never species defaults.
    """
    if method == 'belt_motion':
        if pixels_per_cm is not None or reference_segment is not None or reference_length_cm is not None:
            raise ValueError('Belt-motion calibration does not use a manual scale or reference segment. Select the corresponding calibration method explicitly.')
        return None
    if method == 'manual':
        _positive_number('Pixels per cm', pixels_per_cm)
        return {'method': 'manual', 'pixels_per_cm': float(pixels_per_cm)}
    if method != 'reference':
        raise ValueError('Calibration method must be belt_motion, manual, or reference.')
    _positive_number('Measured reference length (cm)', reference_length_cm)
    segments = {'ankle_toe': ('ankle', 'toe'), 'hip_knee': ('hip', 'knee'),
                'knee_ankle': ('knee', 'ankle'), 'ankle_mtp': ('ankle', 'mtp')}
    if reference_segment not in segments:
        raise ValueError('Unknown reference segment. Choose ' + ', '.join(segments) + '.')
    first, second = segments[reference_segment]
    valid = None
    for threshold in (.9, .7, .5):
        mask = (dataframe[f'{first} likelihood'] > threshold) & (dataframe[f'{second} likelihood'] > threshold)
        if int(mask.sum()) >= 10:
            valid = mask
            break
    if valid is None:
        raise ValueError(f'Insufficient confident tracking for reference calibration of {reference_segment}: need at least 10 jointly confident frames (>0.5).')
    distances = np.hypot(dataframe.loc[valid, f'{first} x'] - dataframe.loc[valid, f'{second} x'],
                         dataframe.loc[valid, f'{first} y'] - dataframe.loc[valid, f'{second} y'])
    median = float(np.median(distances))
    if median <= 0:
        raise ValueError('Reference segment has zero pixel length; check the body-part mapping.')
    variation = float(np.std(distances) / median)
    # Retain the existing reference-calibration helper's variability safeguard.
    if variation > .30:
        raise ValueError(f'Reference calibration is unreliable: {reference_segment} length CV is {variation:.1%} (>30%). Use a stable measured segment or measured manual scale; no fallback was applied.')
    return {'method': 'reference', 'pixels_per_cm': median / reference_length_cm,
            'reference_segment': reference_segment, 'reference_length_cm': float(reference_length_cm),
            'reference_median_pixels': median, 'reference_length_cv': variation,
            'reference_confidence_threshold': threshold, 'reference_frames': int(valid.sum())}


def prepare_input(dataframe, frame_rate, cutoff_f, bodypart, cm_speed,
                  right_to_left, limits):
    _positive_number('Frame rate', frame_rate)
    _positive_number('Treadmill speed (cm/s)', cm_speed)
    _positive_number('Lowpass cutoff (Hz)', cutoff_f)
    if cutoff_f >= frame_rate / 2:
        raise ValueError('Lowpass cutoff must be less than half the frame rate (Nyquist frequency).')
    if right_to_left not in (True, False, 'auto', None):
        raise ValueError('Walking direction must be True, False, "auto", or None.')
    for name in ('step_height', 'stride_length'):
        low, high = limits[f'{name}_min_cm'], limits[f'{name}_max_cm']
        if not np.isfinite([low, high]).all() or low < 0 or high < low:
            raise ValueError(f'{name.replace("_", " ").capitalize()} limits must be finite, non-negative, and minimum <= maximum.')
    if not np.isfinite(limits['drag_clearance_cm']) or limits['drag_clearance_cm'] < 0:
        raise ValueError('Drag clearance must be finite and non-negative.')
    count = limits['drag_min_consecutive_frames']
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 1:
        raise ValueError('Consecutive drag frames must be a positive integer.')
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('Treadmill input must be a pandas DataFrame with mapped body-part columns.')
    if dataframe.columns.has_duplicates:
        duplicate = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise ValueError(f'Duplicate mapped columns: {duplicate}. Select one limb in the body-part mapping.')
    required = [f'{bp} {coord}' for bp in BODY_PARTS for coord in ('x', 'y', 'likelihood')]
    required += [f'{bodypart} x', f'{bodypart} likelihood']
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError('Missing required treadmill columns: ' + ', '.join(missing) + '. Check the body-part mapping.')
    # scipy.signal.filtfilt's default padding for this fifth-order filter is 18.
    if len(dataframe) <= 18:
        raise ValueError(f'At least 19 consecutive frames are required for the current lowpass filter; received {len(dataframe)}.')
    required = list(dict.fromkeys(required))
    nonnumeric = [col for col in required if not is_numeric_dtype(dataframe[col])]
    if nonnumeric:
        raise ValueError('Non-numeric tracking columns: ' + ', '.join(nonnumeric) + '.')
    invalid = {col: int((~np.isfinite(dataframe[col].to_numpy(dtype=float))).sum()) for col in required}
    invalid = {col: count for col, count in invalid.items() if count}
    if invalid:
        details = ', '.join(f'{col} ({count} frames)' for col, count in invalid.items())
        raise ValueError('Missing or infinite tracking values: ' + details +
                         '. Repair tracking or analyze a continuous valid segment; gaps are not interpolated automatically.')
    quality = {}
    for bp in BODY_PARTS:
        likelihood = dataframe[f'{bp} likelihood']
        if ((likelihood < 0) | (likelihood > 1)).any():
            raise ValueError(f'{bp} likelihood must be between 0 and 1.')
        quality[bp] = {
            'frames_at_or_below_0_5': int((likelihood <= 0.5).sum()),
            'frames_at_or_below_0_6': int((likelihood <= 0.6).sum()),
            'mean_likelihood': float(likelihood.mean()),
        }
    if int((dataframe[f'{bodypart} likelihood'] > 0.5).sum()) < 3:
        raise ValueError(f'Insufficient confident {bodypart} tracking to estimate treadmill motion: need at least three frames with likelihood > 0.5.')
    report = {
        'report_version': 1, 'analysis_type': 'treadmill', 'frames': len(dataframe),
        'frame_number_basis': 'zero-based CSV row position',
        'detector': 'existing_treadmill_detector', 'drag_run_filter': 'complete_runs_v1',
        'settings': {'frame_rate': float(frame_rate), 'filter_cutoff_hz': float(cutoff_f),
                     'treadmill_speed_cm_s': float(cm_speed), 'calibration_bodypart': bodypart,
                     'requested_right_to_left': right_to_left, **limits},
        'tracking_confidence': quality,
        'warnings': [f'{bp}: {q["frames_at_or_below_0_5"]}/{len(dataframe)} frames have likelihood <= 0.5; affected measurements require review.'
                     for bp, q in quality.items() if q['frames_at_or_below_0_5']],
    }
    # The numerical core modifies coordinates in place. Never pass the caller's
    # frame/index objects into it, including when the same input is analyzed twice.
    working = dataframe.copy(deep=True).reset_index(drop=True)
    working.attrs = {}
    return working, report


def attach_report(result, report):
    parameters, coords, _, _, _, _ = result
    report['calibration'] = coords.attrs.pop('_treadmill_calibration', {})
    report['intervals'] = coords.attrs.pop('_treadmill_intervals', [])
    for interval in report['intervals']:
        span = coords.iloc[interval['start_frame']:interval['end_frame'] + 1]
        interval['low_confidence_frames_by_joint'] = {
            bp: int((span[f'{bp} likelihood'] <= 0.5).sum()) for bp in BODY_PARTS
            if (span[f'{bp} likelihood'] <= 0.5).any()
        }
    durations = pd.to_numeric(parameters['cycle duration (s)'], errors='coerce')
    report['candidate_intervals'] = len(parameters)
    report['retained_intervals'] = int(durations.notna().sum())
    report['rejected_intervals'] = len(parameters) - report['retained_intervals']
    if report['retained_intervals'] == 0:
        report['status'] = 'no_usable_strides'
        report['warnings'].append('No usable stride measurements were retained. Inspect tracking, direction, calibration, and stride limits.')
    else:
        report['status'] = 'completed_with_warnings' if report['warnings'] else 'completed'
    boundary = sum(item['ends_at_recording_boundary'] for item in report['intervals'])
    if boundary:
        report['warnings'].append('The last candidate interval reaches the recording boundary; its completeness requires review.')
        if report['status'] == 'completed':
            report['status'] = 'completed_with_warnings'
    joint_columns = [col for col in parameters if ' joint ' in col]
    unavailable = {col: int(pd.to_numeric(parameters[col], errors='coerce')[durations.notna()].isna().sum())
                   for col in joint_columns}
    report['unavailable_retained_joint_measurements'] = {col: n for col, n in unavailable.items() if n}
    if report['unavailable_retained_joint_measurements']:
        report['warnings'].append('Some retained joint measurements are unavailable because joint geometry is degenerate.')
        if report['status'] == 'completed':
            report['status'] = 'completed_with_warnings'
    parameters.attrs['analysis_report'] = report
    return result
