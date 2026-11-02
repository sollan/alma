"""Bounded coordinate reconstruction with local observed-geometry checks.

Reconstructed samples retain separate masks; likelihoods are never replaced.
These checks limit obvious artifacts, but do not establish anatomical truth.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from Functions.TreadmillValidation import BODY_PARTS


def reconstruct_short_gaps(observed, max_frames, method='linear', max_segment_change=.35,
                           context_frames=24):
    """Fill short internal gaps using reliable endpoints and adjacent observations.

    ``observed`` contains confidence-masked (N, 2) arrays. Geometry is checked
    against adjacent joints' *observed* coordinates, never another proposal.
    Segment lengths are compared with their local endpoint values, without
    assuming a species-specific anatomical length. Long gaps are untouched.
    """
    if method not in ('linear', 'pchip'):
        raise ValueError('Joint interpolation method must be linear or pchip.')
    output = {bp: np.asarray(xy, dtype=float).copy() for bp, xy in observed.items()}
    masks = {bp: np.zeros(len(xy), dtype=bool) for bp, xy in observed.items()}
    reports = {bp: {'filled_gaps': [], 'rejected_gaps': []} for bp in observed}
    for bp in BODY_PARTS:
        values = np.asarray(observed[bp], dtype=float)
        valid = np.isfinite(values).all(axis=1)
        transitions = np.diff(np.r_[False, ~valid, False].astype(np.int8))
        gaps = zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))
        index = BODY_PARTS.index(bp)
        neighbors = list(BODY_PARTS[max(0, index-1):index]) + list(BODY_PARTS[index+1:index+2])
        for start, end in gaps:
            start, end = int(start), int(end)
            item = {'start_frame': start, 'end_frame_exclusive': end, 'frames': end-start}
            reasons = []
            if start == 0 or end == len(values):
                reasons.append('recording_edge_no_extrapolation')
            elif end-start > max_frames:
                reasons.append('gap_exceeds_time_limit')
            if reasons:
                item['rejection_reasons'] = reasons
                reports[bp]['rejected_gaps'].append(item)
                continue
            query = np.arange(start, end)
            if method == 'linear':
                estimate = np.column_stack([np.interp(query, [start-1, end], values[[start-1, end], axis])
                                            for axis in (0, 1)])
            else:
                anchors = [start-1, end]
                if start > 1 and valid[start-2]: anchors.insert(0, start-2)
                if end+1 < len(values) and valid[end+1]: anchors.append(end+1)
                estimate = PchipInterpolator(anchors, values[anchors], axis=0, extrapolate=False)(query)
            checked = []
            for neighbor in neighbors:
                other = np.asarray(observed[neighbor], dtype=float)
                endpoints = other[[start-1, end]]
                usable = np.isfinite(other[start:end]).all(axis=1)
                if not np.isfinite(endpoints).all() or not usable.any():
                    continue
                endpoint_lengths = np.linalg.norm(values[[start-1, end]]-endpoints, axis=1)
                if np.min(endpoint_lengths) <= 1e-12:
                    reasons.append('degenerate_observed_segment')
                    continue
                expected = np.interp(query, [start-1, end], endpoint_lengths)
                actual = np.linalg.norm(estimate-other[start:end], axis=1)
                deviation = np.abs(actual[usable]-expected[usable])/expected[usable]
                maximum = float(np.max(deviation))
                checked.append({'neighbor': neighbor, 'checked_frames': int(usable.sum()),
                                'max_relative_length_deviation': maximum})
                if maximum > max_segment_change:
                    reasons.append('local_segment_length_deviation')
                if np.ptp(endpoint_lengths)/np.mean(endpoint_lengths) > 2*max_segment_change:
                    reasons.append('inconsistent_endpoint_segment_lengths')
            if not checked:
                reasons.append('no_observed_adjacent_joint_for_geometry_check')
            # Compare proposed motion with observed frame-to-frame motion nearby.
            lo, hi = max(0, start-context_frames), min(len(values), end+context_frames+1)
            steps = np.linalg.norm(np.diff(values[lo:hi], axis=0), axis=1)
            steps = steps[np.isfinite(steps)]
            item['speed_guard_available'] = len(steps) >= 6
            if len(steps) >= 6:
                median = np.median(steps)
                mad = 1.4826*np.median(np.abs(steps-median))
                limit = max(float(3*np.quantile(steps, .95)), float(median+6*mad), 1e-12)
                proposed_steps = np.linalg.norm(np.diff(np.vstack([values[start-1], estimate, values[end]]), axis=0), axis=1)
                item['max_proposed_step_pixels'] = float(np.max(proposed_steps))
                item['local_step_limit_pixels'] = limit
                if np.max(proposed_steps) > limit:
                    reasons.append('local_motion_jump')
            item['geometry_checks'] = checked
            if reasons:
                item['rejection_reasons'] = sorted(set(reasons))
                reports[bp]['rejected_gaps'].append(item)
            else:
                output[bp][start:end] = estimate
                masks[bp][start:end] = True
                reports[bp]['filled_gaps'].append(item)
    return output, masks, reports


def interpolation_context(coordinates, filled):
    """Flag whole finite filter runs touched by reconstruction, including context.

    Forward/backward filtering can alter observed samples around a filled gap.
    A direct-fill mask alone cannot describe that influence.
    """
    finite = np.isfinite(coordinates).all(axis=1)
    changes = np.diff(np.r_[False, finite, False].astype(np.int8))
    context = np.zeros(len(finite), dtype=bool)
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if filled[start:end].any(): context[start:end] = True
    return context
