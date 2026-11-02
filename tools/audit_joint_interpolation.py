"""Mask known coordinates to measure short-gap reconstruction errors.

The built-in articulated trajectories have known positions. CSV references are
tracked observations, not independent anatomical ground truth. Both methods
receive the same held-out windows; rejected reconstructions are counted.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from Functions.JointInterpolation import reconstruct_short_gaps
from Functions.RunwayAnalysis import ANGLE_PARTS, _angles, filter_segments, runs
from Functions.TreadmillValidation import BODY_PARTS, _positive_number


def articulated_motion(fps=240., seconds=4.):
    time = np.arange(int(seconds*fps))/fps
    phase = 2*np.pi*time/.4
    hip = np.column_stack([100*time, 60+2*np.sin(phase)])
    crest = hip + [-15, -22]
    thigh = 1.2+.25*np.sin(phase)
    shank = thigh + 1.05+.35*np.sin(phase+.65)
    foot = .1+.3*np.sin(phase+1.2)
    knee = hip + 36*np.column_stack([np.cos(thigh), np.sin(thigh)])
    ankle = knee + 34*np.column_stack([np.cos(shank), np.sin(shank)])
    mtp = ankle + 17*np.column_stack([np.cos(foot), np.sin(foot)])
    toe = mtp + 12*np.column_stack([np.cos(foot+.25), np.sin(foot+.25)])
    return dict(zip(BODY_PARTS, [toe, mtp, ankle, knee, hip, crest]))


def audit(observed, fps, max_gap_s=.02, trials=40, seed=7, max_segment_change=.35):
    rng = np.random.default_rng(seed)
    maximum = int(np.floor(max_gap_s*fps+1e-9))
    reference = {bp: np.column_stack([filter_segments(xy[:, a], fps, 6.) for a in (0, 1)])
                 for bp, xy in observed.items()}
    angles = {joint: _angles(reference, *parts) for joint, parts in ANGLE_PARTS.items()}
    rows = []
    margin = max(20, int(round(.15*fps)))
    for bp in BODY_PARTS:
        valid = np.isfinite(observed[bp]).all(axis=1)
        for gap in range(1, maximum+1):
            choices = [s for lo, hi in runs(valid) for s in range(lo+margin, hi-margin-gap+1)]
            if not choices: continue
            starts = rng.choice(choices, size=min(trials, len(choices)), replace=False)
            for start in starts:
                start, end = int(start), int(start+gap)
                # Isolate a fully observed window of the held-out joint. Filling
                # unrelated pre-existing gaps would confound the error estimate.
                lo, hi = start-margin, end+margin
                masked = {b: xy[lo:hi].copy() for b, xy in observed.items()}
                masked[bp][margin:margin+gap] = np.nan
                for method in ('linear', 'pchip'):
                    recovered, flags, _ = reconstruct_short_gaps(masked, maximum, method,
                        max_segment_change, context_frames=max(6, int(round(.1*fps))))
                    accepted = bool(flags[bp][margin:margin+gap].all())
                    row = dict(body_part=bp, gap_frames=gap, start_frame=start, method=method,
                               accepted=accepted, coordinate_rmse_pixels=np.nan,
                               coordinate_max_error_pixels=np.nan, angle_max_error_deg=np.nan,
                               extension_error_deg=np.nan, flexion_error_deg=np.nan, amplitude_error_deg=np.nan)
                    if accepted:
                        estimate = recovered[bp][margin:margin+gap]
                        error = np.linalg.norm(estimate-observed[bp][start:end], axis=1)
                        row.update(coordinate_rmse_pixels=float(np.sqrt(np.mean(error**2))),
                                   coordinate_max_error_pixels=float(np.max(error)))
                        selected = reference.copy()
                        restored = observed[bp].copy()
                        restored[start:end] = estimate
                        selected[bp] = np.column_stack([filter_segments(restored[:, a], fps, 6.) for a in (0, 1)])
                        local = slice(start-margin, end+margin)
                        angle_error, extension, flexion, amplitude = [], [], [], []
                        for joint, parts in ANGLE_PARTS.items():
                            if bp not in parts: continue
                            before, after = angles[joint][local], _angles(selected, *parts)[local]
                            # Full local coverage is required for peak comparisons.
                            if not (np.isfinite(before).all() and np.isfinite(after).all()): continue
                            angle_error.append(float(np.max(np.abs(after-before))))
                            extension.append(float(abs(np.max(after)-np.max(before))))
                            flexion.append(float(abs(np.min(after)-np.min(before))))
                            amplitude.append(float(abs(np.ptp(after)-np.ptp(before))))
                        for key, values in [('angle_max_error_deg', angle_error), ('extension_error_deg', extension),
                                            ('flexion_error_deg', flexion), ('amplitude_error_deg', amplitude)]:
                            if values: row[key] = max(values)
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows):
    result = {}
    for method, group in rows.groupby('method'):
        selected = group[group['accepted']]
        item = {'attempted_windows': len(group), 'accepted_windows': len(selected),
                'rejected_windows': int((~group['accepted']).sum()), 'errors': {}}
        for column in group.columns:
            if not column.endswith(('_pixels', '_deg')): continue
            values = selected[column].dropna()
            item['errors'][column] = {'windows': len(values), 'median': float(values.median()) if len(values) else None,
                'p95': float(values.quantile(.95)) if len(values) else None,
                'maximum': float(values.max()) if len(values) else None}
        result[method] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--fps', type=float, default=240.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=.9)
    parser.add_argument('--max-gap-ms', type=float, default=20.)
    parser.add_argument('--trials', type=int, default=40, help='Windows per body part and gap length.')
    args = parser.parse_args()
    _positive_number('Frame rate', args.fps)
    _positive_number('Maximum gap', args.max_gap_ms)
    _positive_number('Trials', args.trials)
    if args.fps <= 12 or not 0 < args.confidence < 1:
        parser.error('FPS must exceed 12 for the 6 Hz audit filter; confidence must be between zero and one.')
    if args.csv:
        from Functions.KinematicsFunctions import fix_column_names
        with contextlib.redirect_stdout(io.StringIO()):
            data, _, _ = fix_column_names(pd.read_csv(args.csv, header=[1, 2]))
        observed = {}
        for bp in BODY_PARTS:
            names = [f'{bp} {c}' for c in ('x', 'y', 'likelihood')]
            values = data[names].to_numpy(dtype=float) if all(col in data for col in names) else np.full((len(data), 3), np.nan)
            valid = np.isfinite(values).all(axis=1) & (values[:, 2] >= args.confidence)
            observed[bp] = np.where(valid[:, None], values[:, :2], np.nan)
        reference_type = 'confident tracked positions; not anatomical ground truth'
    else:
        observed = articulated_motion(args.fps)
        reference_type = 'generated articulated trajectories with known coordinates'
    rows = audit(observed, args.fps, args.max_gap_ms/1000, args.trials)
    if rows.empty: parser.error('No eligible observed windows at the requested settings.')
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output/'held_out_windows.csv', index=False)
    report = {'reference': reference_type, 'frame_rate': args.fps, 'max_gap_ms': args.max_gap_ms,
              'seed': 7, 'confidence': args.confidence if args.csv else None, 'methods': summarize(rows)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for method, item in report['methods'].items():
        print(method, item['accepted_windows'], '/', item['attempted_windows'], 'windows accepted;',
              'coordinate RMSE p95:', item['errors']['coordinate_rmse_pixels']['p95'])


if __name__ == '__main__':
    main()
