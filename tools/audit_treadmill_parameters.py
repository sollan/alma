"""One-control-at-a-time audit through ALMA's actual GUI analysis worker.

Values are sensitivity experiments, not recommended recording settings.
Use --implementation-root with an archived Functions/ and Panels/ snapshot
to reproduce earlier implementations in a separate process.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
import pandas as pd


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cases():
    return [
        ('baseline', {}),
        ('fps_60', {'frame_rate': 60.}),
        ('fps_119_88', {'frame_rate': 119.88}),
        ('fps_240', {'frame_rate': 240.}),
        ('belt_speed_20', {'cm_speed': 20.}),
        ('belt_speed_40', {'cm_speed': 40.}),
        ('direction_auto', {'right_to_left': 'auto'}),
        ('direction_right_to_left', {'right_to_left': True}),
        ('cutoff_3', {'filter_cutoff': 3.}),
        ('cutoff_10', {'filter_cutoff': 10.}),
        ('minimum_step_0_3', {'step_height_min_cm': .3}),
        ('minimum_step_0_5', {'step_height_min_cm': .5}),
        ('maximum_step_0_3', {'step_height_max_cm': .3}),
        ('maximum_step_0_5', {'step_height_max_cm': .5}),
        ('minimum_stride_5', {'stride_length_min_cm': 5.}),
        ('minimum_stride_7', {'stride_length_min_cm': 7.}),
        ('maximum_stride_6', {'stride_length_max_cm': 6.}),
        ('maximum_stride_10', {'stride_length_max_cm': 10.}),
        ('clearance_0_05', {'drag_clearance_cm': .05}),
        ('clearance_0_2', {'drag_clearance_cm': .2}),
        ('minimum_drag_1', {'drag_min_consecutive_frames': 1}),
        ('minimum_drag_8', {'drag_min_consecutive_frames': 8}),
        ('manual_matches_belt', {'calibration_method': 'manual', 'pixels_per_cm': 64.81955115441923}),
        ('manual_50', {'calibration_method': 'manual', 'pixels_per_cm': 50.}),
        ('manual_100', {'calibration_method': 'manual', 'pixels_per_cm': 100.}),
        ('manual_100_speed_20', {'calibration_method': 'manual', 'pixels_per_cm': 100., 'cm_speed': 20.}),
        ('manual_100_speed_40', {'calibration_method': 'manual', 'pixels_per_cm': 100., 'cm_speed': 40.}),
        ('reference_ankle_toe_1_5', {'calibration_method': 'reference', 'reference_segment': 'ankle_toe', 'reference_length_cm': 1.5}),
        ('reference_ankle_toe_1', {'calibration_method': 'reference', 'reference_segment': 'ankle_toe', 'reference_length_cm': 1.}),
        ('reference_ankle_toe_2', {'calibration_method': 'reference', 'reference_segment': 'ankle_toe', 'reference_length_cm': 2.}),
        ('reference_ankle_mtp_0_8', {'calibration_method': 'reference', 'reference_segment': 'ankle_mtp', 'reference_length_cm': .8}),
    ]


def aligned_retained(params):
    numeric = params.apply(pd.to_numeric, errors='coerce')
    return numeric[numeric['cycle duration (s)'].notna()].set_index(
        ['stride_start (frame)', 'stride_end (frame)'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--implementation-root', type=Path, default=ROOT)
    parser.add_argument('--select', nargs='+', help='Run only named cases plus baseline.')
    parser.add_argument('--extend', action='store_true', help='Merge selected cases into an existing report.')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    impl = args.implementation_root.resolve()
    load('Functions.TreadmillValidation', impl/'Functions/TreadmillValidation.py')
    k = load('audit_kinematics', impl/'Functions/KinematicsFunctions.py')
    panel = load('audit_panel', impl/'Panels/AnalyzeStride.py')
    panel.KinematicsFunctions = k
    demo = ROOT/'Resources/Demo_Mouse_Treadmill_30cm_s_650000_filtered.csv'
    base = dict(frame_rate=120., cm_speed=30., filter_cutoff=6., right_to_left=False,
        calibration_method='belt_motion', pixels_per_cm=None, reference_segment=None,
        reference_length_cm=None, step_height_min_cm=0., step_height_max_cm=2.,
        stride_length_min_cm=0., stride_length_max_cm=8., drag_clearance_cm=.1,
        drag_min_consecutive_frames=4)
    real_extract = k.extract_parameters
    captured = []
    def capture(*a, **kw):
        result = real_extract(*a, **kw)
        captured.append(result)
        return result
    k.extract_parameters = capture
    # Plotting occurs after the extraction and is irrelevant to this audit.
    k.return_continuous = lambda *a, **kw: None
    records = []
    baseline = None
    for name, overrides in cases():
        if args.select and name != 'baseline' and name not in args.select:
            continue
        folder = out/name
        folder.mkdir(exist_ok=True)
        settings = {**base, **overrides}
        parent = SimpleNamespace(selected_files=[str(demo)], custom_bodypart_mapping=None,
            analysis_type='Treadmill', output_folder=str(folder), analysis_settings=settings)
        captured.clear()
        messages = []
        worker = panel.AnalysisThread(parent)
        worker.analysis_completed.connect(lambda ok, msg: messages.append([ok, msg]))
        log = io.StringIO()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            worker.run()
        (folder/'worker.log').write_text(log.getvalue())
        rec = dict(case=name, overrides=overrides, settings=settings, worker_result=messages)
        if not captured:
            rec.update(status='error', error=messages[-1][1] if messages else 'No worker result')
        else:
            result = captured[-1]
            params = result[0]
            quality = params.attrs['analysis_report']
            retained = aligned_retained(params)
            starts = np.asarray(result[5], dtype=int)
            phases = np.asarray(result[2], dtype=int)
            np.savez(folder/'events.npz', starts=starts, stance=phases)
            numeric = params.apply(pd.to_numeric, errors='coerce')
            means = {c: float(v) if np.isfinite(v) else None
                     for c, v in numeric[numeric['cycle duration (s)'].notna()].mean().items()}
            if baseline is None:
                baseline = dict(params=params.copy(), retained=retained.copy(),
                                starts=starts.copy(), phases=phases.copy())
            common = baseline['retained'].index.intersection(retained.index)
            a, b = baseline['retained'].loc[common], retained.loc[common]
            changed_columns = []
            max_changes = {}
            for c in b:
                same = np.isclose(a[c], b[c], rtol=0, atol=1e-10, equal_nan=True)
                if not same.all():
                    changed_columns.append(c)
                    delta = (a[c]-b[c]).abs().max()
                    max_changes[c] = float(delta) if np.isfinite(delta) else None
            phase_changes = int(np.count_nonzero(phases != baseline['phases'])) if phases.shape == baseline['phases'].shape else None
            onset_changes = 0
            for idx in common:
                old_onset = idx[0] + round(a.loc[idx, 'stance duration (s)']*base['frame_rate'])
                new_onset = idx[0] + round(b.loc[idx, 'stance duration (s)']*settings['frame_rate'])
                onset_changes += old_onset != new_onset
            rec.update(status=quality['status'], candidate_intervals=len(params),
                retained_intervals=len(retained), calibration=quality['calibration'],
                starts_added=sorted(set(starts.tolist())-set(baseline['starts'].tolist())),
                starts_removed=sorted(set(baseline['starts'].tolist())-set(starts.tolist())),
                detector_stance_frames_changed=phase_changes,
                common_retained_intervals=len(common),
                retained_intervals_added=len(retained.index.difference(baseline['retained'].index)),
                retained_intervals_removed=len(baseline['retained'].index.difference(retained.index)),
                swing_onsets_changed_common_intervals=int(onset_changes),
                changed_columns_common_intervals=changed_columns,
                maximum_absolute_changes_common_intervals=max_changes, means=means)
        records.append(rec)
        print(name, rec['status'], rec.get('retained_intervals'), rec.get('calibration',{}).get('pixels_per_cm'), flush=True)
    if args.extend and (out/'audit_results.json').exists():
        existing = json.loads((out/'audit_results.json').read_text())
        merged = {record['case']: record for record in existing['cases']}
        merged.update({record['case']: record for record in records})
        records = list(merged.values())
    report = dict(demo=str(demo), demo_sha256=hashlib.sha256(demo.read_bytes()).hexdigest(),
        implementation_root=str(impl),
        implementation_sha256=hashlib.sha256((impl/'Functions/KinematicsFunctions.py').read_bytes()).hexdigest(),
        component_sha256={str(path): hashlib.sha256((impl/path).read_bytes()).hexdigest()
                          for path in ['Functions/KinematicsFunctions.py','Functions/TreadmillValidation.py','Panels/AnalyzeStride.py']},
        baseline_settings=base,
        caution='Sensitivity experiments, not optimal settings or validated calibration. Aggregate means change when the retained cohort changes; common-interval comparisons control for that.',
        cases=records)
    (out/'audit_results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    summary=[]
    for rec in records:
        summary.append(dict(case=rec['case'], status=rec['status'], overrides=json.dumps(rec['overrides']),
            retained=rec.get('retained_intervals'), candidates=rec.get('candidate_intervals'),
            pixels_per_cm=rec.get('calibration',{}).get('pixels_per_cm'),
            starts_added=len(rec.get('starts_added',[])),starts_removed=len(rec.get('starts_removed',[])),
            detector_stance_frames_changed=rec.get('detector_stance_frames_changed'),
            common_retained_intervals=rec.get('common_retained_intervals'),
            swing_onsets_changed=rec.get('swing_onsets_changed_common_intervals'),
            mean_stride_cm=rec.get('means',{}).get('stride length (cm)'),
            mean_cycle_s=rec.get('means',{}).get('cycle duration (s)'),
            mean_step_cm=rec.get('means',{}).get('step height (cm)'),
            mean_drag_s=rec.get('means',{}).get('drag duration (s)'),
            changed_columns='; '.join(rec.get('changed_columns_common_intervals',[]))))
    pd.DataFrame(summary).to_csv(out/'parameter_summary.csv',index=False)


if __name__ == '__main__':
    main()
