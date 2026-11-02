"""Reconstruction accuracy, bounded failures and independent timing behavior."""
import json
import os
import unittest

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from Functions import KinematicsFunctions as k
from Functions.JointInterpolation import reconstruct_short_gaps, interpolation_context
from Panels.AnalyzeStride import AnalyzeStridePanel
from runway_fixtures import synthetic_walk
from tools.audit_joint_interpolation import articulated_motion, audit, summarize


class JointInterpolation(unittest.TestCase):
    def analyze(self, data, **settings):
        return k.extract_spontaneous_parameters(100., data, 6., pixels_per_cm=50.,
            joint_interpolation='linear', **settings)

    def test_bounded_gaps_fill_without_extrapolating_or_crossing_long_gaps(self):
        raw = articulated_motion()
        masked = {bp: xy.copy() for bp, xy in raw.items()}
        masked['knee'][:2] = np.nan
        masked['knee'][100:104] = np.nan
        masked['knee'][300:305] = np.nan
        masked['knee'][-2:] = np.nan
        before = {bp: xy.copy() for bp, xy in masked.items()}
        for method in ('linear', 'pchip'):
            result, flags, report = reconstruct_short_gaps(masked, 4, method)
            self.assertTrue(flags['knee'][100:104].all())
            self.assertTrue(np.isnan(result['knee'][300:305]).all())
            self.assertTrue(np.isnan(result['knee'][:2]).all())
            self.assertTrue(np.isnan(result['knee'][-2:]).all())
            self.assertEqual(int(flags['knee'].sum()), 4)
            reasons = [reason for gap in report['knee']['rejected_gaps'] for reason in gap['rejection_reasons']]
            self.assertIn('gap_exceeds_time_limit', reasons)
            self.assertIn('recording_edge_no_extrapolation', reasons)
            for bp in masked: np.testing.assert_array_equal(masked[bp], before[bp])

    def test_local_geometry_rejects_a_collapsed_segment(self):
        raw = articulated_motion()
        start, end = 100, 103
        raw['knee'][start-1] = raw['hip'][start-1] + [-36, 0]
        raw['knee'][end] = raw['hip'][end] + [36, 0]
        raw['knee'][start:end] = np.nan
        result, flags, report = reconstruct_short_gaps(raw, 4)
        self.assertFalse(flags['knee'][start:end].any())
        self.assertTrue(np.isnan(result['knee'][start:end]).all())
        self.assertIn('local_segment_length_deviation', report['knee']['rejected_gaps'][0]['rejection_reasons'])

    def test_geometry_checks_require_observed_adjacent_joints(self):
        raw = articulated_motion()
        for bp in ('knee', 'hip', 'ankle'): raw[bp][100:102] = np.nan
        _, flags, report = reconstruct_short_gaps(raw, 4)
        self.assertFalse(flags['knee'][100:102].any())
        self.assertIn('no_observed_adjacent_joint_for_geometry_check', report['knee']['rejected_gaps'][0]['rejection_reasons'])

    def test_motion_guard_rejects_a_jump_hidden_inside_a_gap(self):
        time = np.arange(200)
        offset = np.interp(time, [99, 103], [0., 100.])
        raw = {bp: np.column_stack([np.full(200, 20.*i)+offset, np.full(200, 15.*i)])
               for i, bp in enumerate(('toe','mtp','ankle','knee','hip','iliac crest'))}
        raw['knee'][100:103] = np.nan
        _, flags, report = reconstruct_short_gaps(raw, 4)
        self.assertFalse(flags['knee'][100:103].any())
        reasons = report['knee']['rejected_gaps'][0]['rejection_reasons']
        self.assertIn('local_motion_jump', reasons)
        self.assertNotIn('local_segment_length_deviation', reasons)

    def test_spatial_scaling_and_mirroring_preserve_guard_decisions(self):
        raw = articulated_motion()
        raw['knee'][100:104] = np.nan
        for method in ('linear', 'pchip'):
            a, flags, _ = reconstruct_short_gaps(raw, 4, method)
            transformed = {bp: xy*np.array([-2., 2.]) for bp, xy in raw.items()}
            b, other_flags, _ = reconstruct_short_gaps(transformed, 4, method)
            for bp in raw:
                np.testing.assert_array_equal(flags[bp], other_flags[bp])
                np.testing.assert_allclose(a[bp]*[-2, 2], b[bp], equal_nan=True)

    def test_filter_context_includes_observed_samples_around_reconstruction(self):
        xy = np.ones((100, 2)); xy[60:70] = np.nan
        filled = np.zeros(100, dtype=bool); filled[20] = True
        context = interpolation_context(xy, filled)
        self.assertTrue(context[:60].all())
        self.assertFalse(context[60:].any())

    def test_held_out_known_motion_limits_coordinate_and_peak_errors(self):
        rows = audit(articulated_motion(), 240., trials=3)
        summary = summarize(rows)
        for method, result in summary.items():
            self.assertGreater(result['accepted_windows'], .95*result['attempted_windows'])
            self.assertLess(result['errors']['coordinate_max_error_pixels']['maximum'], .5)
            self.assertLess(result['errors']['angle_max_error_deg']['maximum'], .5)
            self.assertLess(result['errors']['amplitude_error_deg']['maximum'], .5)
        self.assertEqual(summary['linear']['attempted_windows'], summary['pchip']['attempted_windows'])

    def test_opt_in_with_no_gaps_preserves_all_parameter_values(self):
        data = synthetic_walk()
        observed = k.extract_spontaneous_parameters(100., data, 6., pixels_per_cm=50.)
        reconstructed = self.analyze(data)
        pd.testing.assert_frame_equal(observed[0], reconstructed[0], check_exact=True)
        np.testing.assert_array_equal(observed[2], reconstructed[2])
        np.testing.assert_array_equal(observed[5], reconstructed[5])

    def test_short_joint_gap_recovers_angles_without_changing_phase_events(self):
        data = synthetic_walk(); data.loc[490, ['knee x', 'knee y']] = np.nan
        before = data.copy(deep=True)
        observed = k.extract_spontaneous_parameters(100., data, 6., pixels_per_cm=50.)
        result = self.analyze(data)
        np.testing.assert_array_equal(result[2], observed[2])
        np.testing.assert_array_equal(result[5], observed[5])
        affected = observed[0]['knee joint extension (deg)'].isna()
        self.assertTrue(affected.any())
        self.assertTrue(result[0].loc[affected, 'knee joint extension (deg)'].notna().all())
        self.assertTrue(result[1].loc[490, 'knee metric interpolated'])
        self.assertTrue(np.isnan(result[1].loc[490, 'knee raw x']))
        self.assertTrue(np.isnan(result[1].loc[490, 'knee observed filtered x']))
        self.assertEqual(result[1].loc[490, 'knee likelihood'], 1.)
        pd.testing.assert_frame_equal(data, before, check_exact=True)
        report = result[0].attrs['analysis_report']
        interval = next(r for r in report['intervals'] if r['start_frame'] <= 490 < r['end_frame'])
        quality = interval['metric_quality']['knee_angle']
        self.assertEqual(quality['status'], 'estimated')
        self.assertEqual(quality['interpolated_frames'], 1)
        self.assertIsInstance(quality['extension_at_interpolated_frame'], bool)
        self.assertIsNone(interval['observed_only_metrics']['knee joint extension (deg)'])
        json.dumps(report, allow_nan=False)

    def test_per_metric_fraction_limit_keeps_timing_and_unrelated_angles(self):
        data = synthetic_walk(); data.loc[490, 'knee likelihood'] = .1
        result = self.analyze(data, joint_max_interpolated_fraction=0.)
        intervals = result[0].attrs['analysis_report']['intervals']
        index = next(i for i, row in enumerate(intervals) if row['start_frame'] <= 490 < row['end_frame'])
        self.assertTrue(np.isnan(result[0].loc[index, 'knee joint extension (deg)']))
        self.assertTrue(np.isfinite(result[0].loc[index, 'mtp joint extension (deg)']))
        self.assertTrue(np.isfinite(result[0].loc[index, 'cycle duration (s)']))
        self.assertIn('fraction exceeds', intervals[index]['unavailable_metrics']['knee joint extension (deg)'])

    def test_long_joint_gaps_leave_dependent_metrics_unavailable(self):
        data = synthetic_walk(); data.loc[480:490, 'knee likelihood'] = .1
        result = self.analyze(data)
        self.assertFalse(result[1].loc[480:490, 'knee metric interpolated'].any())
        self.assertTrue(result[0]['knee joint extension (deg)'].isna().any())
        self.assertTrue(result[0]['cycle duration (s)'].notna().all())

    def test_invalid_joint_settings_are_explanatory(self):
        for settings in [{'joint_interpolation': 'cubic'}, {'joint_max_gap_s': None},
                         {'joint_max_gap_s': 1+2j},
                         {'joint_max_interpolated_fraction': 1.1}, {'joint_max_segment_change': -1}]:
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, 'Joint'):
                k.extract_spontaneous_parameters(100., synthetic_walk(), 6., pixels_per_cm=50., **settings)

    def test_gui_requires_explicit_joint_reconstruction_selection(self):
        app = QApplication.instance() or QApplication([])
        panel = AnalyzeStridePanel(None)
        panel.spontaneous_radio.click()
        self.assertEqual(panel.joint_interpolation_input.currentData(), 'none')
        self.assertFalse(panel.joint_gap_input.isEnabled())
        panel.joint_interpolation_input.setCurrentIndex(panel.joint_interpolation_input.findData('pchip'))
        self.assertTrue(panel.joint_gap_input.isEnabled())
        panel.joint_gap_input.setValue(15.)
        panel.joint_fraction_input.setValue(5.)
        panel.collect_analysis_settings()
        self.assertEqual(panel.analysis_settings['joint_interpolation'], 'pchip')
        self.assertEqual(panel.analysis_settings['joint_max_gap_s'], .015)
        self.assertEqual(panel.analysis_settings['joint_max_interpolated_fraction'], .05)
        panel.close()


if __name__ == '__main__':
    unittest.main()
