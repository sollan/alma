"""Tracking visibility must stay independent of metric/filter availability."""
import os
import unittest

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from Functions import KinematicsFunctions as k
from Functions.SkeletonVisualization import skeleton_coordinates, OBSERVED, LOW_CONFIDENCE, RECONSTRUCTED, MISSING
from Panels.AnalyzeStride import AnalyzeStridePanel
from runway_fixtures import synthetic_walk


class SkeletonReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def analyze(self, data, **options):
        return k.extract_spontaneous_parameters(100., data, 6., pixels_per_cm=50., **options)

    def test_confident_short_segments_remain_visible_in_tracked_view(self):
        data = synthetic_walk()
        data.loc[480:510, 'knee likelihood'] = .1
        data.loc[490:494, 'knee likelihood'] = 1.
        result = self.analyze(data)
        coords = result[1]; before = coords.copy(deep=True)
        raw, raw_states = skeleton_coordinates(coords, 'tracked')
        filtered, filtered_states = skeleton_coordinates(coords, 'analysis')
        self.assertTrue(np.isnan(filtered[490:495,3]).all())
        np.testing.assert_array_equal(raw[490:495,3],data.loc[490:494,['knee x','knee y']])
        self.assertTrue((raw_states[490:495,3] == OBSERVED).all())
        self.assertTrue((filtered_states[490:495,3] == MISSING).all())
        pd.testing.assert_frame_equal(coords,before,check_exact=True)

    def test_raw_low_confidence_is_visible_but_distinct_from_reconstruction(self):
        data = synthetic_walk(); data.loc[490,'knee likelihood'] = .1
        result = self.analyze(data,joint_interpolation='linear')
        raw, raw_states = skeleton_coordinates(result[1],'tracked')
        filtered, filtered_states = skeleton_coordinates(result[1],'analysis')
        self.assertTrue(np.isfinite(raw[490,3]).all())
        self.assertEqual(raw_states[490,3],LOW_CONFIDENCE)
        self.assertEqual(filtered_states[490,3],RECONSTRUCTED)
        self.assertTrue(np.isfinite(filtered[490,3]).all())

    def test_tracked_view_never_fills_a_missing_coordinate(self):
        data = synthetic_walk(); data.loc[490,['knee x','knee y']] = np.nan
        result = self.analyze(data,joint_interpolation='linear')
        raw, states = skeleton_coordinates(result[1],'tracked')
        self.assertTrue(np.isnan(raw[490,3]).all())
        self.assertEqual(states[490,3],MISSING)
        _, other = skeleton_coordinates(result[1],'analysis')
        self.assertEqual(other[490,3],RECONSTRUCTED)

    def test_joint_cutoff_inherits_main_filter_without_numeric_changes(self):
        data = synthetic_walk()
        before = self.analyze(data)
        after = self.analyze(data,joint_cutoff_f=6.)
        pd.testing.assert_frame_equal(before[0],after[0],check_exact=True)
        np.testing.assert_array_equal(before[2],after[2])
        self.assertTrue(before[0].attrs['analysis_report']['settings']['joint_filter_inherits_main_cutoff'])

    def test_independent_joint_cutoff_changes_angles_without_moving_stride_events(self):
        data = synthetic_walk()
        a = self.analyze(data); b = self.analyze(data,joint_cutoff_f=12.)
        fixed = [c for c in a[0] if ' joint ' not in c and 'toe-to-crest' not in c]
        pd.testing.assert_frame_equal(a[0][fixed],b[0][fixed],check_exact=True)
        np.testing.assert_array_equal(a[2],b[2]); np.testing.assert_array_equal(a[5],b[5])
        self.assertFalse(np.allclose(a[0]['knee joint extension (deg)'],b[0]['knee joint extension (deg)'],rtol=1e-8))
        for bp in ('toe','mtp','ankle','knee','hip','iliac crest'):
            pd.testing.assert_frame_equal(a[1][[f'{bp} raw x',f'{bp} raw y']],b[1][[f'{bp} raw x',f'{bp} raw y']],check_exact=True)

    def test_invalid_joint_cutoff_is_not_accepted(self):
        for value in (0.,-1.,np.nan,50.,100.):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'Joint filter cutoff'):
                self.analyze(synthetic_walk(),joint_cutoff_f=value)

    def test_gui_separates_plot_source_and_joint_filter_from_detection(self):
        app = self.app
        panel = AnalyzeStridePanel(None)
        panel.spontaneous_radio.click(); panel.pixels_per_cm_input.setValue(50.)
        panel.wizard_step=3
        self.assertFalse(panel.joint_cutoff_checkbox.isChecked())
        self.assertEqual(panel.stickplot_source_input.currentData(),'tracked')
        panel.joint_cutoff_checkbox.setChecked(True); panel.joint_cutoff_input.setValue(10.)
        panel.collect_analysis_settings()
        self.assertEqual(panel.analysis_settings['joint_cutoff_f'],10.)
        self.assertEqual(panel.analysis_settings['stickplot_coordinate_source'],'tracked')
        self.assertEqual(panel.analysis_settings['filter_cutoff'],6.)
        panel.joint_cutoff_input.setValue(panel.frame_rate_input.value()/2)
        self.assertFalse(panel.next_button.isEnabled())
        self.assertIn('Joint filter cutoff',panel.settings_status_label.text())
        panel.close()
        panel.deleteLater()
        app.processEvents()


if __name__=='__main__':
    unittest.main()
