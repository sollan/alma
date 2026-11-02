"""Verify calibration routing and the effects of exposed treadmill controls."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from Functions import KinematicsFunctions as k
from Functions.TreadmillValidation import resolve_calibration
from Panels import AnalyzeStride

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT/'Resources/Demo_Mouse_Treadmill_30cm_s_650000_filtered.csv'
ARGS = dict(cm_speed=30., right_to_left=False, step_height_max_cm=2., drag_clearance_cm=.1)


class TreadmillParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(ROOT)
        cls.app = QApplication.instance() or QApplication([])
        cls.data, _, _ = k.fix_column_names(pd.read_csv(DEMO, header=[1,2]))
        cls.baseline = k.extract_parameters(120., cls.data, 6., 'toe', **ARGS)

    def run_core(self, **kwargs):
        return k.extract_parameters(120., self.data, 6., 'toe', **{**ARGS, **kwargs})

    def test_manual_scale_matching_default_preserves_all_46_columns(self):
        scale = self.baseline[0].attrs['analysis_report']['calibration']['pixels_per_cm']
        result = self.run_core(calibration_method='manual', pixels_per_cm=scale)
        pd.testing.assert_frame_equal(result[0], self.baseline[0], check_exact=True)
        np.testing.assert_array_equal(result[2], self.baseline[2])
        self.assertEqual(result[0].attrs['analysis_report']['calibration']['method'], 'manual')

    def test_manual_scale_changes_units_without_changing_pixel_events(self):
        result = self.run_core(calibration_method='manual', pixels_per_cm=100.)
        scale = self.baseline[0].attrs['analysis_report']['calibration']['pixels_per_cm']
        np.testing.assert_array_equal(result[5], self.baseline[5])
        np.testing.assert_array_equal(result[2], self.baseline[2])
        pd.testing.assert_frame_equal(result[1], self.baseline[1], check_exact=True)
        a = self.baseline[0].set_index(['stride_start (frame)','stride_end (frame)'])
        b = result[0].set_index(['stride_start (frame)','stride_end (frame)'])
        common = a.index[a['cycle duration (s)'].notna()].intersection(b.index[b['cycle duration (s)'].notna()])
        np.testing.assert_allclose(b.loc[common,'stride length (cm)'],a.loc[common,'stride length (cm)']*scale/100.,rtol=1e-12)
        np.testing.assert_array_equal(b.loc[common,'knee joint extension (deg)'],a.loc[common,'knee joint extension (deg)'])

    def test_minimum_step_height_excludes_only_below_limit_candidates(self):
        result = self.run_core(step_height_min_cm=.5)
        expected = self.baseline[0]['cycle duration (s)'].notna() & (self.baseline[0]['step height (cm)']>=.5)
        np.testing.assert_array_equal(result[0]['cycle duration (s)'].notna(),expected)
        np.testing.assert_array_equal(result[5],self.baseline[5])
        rejected = result[0].attrs['analysis_report']['intervals']
        self.assertTrue(any('step_height_below_minimum' in r['rejection_reasons'] for r in rejected))

    def test_recorded_speed_is_crosscheck_when_scale_is_measured(self):
        a = self.run_core(calibration_method='manual',pixels_per_cm=100.,cm_speed=20.)
        b = self.run_core(calibration_method='manual',pixels_per_cm=100.,cm_speed=40.)
        pd.testing.assert_frame_equal(a[0],b[0],check_exact=True)
        first,second = a[0].attrs['analysis_report'],b[0].attrs['analysis_report']
        self.assertEqual(first['settings']['treadmill_speed_cm_s'],20.)
        self.assertEqual(second['settings']['treadmill_speed_cm_s'],40.)
        self.assertEqual(first['calibration']['tracking_implied_belt_speed_cm_s'],
                         second['calibration']['tracking_implied_belt_speed_cm_s'])

    def test_reference_length_has_explicit_inverse_scale(self):
        a = resolve_calibration(self.data,'reference',reference_segment='ankle_toe',reference_length_cm=1.5)
        b = resolve_calibration(self.data,'reference',reference_segment='ankle_toe',reference_length_cm=3.)
        self.assertEqual(a['pixels_per_cm']/2,b['pixels_per_cm'])
        self.assertGreaterEqual(a['reference_frames'],10)
        result = self.run_core(calibration_method='reference',reference_segment='ankle_toe',reference_length_cm=1.5)
        report = result[0].attrs['analysis_report']
        self.assertEqual(report['calibration']['pixels_per_cm'],a['pixels_per_cm'])
        self.assertEqual(report['calibration']['method'],'reference')
        np.testing.assert_array_equal(result[5],self.baseline[5])

    def test_reference_requires_confident_frames_and_has_no_fallback(self):
        data = self.data.copy()
        data['ankle likelihood'] = .1
        with self.assertRaisesRegex(ValueError,'10 jointly confident frames'):
            resolve_calibration(data,'reference',reference_segment='ankle_toe',reference_length_cm=1.5)
        data = self.data.copy()
        data['ankle x'] = data['toe x']
        data['ankle y'] = data['toe y']
        with self.assertRaisesRegex(ValueError,'zero pixel length'):
            resolve_calibration(data,'reference',reference_segment='ankle_toe',reference_length_cm=1.5)
        data['ankle likelihood'] = data['toe likelihood'] = 1.
        data['ankle x'] = data['toe x'] + np.resize([10.,1000.],len(data))
        with self.assertRaisesRegex(ValueError,'unreliable.*CV'):
            resolve_calibration(data,'reference',reference_segment='ankle_toe',reference_length_cm=1.5)

    def test_calibration_errors_are_explicit(self):
        for settings, message in [({'calibration_method':'unknown'},'Calibration method'),
                                  ({'calibration_method':'manual'},'Pixels per cm'),
                                  ({'calibration_method':'manual','pixels_per_cm':-1},'Pixels per cm'),
                                  ({'pixels_per_cm':50},'Select the corresponding'),
                                  ({'calibration_method':'reference','reference_segment':'ankle_toe'},'Measured reference length')]:
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError,message):
                self.run_core(**settings)

    def test_gui_collects_only_the_selected_calibration_inputs(self):
        panel = AnalyzeStride.AnalyzeStridePanel(None)
        panel.collect_analysis_settings()
        self.assertEqual(panel.analysis_settings['calibration_method'],'belt_motion')
        self.assertIsNone(panel.analysis_settings['pixels_per_cm'])
        panel.manual_radio.click()
        panel.pixels_per_cm_input.setValue(100.)
        panel.collect_analysis_settings()
        self.assertEqual(panel.analysis_settings['calibration_method'],'manual')
        self.assertEqual(panel.analysis_settings['pixels_per_cm'],100.)
        self.assertIsNone(panel.analysis_settings['reference_segment'])
        panel.reference_radio.click()
        panel.reference_length_input.setValue(1.23)
        panel.collect_analysis_settings()
        self.assertEqual(panel.analysis_settings['calibration_method'],'reference')
        self.assertEqual(panel.analysis_settings['reference_length_cm'],1.23)
        self.assertIsNone(panel.analysis_settings['pixels_per_cm'])
        panel.close()

    def test_runway_does_not_offer_unsupported_calibration_modes(self):
        panel = AnalyzeStride.AnalyzeStridePanel(None)
        panel.spontaneous_radio.click()
        panel.collect_analysis_settings()
        self.assertFalse(panel.belt_radio.isEnabled())
        self.assertFalse(panel.reference_radio.isEnabled())
        self.assertEqual(panel.analysis_settings['calibration_method'],'manual')
        self.assertEqual(panel.analysis_settings['pixels_per_cm'],panel.pixels_per_cm_input.value())
        panel.close()

    def test_measured_modes_require_an_entered_scale_or_length(self):
        panel = AnalyzeStride.AnalyzeStridePanel(None)
        panel.wizard_step = 3
        panel.wizard_stack.setCurrentIndex(2)
        panel.manual_radio.click()
        self.assertEqual(panel.pixels_per_cm_input.value(),0.)
        self.assertFalse(panel.next_button.isEnabled())
        self.assertIn('measured pixels-per-cm',panel.settings_status_label.text())
        panel.pixels_per_cm_input.setValue(100.)
        self.assertTrue(panel.next_button.isEnabled())
        panel.reference_radio.click()
        self.assertEqual(panel.reference_length_input.value(),0.)
        self.assertFalse(panel.next_button.isEnabled())
        panel.reference_length_input.setValue(1.23)
        self.assertTrue(panel.next_button.isEnabled())
        panel.close()

    def test_gui_worker_uses_measured_scale_and_exports_it(self):
        panel = AnalyzeStride.AnalyzeStridePanel(None)
        panel.frame_rate_input.setValue(120.)
        panel.treadmill_speed_input.setValue(30.)
        panel.direction_combo.setCurrentText('Left-to-Right')
        panel.filter_cutoff_input.setValue(6.)
        panel.step_height_max_input.setValue(2.)
        panel.drag_clearance_input.setValue(.1)
        panel.manual_radio.click()
        panel.pixels_per_cm_input.setValue(100.)
        panel.collect_analysis_settings()
        with tempfile.TemporaryDirectory() as folder:
            parent = SimpleNamespace(selected_files=[str(DEMO)],custom_bodypart_mapping=None,
                analysis_type='Treadmill',output_folder=folder,analysis_settings=panel.analysis_settings)
            worker = AnalyzeStride.AnalysisThread(parent)
            messages=[]
            worker.analysis_completed.connect(lambda ok,msg:messages.append([ok,msg]))
            with patch.object(k,'return_continuous',return_value=None), contextlib.redirect_stdout(io.StringIO()):
                worker.run()
            self.assertTrue(messages[-1][0],messages)
            report=json.loads((Path(folder)/(DEMO.stem+'_analysis_report.json')).read_text())
            self.assertEqual(report['calibration']['method'],'manual')
            self.assertEqual(report['calibration']['pixels_per_cm'],100.)
            self.assertEqual(report['retained_intervals'],41)
        panel.close()


if __name__=='__main__':
    unittest.main()
