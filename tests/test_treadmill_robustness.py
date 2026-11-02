"""Regression checks and failure cases derived from ALMA's bundled demo.

Run: python -m unittest discover -s tests -v
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from Functions import KinematicsFunctions as k
from Functions.TreadmillValidation import PARAMETER_COLUMNS
from Panels import AnalyzeStride
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures'
DEMO = ROOT / 'Resources/Demo_Mouse_Treadmill_30cm_s_650000_filtered.csv'
ARGS = dict(cm_speed=30, right_to_left=False, step_height_max_cm=2, drag_clearance_cm=0.1)
DRAG_COLUMNS = ['drag duration (s)', 'drag percentage (%)']


class TreadmillRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(ROOT)
        cls.app = QApplication.instance() or QApplication([])
        cls.raw = pd.read_csv(DEMO, header=[1,2])
        cls.data, _, _ = k.fix_column_names(cls.raw.copy())
        cls.result = k.extract_parameters(120, cls.data, 6, 'toe', **ARGS)
        cls.expected = pd.read_csv(FIXTURE/'treadmill_parameters.csv', float_precision='round_trip')
        cls.reference = json.loads((FIXTURE/'treadmill_baseline.json').read_text())

    def run_core(self, data=None, **overrides):
        args = {**ARGS, **overrides}
        return k.extract_parameters(120, self.data if data is None else data, 6, 'toe', **args)

    def test_demo_44_non_drag_columns_match_baseline_exactly(self):
        self.assertEqual(hashlib.sha256(DEMO.read_bytes()).hexdigest(), self.reference['demo_csv_sha256'])
        pd.testing.assert_frame_equal(self.result[0].drop(columns=DRAG_COLUMNS),
                                     self.expected.drop(columns=DRAG_COLUMNS), check_exact=True)
        self.assertEqual(len(self.result[0]), 49)
        self.assertEqual(self.result[0]['cycle duration (s)'].notna().sum(), 37)

    def test_demo_coordinates_and_events_match_baseline_exactly(self):
        self.assertEqual(hashlib.sha256(self.result[1].to_csv(index=False).encode()).hexdigest(),
                         self.reference['coordinates_csv_sha256'])
        with np.load(FIXTURE/'treadmill_events.npz') as expected:
            np.testing.assert_array_equal(self.result[2], expected['stance'])
            np.testing.assert_array_equal(self.result[5], expected['starts'])

    def test_same_input_can_be_analyzed_repeatedly_without_mutation(self):
        before = self.data.copy(deep=True)
        a, b = self.run_core(), self.run_core()
        pd.testing.assert_frame_equal(self.data, before, check_exact=True)
        pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
        pd.testing.assert_frame_equal(a[1], b[1], check_exact=True)

    def test_nonzero_dataframe_index_preserves_row_based_results(self):
        data = self.data.copy()
        data.index = np.arange(1000, 1000+len(data))
        result = self.run_core(data)
        pd.testing.assert_frame_equal(result[0], self.result[0], check_exact=True)

    def test_aliases_and_mapping_order(self):
        mapping, _ = k.detect_bodypart_mapping(['back_toe','MTP','Ankle','Knee','Hip','iliac_crest'])
        self.assertEqual(mapping['back_toe'], 'toe')
        self.assertEqual(mapping['iliac_crest'], 'iliac crest')
        order = {'hip':'hip','toe':'toe','mtp':'mtp','ankle':'ankle','knee':'knee','iliac crest':'iliac crest'}
        _, bodyparts, _ = k.fix_column_names(self.raw.copy(), order)
        self.assertEqual(bodyparts, list(order.values()))

    def test_missing_required_joint_is_actionable(self):
        with self.assertRaisesRegex(ValueError, 'Missing required.*knee x'):
            self.run_core(self.data.drop(columns=['knee x']))

    def test_duplicate_mapped_limb_columns_are_rejected(self):
        data = pd.concat([self.data, self.data[['toe x']]], axis=1)
        with self.assertRaisesRegex(ValueError, 'Duplicate mapped columns'):
            self.run_core(data)

    def test_missing_or_infinite_coordinates_are_rejected_before_filtering(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                data = self.data.copy()
                data.loc[10, 'knee x'] = value
                with self.assertRaisesRegex(ValueError, 'knee x.*1 frames'):
                    self.run_core(data)

    def test_nonnumeric_coordinates_are_rejected(self):
        data = self.data.copy()
        data['knee x'] = data['knee x'].astype(str)
        with self.assertRaisesRegex(ValueError, 'Non-numeric.*knee x'):
            self.run_core(data)

    def test_invalid_settings(self):
        for name, value, message in [('cm_speed',0,'Treadmill speed'),
                                     ('cm_speed',np.nan,'Treadmill speed'),
                                     ('drag_clearance_cm',-1,'Drag clearance'),
                                     ('drag_min_consecutive_frames',0,'Consecutive drag'),
                                     ('stride_length_max_cm',-1,'Stride length limits')]:
            with self.subTest(setting=name):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_core(**{name:value})
        for fps, cutoff, message in [(0,6,'Frame rate'),(120,60,'Nyquist'),(120,-1,'Lowpass')]:
            with self.subTest(fps=fps,cutoff=cutoff):
                with self.assertRaisesRegex(ValueError,message):
                    k.extract_parameters(fps,self.data,cutoff,'toe',**ARGS)

    def test_short_clip_is_actionable(self):
        with self.assertRaisesRegex(ValueError, 'At least 19.*18'):
            self.run_core(self.data.iloc[:18])

    def test_insufficient_toe_confidence_is_actionable(self):
        data = self.data.copy()
        data['toe likelihood'] = 0.1
        with self.assertRaisesRegex(ValueError, 'Insufficient confident toe tracking'):
            self.run_core(data)

    def test_out_of_range_likelihood_is_rejected(self):
        data = self.data.copy()
        data.loc[0,'knee likelihood'] = 1.1
        with self.assertRaisesRegex(ValueError, 'knee likelihood must be between'):
            self.run_core(data)

    def test_no_motion_and_wrong_direction_have_clear_errors(self):
        data = self.data.copy()
        data['toe x'] = 100.
        with self.assertRaisesRegex(ValueError, 'No usable toe movement'):
            self.run_core(data)
        data['toe x'] = np.arange(len(data),dtype=float)**2
        data['toe likelihood'] = 1.
        with self.assertRaisesRegex(ValueError, 'selected walking direction'):
            self.run_core(data)

    def test_degenerate_and_nearly_collinear_angles(self):
        self.assertTrue(np.isnan(k.angle([0,0],[1,1])))
        self.assertTrue(np.isnan(k.angle([np.nan,0],[1,1])))
        rng = np.random.default_rng(3)
        for v in rng.uniform(-1,1,(100,2)):
            self.assertAlmostEqual(k.angle(v,v),0.,places=6)

    def test_drag_keeps_whole_runs_and_rejects_short_contacts(self):
        cases = [([1]*10,4,[1]*10),([1]*3,4,[0]*3),
                 ([1]*4,4,[1]*4),([0,1,1,1,1,0,1,1,1],4,[0,1,1,1,1,0,0,0,0]),
                 ([0,0,0],1,[0,0,0]),([1,0,1],1,[1,0,1])]
        for contacts, minimum, expected in cases:
            with self.subTest(contacts=contacts,minimum=minimum):
                y = np.array(contacts,dtype=float)*100
                count, fraction, mask = k.find_drag(y,0,100.,0,len(y),0,minimum)
                np.testing.assert_array_equal(mask,np.array(expected,dtype=bool))
                self.assertEqual(count,sum(expected))
                self.assertEqual(fraction,sum(expected)/len(expected))

    def test_drag_masks_and_reported_durations_are_consistent(self):
        params, _, _, _, masks, starts = self.result
        for i,row in params.iterrows():
            if pd.notna(row['cycle duration (s)']):
                self.assertAlmostEqual(row['drag duration (s)'],np.count_nonzero(masks[i])/120.)
                active = np.flatnonzero(np.diff(np.r_[False,masks[i],False].astype(int)))
                self.assertTrue(all(end-start>=4 for start,end in zip(active[::2],active[1::2])))

    def test_quality_report_explains_rejections_and_confidence(self):
        report = self.result[0].attrs['analysis_report']
        self.assertEqual(report['retained_intervals'],37)
        self.assertEqual(report['rejected_intervals'],12)
        self.assertEqual(report['tracking_confidence']['toe']['frames_at_or_below_0_5'],329)
        rejected = [r for r in report['intervals'] if not r['retained']]
        self.assertTrue(all(r['rejection_reasons'] for r in rejected))
        json.dumps(report,allow_nan=False)

    def test_all_rejected_is_not_reported_as_usable(self):
        result = self.run_core(stride_length_max_cm=0.00001)
        self.assertEqual(result[0].attrs['analysis_report']['status'],'no_usable_strides')
        self.assertEqual(result[0].attrs['analysis_report']['retained_intervals'],0)

    def test_no_detected_cycles_returns_consistent_empty_result(self):
        no_events = ([False]*(len(self.data)-1),[],[],100.)
        with patch.object(k,'_robust_stance_segments_treadmill',return_value=no_events), \
             patch.object(k,'_detect_dragging_movement_cycles',return_value=no_events):
            result = self.run_core()
        self.assertEqual(len(result),6)
        self.assertTrue(result[0].empty)
        self.assertEqual(list(result[0]),PARAMETER_COLUMNS)
        self.assertEqual(result[0].attrs['analysis_report']['status'],'no_usable_strides')

    def run_worker(self,mapping,folder,max_length=8):
        parent = SimpleNamespace(selected_files=[str(DEMO)],custom_bodypart_mapping=mapping,
            analysis_type='Treadmill',output_folder=folder,analysis_settings={
                **ARGS,'frame_rate':120.,'filter_cutoff':6.,'step_height_min_cm':0.,
                'stride_length_min_cm':0.,'stride_length_max_cm':max_length,
                'drag_min_consecutive_frames':4})
        thread = AnalyzeStride.AnalysisThread(parent)
        results=[]
        thread.analysis_completed.connect(lambda ok,msg:results.append((ok,msg)))
        with contextlib.redirect_stdout(io.StringIO()):
            thread.run()
        return results

    def test_gui_worker_mapping_order_and_automatic_mapping(self):
        for mapping in [None,{bp:bp for bp in ['hip','knee','ankle','mtp','toe','iliac crest']}]:
            with self.subTest(mapping=mapping),tempfile.TemporaryDirectory() as folder:
                result=self.run_worker(mapping,folder)
                self.assertTrue(result[-1][0],result)
                exported=pd.read_csv(Path(folder)/(DEMO.stem+'_parameters.csv'),float_precision='round_trip')
                pd.testing.assert_frame_equal(exported,self.result[0],check_exact=True)
                report=json.loads((Path(folder)/(DEMO.stem+'_analysis_report.json')).read_text())
                self.assertEqual(report['retained_intervals'],37)
                self.assertIn('Quality warnings',result[-1][1])

    def test_gui_worker_reports_zero_usable_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            result=self.run_worker(None,folder,max_length=0.00001)
            self.assertFalse(result[-1][0])
            self.assertIn('No usable stride',result[-1][1])
            self.assertTrue((Path(folder)/(DEMO.stem+'_analysis_report.json')).exists())

    def test_fps_preserves_two_decimals_and_remains_visible(self):
        panel=AnalyzeStride.AnalyzeStridePanel(None)
        panel.frame_rate_input.setValue(239.78)
        panel.spontaneous_radio.setChecked(True)
        panel.on_analysis_type_changed()
        panel.wizard_stack.setCurrentIndex(2)
        panel.show()
        self.app.processEvents()
        self.assertEqual(panel.frame_rate_input.value(),239.78)
        self.assertTrue(panel.frame_rate_input.isVisible())
        self.assertTrue(panel.speed_group.isHidden())
        panel.close()


if __name__ == '__main__':
    unittest.main()
