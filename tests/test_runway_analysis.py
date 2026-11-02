"""General runway failure cases; no private recording fixtures are required."""
import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ['OMP_NUM_THREADS']='1'
os.environ['LOKY_MAX_CPU_COUNT']='1'
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
import pandas as pd
from Functions import KinematicsFunctions as k
from Functions.RunwayAnalysis import COLUMNS
from runway_fixtures import synthetic_walk
from PySide6.QtWidgets import QApplication
from Panels import AnalyzeStride


class RunwayAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=synthetic_walk()
        cls.baseline=k.extract_spontaneous_parameters(100,cls.data,6,pixels_per_cm=50,drag_clearance_cm=.1)

    def analyze(self,data=None,fps=100,**kwargs):
        return k.extract_spontaneous_parameters(fps,self.data if data is None else data,6,
            **{'pixels_per_cm':50,'drag_clearance_cm':.1,**kwargs})

    def test_complete_cycles_have_known_synthetic_contact_timing(self):
        params=self.baseline[0]
        self.assertGreater(len(params),8)
        self.assertTrue(params['cycle duration (s)'].notna().all())
        for start in params['stride_start (frame)']:
            offset=(start/100.-2.)/.4
            self.assertLess(abs(offset-round(offset))*.4,.04)
        self.assertTrue((params['stride_end (frame)']>params['stride_start (frame)']).all())
        np.testing.assert_allclose(params['stance duration (s)']+params['swing duration (s)'],params['cycle duration (s)'])

    def test_no_intervals_are_created_during_stationary_or_invisible_tails(self):
        for interval in self.baseline[0].attrs['analysis_report']['intervals']:
            self.assertGreaterEqual(interval['start_frame'],190)
            self.assertLessEqual(interval['end_frame'],810)
        self.assertTrue((self.baseline[2][:100]==-1).all())
        self.assertTrue((self.baseline[2][900:]==-1).all())

    def test_stationary_tracking_noise_does_not_create_cycles(self):
        data=synthetic_walk()
        for bp in ('toe','mtp','ankle','knee','hip','iliac crest'):
            data[f'{bp} x'] -= np.clip(np.arange(len(data))/100.-2.,0.,6.)*100.
        # Fix the foot to a supported position; leave only small tracking noise.
        rng=np.random.default_rng(11)
        data['toe x']=20+rng.normal(0,.15,len(data))
        result=self.analyze(data)
        self.assertTrue(result[0].empty)
        self.assertEqual(result[0].attrs['analysis_report']['status'],'no_usable_strides')

    def test_input_data_and_frame_axis_are_preserved(self):
        data=self.data.copy(deep=True)
        data.index=np.arange(4000,4000+len(data))
        before=data.copy(deep=True)
        a=self.analyze(data); b=self.analyze(data)
        pd.testing.assert_frame_equal(data,before,check_exact=True)
        pd.testing.assert_frame_equal(a[0],b[0],check_exact=True)
        self.assertEqual(len(a[1]),len(data))
        self.assertEqual(len(a[2]),len(data))
        np.testing.assert_array_equal(a[1]['runway phase'],a[2])
        self.assertTrue(a[1]['runway drag'].iloc[900:].isna().all())

    def test_rounding_is_not_required_for_detection(self):
        rounded=self.data.copy()
        rounded['toe x']=(rounded['toe x']/.25).round()*.25
        result=self.analyze(rounded)
        self.assertLessEqual(abs(len(result[0])-len(self.baseline[0])),1)
        for frame in result[5]:
            self.assertLessEqual(min(abs(frame-np.asarray(self.baseline[5]))),3)

    def test_spatial_resolution_does_not_change_event_detection(self):
        result=self.analyze(synthetic_walk(scale=2.),pixels_per_cm=100.)
        np.testing.assert_array_equal(result[5],self.baseline[5])
        np.testing.assert_array_equal(result[2],self.baseline[2])
        np.testing.assert_allclose(result[0]['stride length (cm)'],self.baseline[0]['stride length (cm)'])

    def test_time_based_detection_scales_with_sampling_rate(self):
        high=self.analyze(synthetic_walk(fps=200),fps=200)
        self.assertLessEqual(abs(len(high[5])-len(self.baseline[5])),1)
        for time in np.asarray(high[5])/200:
            self.assertLess(min(abs(time-np.asarray(self.baseline[5])/100)),.04)

    def test_pause_inside_a_recording_splits_complete_cycles(self):
        result=self.analyze(synthetic_walk(pause=(4.,5.)))
        self.assertGreater(len(result[0]),5)
        self.assertTrue((result[2][420:480]==-1).all())
        for r in result[0].attrs['analysis_report']['intervals']:
            self.assertFalse(r['start_frame']<420 and r['end_frame']>480)

    def test_body_reference_can_fall_back_without_a_hip(self):
        data=self.data.drop(columns=['hip x','hip y','hip likelihood'])
        result=self.analyze(data)
        self.assertGreater(len(result[0]),8)
        self.assertEqual(result[0].attrs['analysis_report']['body_reference'],'iliac crest')

    def test_direction_mirroring_and_explicit_direction(self):
        mirrored=self.analyze(synthetic_walk(reverse=True))
        np.testing.assert_array_equal(mirrored[5],self.baseline[5])
        requested=self.analyze(synthetic_walk(reverse=True),right_to_left=True)
        np.testing.assert_array_equal(requested[5],mirrored[5])
        wrong=self.analyze(synthetic_walk(reverse=True),right_to_left=False)
        self.assertTrue(wrong[0].empty)
        self.assertTrue(any('direction' in s for s in wrong[0].attrs['analysis_report']['warnings']))

    def test_missing_knee_does_not_remove_timing_or_unrelated_angles(self):
        data=self.data.copy()
        data[['knee x','knee y']]=np.nan
        result=self.analyze(data)
        np.testing.assert_array_equal(result[5],self.baseline[5])
        np.testing.assert_array_equal(result[0]['cycle duration (s)'],self.baseline[0]['cycle duration (s)'])
        self.assertTrue(result[0]['knee joint extension (deg)'].isna().all())
        self.assertTrue(result[0]['ankle joint extension (deg)'].isna().all())
        self.assertTrue(result[0]['mtp joint extension (deg)'].notna().all())
        self.assertTrue(any(r['unavailable_metrics'] for r in result[0].attrs['analysis_report']['intervals']))

    def test_one_joint_gap_does_not_poison_the_entire_trial(self):
        data=self.data.copy()
        data.loc[490,'knee x']=np.nan
        result=self.analyze(data)
        self.assertTrue(result[0]['knee joint extension (deg)'].isna().any())
        self.assertTrue(result[0]['knee joint extension (deg)'].notna().any())
        np.testing.assert_array_equal(result[5],self.baseline[5])

    def test_absent_optional_joints_are_explicitly_unavailable(self):
        data=self.data.drop(columns=['knee x','knee y','knee likelihood'])
        result=self.analyze(data)
        np.testing.assert_array_equal(result[5],self.baseline[5])
        self.assertTrue(result[0]['knee joint extension (deg)'].isna().all())

    def test_long_foot_gap_splits_bouts_without_compressing_time(self):
        data=self.data.copy()
        data.loc[460:490,'toe likelihood']=.1
        result=self.analyze(data)
        for r in result[0].attrs['analysis_report']['intervals']:
            self.assertFalse(r['start_frame']<460 and r['end_frame']>490)
        self.assertEqual(len(result[1]),len(data))
        self.assertTrue((result[2][460:491]==-1).all())

    def test_short_detection_gap_is_flagged_but_not_used_for_angles(self):
        data=self.data.copy()
        data.loc[490,'toe likelihood']=.1
        result=self.analyze(data)
        report=result[0].attrs['analysis_report']
        self.assertEqual(report['tracking_confidence']['toe']['short_gap_frames_interpolated_for_detection'],1)
        self.assertTrue(any(r['detection_interpolated_frames'] for r in report['intervals']))
        self.assertTrue(result[0]['mtp joint extension (deg)'].isna().any())

    def test_truncated_recording_does_not_get_a_fabricated_final_cycle(self):
        data=self.data.iloc[:575].copy()
        result=self.analyze(data)
        self.assertTrue(all(r['end_frame']<len(data)-1 for r in result[0].attrs['analysis_report']['intervals']))
        self.assertGreater(result[0].attrs['analysis_report']['discarded_edge_contacts'],0)

    def test_zero_height_movement_is_not_assumed_to_be_normal_swing(self):
        result=self.analyze(synthetic_walk(noise=0.,lift=0.),drag_clearance_cm=.2)
        self.assertGreater(len(result[0]),5)
        self.assertTrue((result[0]['step height (cm)']<1e-8).all())
        np.testing.assert_allclose(result[0]['drag percentage (%)'],1.)
        excluded=self.analyze(synthetic_walk(noise=0.,lift=0.),step_height_min_cm=.1)
        self.assertEqual(excluded[0].attrs['analysis_report']['retained_intervals'],0)

    def test_optional_range_bypass_does_not_bypass_motion_quality(self):
        result=self.analyze(stride_length_max_cm=.00001,no_outlier_filter=True)
        self.assertGreater(result[0].attrs['analysis_report']['retained_intervals'],8)
        stationary=self.data.iloc[:180].copy()
        result=self.analyze(stationary,no_outlier_filter=True)
        self.assertTrue(result[0].empty)

    def test_optional_drag_filter_has_explicit_rejection_reasons(self):
        result=self.analyze(synthetic_walk(noise=0.,lift=0.),drag_clearance_cm=.2,dragging_filter=True)
        report=result[0].attrs['analysis_report']
        self.assertEqual(report['retained_intervals'],0)
        self.assertTrue(all('near_continuous_dragging_filter' in r['rejection_reasons'] for r in report['intervals']))

    def test_empty_return_schema_and_json_report(self):
        result=self.analyze(self.data.iloc[:10])
        self.assertEqual(len(result),6)
        self.assertEqual(list(result[0]),COLUMNS)
        self.assertTrue(result[0].empty)
        self.assertEqual(result[4],[])
        self.assertEqual(result[5],[])
        json.dumps(result[0].attrs['analysis_report'],allow_nan=False)

    def test_calibration_and_invalid_inputs_are_not_silently_assumed(self):
        for settings, message in [({'pixels_per_cm':None},'Measured pixels per cm'),
                                  ({'pixels_per_cm':0},'Measured pixels per cm'),
                                  ({'confidence_threshold':1},'confidence threshold'),
                                  ({'max_gap_s':-1},'Maximum gap')]:
            with self.subTest(settings=settings),self.assertRaisesRegex(ValueError,message):
                self.analyze(**settings)
        with self.assertRaisesRegex(ValueError,'hip or iliac crest'):
            self.analyze(self.data.drop(columns=[f'{b} {c}' for b in ('hip','iliac crest') for c in ('x','y','likelihood')]))
        with self.assertRaisesRegex(ValueError,'unique mapped'):
            self.analyze(pd.concat([self.data,self.data[['toe x']]],axis=1))

    def test_source_frame_gaps_are_not_mistaken_for_consecutive_samples(self):
        data=self.data.copy()
        data.loc[500:,'bodyparts coords']+=1
        with self.assertRaisesRegex(ValueError,'Source frame labels must be consecutive'):
            self.analyze(data)

    def test_plot_uses_only_one_contiguous_retained_bout(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cycles.svg'
            subset=k.return_continuous(self.baseline[0],plot=True,pd_dataframe_coords=self.baseline[1],
                is_stance=self.baseline[2],drag_masks=self.baseline[4],starts=self.baseline[5],filename=str(path))
            self.assertTrue(path.exists())
            self.assertGreater(len(subset),0)
            intervals=self.baseline[0].attrs['analysis_report']['intervals']
            chosen=[intervals[i] for i in subset.index]
            self.assertEqual(len({r['bout_id'] for r in chosen}),1)
            self.assertTrue(all(a['end_frame']==b['start_frame'] for a,b in zip(chosen,chosen[1:])))

    def test_stickplot_accepts_narrow_coordinate_ranges_without_distortion(self):
        data=self.data.iloc[:10].copy()
        parts=['toe','mtp','ankle','knee','hip','iliac crest']
        for i,bp in enumerate(parts):
            data[f'{bp} x']=np.linspace(0,1,10)
            data[f'{bp} y']=float(i*20)
        x,y=k.collect_filtered_coords(data,parts)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'narrow.svg'
            k.continuous_stickplot(data,parts,[1]*10,x,y,0,9,str(path))
            self.assertTrue(path.exists())

    def test_gui_has_mode_specific_direction_and_detection_controls(self):
        app=QApplication.instance() or QApplication([])
        panel=AnalyzeStride.AnalyzeStridePanel(None)
        panel.direction_combo.setCurrentText('Right-to-Left')
        panel.spontaneous_radio.click()
        self.assertEqual(panel.direction_combo.currentText(),'Auto-detect')
        self.assertFalse(panel.runway_group.isHidden())
        panel.direction_combo.setCurrentText('Left-to-Right')
        panel.treadmill_radio.click()
        self.assertEqual(panel.direction_combo.currentText(),'Right-to-Left')
        self.assertTrue(panel.runway_group.isHidden())
        panel.spontaneous_radio.click()
        self.assertEqual(panel.direction_combo.currentText(),'Left-to-Right')
        panel.close()

    def test_gui_worker_exports_phase_gap_and_metric_quality(self):
        app=QApplication.instance() or QApplication([])
        panel=AnalyzeStride.AnalyzeStridePanel(None)
        panel.spontaneous_radio.click()
        panel.frame_rate_input.setValue(100.)
        panel.pixels_per_cm_input.setValue(50.)
        panel.runway_confidence_input.setValue(.75)
        panel.runway_gap_input.setValue(0.)
        panel.runway_body_speed_input.setValue(.35)
        panel.runway_contact_input.setValue(40.)
        panel.joint_interpolation_input.setCurrentIndex(panel.joint_interpolation_input.findData('pchip'))
        panel.joint_cutoff_checkbox.setChecked(True)
        panel.joint_cutoff_input.setValue(10.)
        panel.collect_analysis_settings()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'motion.csv'
            source=self.data.copy()
            source.columns=pd.MultiIndex.from_tuples([('scorer','bodyparts','coords')]+[
                ('synthetic',col.rsplit(' ',1)[0],col.rsplit(' ',1)[1]) for col in source.columns[1:]])
            source.to_csv(path,index=False)
            parent=SimpleNamespace(selected_files=[str(path)],custom_bodypart_mapping=None,
                analysis_type='Spontaneous walking',output_folder=folder,analysis_settings=panel.analysis_settings)
            worker=AnalyzeStride.AnalysisThread(parent)
            messages=[]
            worker.analysis_completed.connect(lambda ok,message:messages.append((ok,message)))
            with contextlib.redirect_stdout(io.StringIO()): worker.run()
            self.assertTrue(messages[-1][0],messages)
            report=json.loads((Path(folder)/'motion_analysis_report.json').read_text())
            self.assertEqual(report['analysis_type'],'spontaneous')
            self.assertEqual(report['settings']['confidence_threshold'],.75)
            self.assertEqual(report['settings']['max_gap_s'],0.)
            self.assertEqual(report['settings']['min_contact_s'],.04)
            self.assertEqual(report['settings']['joint_interpolation'],'pchip')
            self.assertEqual(report['settings']['joint_filter_cutoff_hz'],10.)
            self.assertEqual(report['stickplot_coordinate_source'],'tracked')
            self.assertEqual(report['settings']['joint_max_interpolated_fraction'],.1)
            self.assertGreater(report['retained_intervals'],5)
            self.assertTrue((Path(folder)/'motion_stickplot.svg').exists())
        panel.close()


if __name__=='__main__':
    unittest.main()
