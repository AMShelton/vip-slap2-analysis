from pathlib import Path
import json
import math
import sys
import tempfile
import numpy as np
import h5py
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vip_slap2_analysis.metadata.t03_start_order import (
    read_slap2_meta,read_behavior_launcher,classify_pair,
    manifest_coverage,audit_session,
)

def build_test_mat_meta(p:Path):
    """Portable minimal .meta fixture with actual MATLAB-style HDF5 header."""
    p.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(p,'w',userblock_size=512) as f:
        f.create_dataset('acqDuration_s',data=np.array([[1830.]],float))
        f.create_dataset('acqDurationCycles',data=np.array([[81827.],[150436.]],float))
        f.create_dataset('linePeriod_s',data=np.array([[1/10686.667]],float))
    with p.open('r+b') as handle:
        handle.write(b'MATLAB 7.3 MAT-file, Platform: PCWIN64, Created on: Tue Sep 15 14:48:56 2026 HDF5 schema 1.00')

class TestClockOrder(unittest.TestCase):
    def test_slap2_first_example(self):
        src={'meta_header_time':'2026-09-15T14:48:56','meta_duration_s':1830.,
             'meta_cycles_dmd1':81827,'meta_period_s':1830/81827}
        bhv={'workflow_launch_time':'2026-09-15T14:49:07',
             'harp_folder_time':'2026-09-15T14:49:11'}
        r=classify_pair(src,bhv,di3_count=80714,measured_di3_period_s=1830/81827)
        self.assertEqual(r['order_classification'],'slap2_first')
        self.assertEqual(r['unrecorded_cycle_count_equivalent'],1113)
        self.assertGreater(r['initial_cycle_offset_estimate'],0)
        self.assertGreater(r['initial_cycle_offset_upper'],r['initial_cycle_offset_lower'])
        self.assertEqual(r['alignment_evidence_status'],'NOT_HARDWARE_VERIFIED')
    def test_count_is_not_sufficient_without_cadence(self):
        src={'meta_header_time':'2026-09-15T14:48:56','meta_duration_s':1830.,
             'meta_cycles_dmd1':81827,'meta_period_s':1830/81827}
        bhv={'harp_folder_time':'2026-09-15T14:49:11'}
        r=classify_pair(src,bhv,di3_count=80714)
        self.assertIsNone(r['unrecorded_cycle_count_equivalent'])
        self.assertFalse(r['di3_dmd1_cadence_verified'])
    def test_early_harp_before_source(self):
        src={'meta_header_time':'2026-03-02T13:13:11','meta_period_s':1800/84741,
             'meta_cycles_dmd1':84741,'meta_duration_s':1800}
        bhv={'workflow_launch_time':'2026-03-02T13:13:04',
             'harp_folder_time':'2026-03-02T13:13:08'}
        r=classify_pair(src,bhv,di3_count=84734)
        self.assertEqual(r['order_classification'],'ambiguous_overlap')
        self.assertEqual(r['initial_cycle_offset_lower'],0)
    def test_missing_never_passes(self):
        r=classify_pair({}, {})
        self.assertEqual(r['order_classification'],'missing_timing_evidence')
        self.assertNotEqual(r['alignment_evidence_status'],'PASS')
    def test_source_filename(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'acquisition_20260915_144855_DMD1.meta'
            build_test_mat_meta(p)
            src=read_slap2_meta(p)
            self.assertEqual(src['meta_header_time'],'2026-09-15T14:48:56')
            self.assertAlmostEqual(src['meta_duration_s'],1830)
            self.assertAlmostEqual(src['meta_cycles_dmd1'],81827)
    def test_launcher_parser(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'launcher.log'
            p.write_text('2026-09-15 14:49:07 - root - INFO - Starting Bonsai workflow attempt 1 (retries used: 0)\n'
                         '2026-09-15 15:19:16 - root - INFO - Selected harp folder: C:\\session\\852836_20260915T144911\\Behavior.harp\n')
            b=read_behavior_launcher(p)
            self.assertEqual(b['workflow_launch_time'],'2026-09-15T14:49:07')
            self.assertEqual(b['harp_folder_time'],'2026-09-15T14:49:11')
    def test_manifest_preview(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.tsv';p.write_text('ASAP7\\123456\\123456_2026-01-01_12-00-00\\slap2\\dynamic_data\\acquisition_123456_DMD1.meta\tfile\n'
                                                'ASAP7\\123456\\123456_2026-01-01_12-00-00\\behavior\\launcher_metadata\\launcher.log\tfile\n')
            df=manifest_coverage(p)
            self.assertEqual(len(df),1)
            self.assertTrue(bool(df.iloc[0].source_dmd1_meta))
            self.assertTrue(bool(df.iloc[0].behavior_launcher_log))
    def test_pairing_multi_acquisition(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'852836_2026-09-15_14-29-34'
            (root/'slap2'/'dynamic_data').mkdir(parents=True)
            (root/'behavior'/'launcher_metadata').mkdir(parents=True)
            build_test_mat_meta(root/'slap2'/'dynamic_data'/'acquisition_144855_DMD1.meta')
            (root/'behavior'/'launcher_metadata'/'launcher.log').write_text(
                '2026-09-15 14:49:07 - root - INFO - Starting Bonsai workflow attempt 1 (retries used: 0)\n'
                '2026-09-15 15:19:16 - root - INFO - Selected harp folder: C:\\session\\852836_20260915T144911\\Behavior.harp\n')
            r=audit_session(root.name,root)
            self.assertEqual(r['order_classification'],'slap2_first')

if __name__=='__main__':unittest.main()
