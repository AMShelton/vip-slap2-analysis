"""Synthetic fixtures: inspect source and derived timestamps without any user data."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# When copied into the target repository, this module is resolved from src/.
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
from vip_slap2_analysis.metadata.export_sync_audit_bundle import export_session


def test_registry_end_to_end(tmp_path):
    session_id = '123456_2026-08-04_08-12-47'
    session = tmp_path / 'datasets' / 'ASAP8' / '123456' / session_id
    session.mkdir(parents=True)
    summary_path = tmp_path / 'datasets' / 'VIP_SD_summary.xlsx'
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        pd.DataFrame({'session_id':[session_id],'subject_id':[123456],
                      'session_dir':[str(session)], 'session_type':['A0'],
                      'paradigm':['change_detection_passive'],
                      'indicator1':['ASAP8']}).to_excel(writer, sheet_name='sessions', index=False)
        pd.DataFrame({'comment':['dummy']}).to_excel(writer, sheet_name='subjects',
                                                    index=False,startrow=3)

    src_dir = session / 'subfolder' / 'source_extraction' / 'dendriticVoltageExtraction'
    src_dir.mkdir(parents=True)
    summat = src_dir / 'dendriticVoltageSummary-260804-114556.mat'
    with h5py.File(summat, 'w') as f:
        summary = f.create_group('summary')
        summary.create_dataset('nDMDs', data=[[2]])
        summary.create_dataset('nTrials', data=[[3]])
        summary.create_dataset('nAnalysisROIs', data=np.asarray([2,2]).reshape(2,1))
        summary.create_dataset('roiGlobalOffsets', data=np.asarray([0,2]).reshape(2,1))
        summary.create_dataset('trialEpoch', data=[[1,1,2]])
        summary.create_dataset('analyzeHz',data=[[1000.]])
    src = src_dir / 'dendriticVoltageTraces-260804-114556.h5'
    with h5py.File(src, 'w') as h:
        h.attrs['outputMode'] = 'trial'
        group = h.create_group('traces')
        for trial in range(1, 4):
            ds = group.create_dataset(f'trial_{trial:04d}',data=np.arange(400,dtype=np.float32).reshape(4,100))
            ds.attrs['trialIdx'] = trial
            ds.attrs['trialFirstLineGlobal'] = trial*100
            ds.attrs['trialLastLineGlobal'] = trial*100+99
            ds.attrs['firstLineByDmd'] = np.array([trial*100,trial*100])
    ddir = session / 'analysis' / 'derived' / 'voltage'
    ddir.mkdir(parents=True)
    dst = ddir / 'voltage_session_traces_dff_robust_f0_trial.h5'
    # Two imaging epochs separated by 2.0s; a single timestamp anomaly in DMD2.
    timestamps = np.r_[np.arange(200)/1000, 2 + np.arange(100)/1000].astype('f8')
    labels = np.r_[np.ones(200), np.full(100,2)].astype(np.int16)
    with h5py.File(dst, 'w') as h:
        for dmd in (1,2):
            group=h.create_group(f'DMD{dmd}')
            tt=timestamps.copy()
            if dmd==2:
                tt[240]+=0.0005
            group.create_dataset('timebase_sec',data=tt)
            group.create_dataset('sample_epoch',data=labels)
            group.create_dataset('trial_lengths_samples',data=[100,100,100])
            group.create_dataset('trial_starts_sec',data=[0,0.1,2])
            group.create_dataset('trial_valid_mask',data=[True,True,True])
            group.create_dataset('raw_f',data=np.arange(600,dtype=np.float32).reshape(2,300))
            group.create_dataset('dff',data=np.arange(600,dtype=np.float32).reshape(2,300)*0.01)
            group.attrs['reconstruction_metadata_json']=json.dumps({'nominal_sample_rate_hz':1000})
    bdir = session / 'analysis' / 'qc' / 'behavior'
    bdir.mkdir(parents=True)
    (bdir / 'imaging_epochs.csv').write_text('epoch_index,start_idx,end_idx,start_time,end_time,duration_s\n1,0,199,0,0.199,0.199\n2,200,299,2,2.099,0.099\n')
    (bdir / 'di3_pulse_train_detection.json').write_text(json.dumps({'n_rising_edges':3,'median_period_s':0.1,'candidate_epochs':[]}))
    harpd=session/'behavior'/'VCO1_Behavior.harp'/'extracted_files'
    harpd.mkdir(parents=True)
    (harpd/'HARP_df.csv').write_text('time,DI3\n0,0\n0.1,1\n0.11,1\n0.12,0\n0.2,1\n0.21,0\n2.1,1\n')
    registry=VIPSessionRegistry.from_excel(summary_path)
    asset=registry.resolve_assets(session_id)
    assert asset.harp_df_csv.is_file()
    assert asset.get_asset('voltage','trace_h5') == src
    out=tmp_path/'export.zip'
    result=export_session(asset,out,chunk=1024,trace_window_samples=20,roi_limit=2,boundary_samples=3)
    assert result['zip_bytes']>0
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert 'harp/di3_rising_edges.csv' in names
        assert 'source/voltage_trace_inventory.json' in names
        assert 'derived/voltage_audit.json' in names
        info=json.loads(z.read('source/voltage_trace_inventory.json'))
        assert len(info['datasets'])==3
        assert info['datasets'][0]['attrs']['trialFirstLineGlobal']==100
        audit=json.loads(z.read('derived/voltage_audit.json'))
        assert audit['dmds']['DMD1']['timebase_scan']['n_epoch_transitions']==1
        assert audit['dmds']['DMD1']['timebase_scan']['n_uneven_steps_within_epoch']==0
        assert audit['dmds']['DMD2']['timebase_scan']['n_uneven_steps_within_epoch']>=1
        assert audit['dmd_comparison']['n_nonidentical']==1
        assert audit['dmd_comparison']['max_abs_diff_sec']>0.00049
        assert audit['dmds']['DMD1']['trial_boundaries'][2]['sample_idx']==100
        raw = np.load(__import__('io').BytesIO(z.read('derived/DMD1_raw_f_start.npy')))
        assert np.array_equal(raw[0], np.arange(20, dtype=np.float32))
        assert np.array_equal(raw[1], np.arange(300, 320, dtype=np.float32))
        assert '0.10000000000000001' in z.read('harp/di3_rising_edges.csv').decode()
        assert len(z.read('harp/di3_rising_edges.csv').decode().splitlines())==4
        manifest=json.loads(z.read('manifest.json'))
        assert 'source/voltage_summary_metadata.json' in names
        sm = json.loads(z.read('source/voltage_summary_metadata.json'))
        assert sm['n_trials'] == 3 and sm['n_dmds'] == 2
    # Guard against accidental overwrite.
    try:
        export_session(asset,out)
    except FileExistsError:
        pass
    else:
        raise AssertionError('overwriting should require explicit --overwrite')
