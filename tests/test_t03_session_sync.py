"""Pure and synthetic T0.3 tests. No VAST and no raw SLAP2 files needed."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# When running under repo tests, package is editable-installed or on PYTHONPATH.
from vip_slap2_analysis.metadata.t03_session_sync import (
    di3_clock_diagnostics, nearest_pulse_phase, rising_edges_from_harp_csv,
    apply_independent_validation, source_voltage_boundaries, giant_frame_index_inventory,
    original_behavior_alignment_method,
)


def test_di3_period_gap_not_dropped_pulse():
    edges=np.r_[np.arange(0,10,.02),np.arange(30,40,.02)]
    q=di3_clock_diagnostics(edges)
    assert q['di3_n_epoch_gaps']==1
    assert q['di3_n_outlier_intervals']==0
    assert abs(q['di3_period_ms']-20)<1e-7


def test_di3_event_phase_positive_and_bounded():
    clocks=np.arange(0,10,.02)
    events=np.arange(.004,9,.4)
    q=nearest_pulse_phase(events,clocks)
    assert q['n']==len(events)
    assert abs(q['median_ms']-4)<1e-8
    assert abs(q['p95_abs_ms']-4)<1e-8


def test_harp_edge_chunk_transition(tmp_path):
    t=np.arange(0,2,.01)
    state=(np.floor(t/.02).astype(int)%2)==0
    pd.DataFrame({'time':t,'DI3':state}).to_csv(tmp_path/'HARP_df.csv',index=False)
    x=rising_edges_from_harp_csv(tmp_path/'HARP_df.csv',chunk_size=3)
    assert len(x)>20
    assert (np.diff(x)>0).all()


def test_external_validation_must_have_source_evidence_and_frame(tmp_path):
    t=pd.DataFrame({'session_id':['a','b','c'], 't03_status':['NOT_INDEPENDENTLY_VERIFIED']*3})
    v=pd.DataFrame([
        {'session_id':'a','imaging_to_stimulus_lag_ms':2,'p95_abs_error_ms':5,'verified_frame_period_ms':16.67,'method':'line_to_di3','evidence_path':'a_report.json','independent_validation':True},
        {'session_id':'b','imaging_to_stimulus_lag_ms':12,'p95_abs_error_ms':20,'verified_frame_period_ms':16.67,'method':'line_to_di3','evidence_path':'b_report.json','independent_validation':True},
        {'session_id':'c','imaging_to_stimulus_lag_ms':1,'p95_abs_error_ms':1,'verified_frame_period_ms':16.67,'method':'','evidence_path':'','independent_validation':False},
    ])
    p=tmp_path/'verified.csv';v.to_csv(p,index=False)
    out=apply_independent_validation(t,p)
    assert list(out.t03_status)==['PASS','FAIL','NOT_INDEPENDENTLY_VERIFIED']


def test_source_repeated_edges(tmp_path):
    import h5py
    p=tmp_path/'source.h5'
    with h5py.File(p,'w') as h:
        grp=h.create_group('traces')
        for i,(start,end) in enumerate([(1,4),(4,7),(7,10)],1):
            ds=grp.create_dataset(f'trial_{i:04d}',data=np.array([[start,start+1,start+2,end]],float))
            ds.attrs['trialFirstLineGlobal']=start
            ds.attrs['trialLastLineGlobal']=end
            ds.attrs['acquisitionPrefix']='E1'
            ds.attrs['epochIdx']=1
    q=source_voltage_boundaries(p)
    assert q['source_boundaries_tested']==2
    assert q['source_shared_lines']==2
    assert q['source_identical_samples']==2


def test_giant_transposed_line_indices(tmp_path):
    import h5py
    p=tmp_path/'experiment_summary.h5'
    with h5py.File(p,'w') as h:
        d=h.create_dataset('Path1/frame_info/frame_line_idxs',data=np.array([[1,10,20,20,31]]))
    q=giant_frame_index_inventory(tmp_path)
    assert q['giant_frame_line_count']==5
    assert q['giant_nonincreasing_frame_line_steps']==1


def test_corrected_event_log_recovers_original_frame_modulo(tmp_path):
    csv=tmp_path/'bonsai_event_log.csv'
    pd.DataFrame({'alignment_method':['frame_modclass_affine_anchored_edges']*3,
                  'corrected_timestamps':[.67,.70,.73]}).to_csv(csv,index=False)
    out=original_behavior_alignment_method({'alignment_method':'existing_corrected_timestamps',
            'reused_existing_corrected_event_log':True},csv)
    assert out['alignment_pathway']=='legacy_frame_modulo'
    assert out['alignment_method']=='frame_modclass_affine_anchored_edges'
    assert out['alignment_method_saved']=='existing_corrected_timestamps'
    assert out['alignment_reused_corrected']
    assert out['alignment_method_source'].startswith('bonsai_event_log.csv')


def test_existing_corrected_without_provenance_stays_unknown(tmp_path):
    csv=tmp_path/'bonsai_event_log.csv'
    pd.DataFrame({'Timestamp':[0,1], 'corrected_timestamps':[.7,1.7]}).to_csv(csv,index=False)
    out=original_behavior_alignment_method({'alignment_method':'existing_corrected_timestamps'},csv)
    assert out['alignment_pathway']=='precorrected_without_method'


def test_conflicting_metadata_event_log_flagged(tmp_path):
    csv=tmp_path/'bonsai_event_log.csv'
    pd.DataFrame({'alignment_method':['frame_modclass_affine_anchored_edges']}).to_csv(csv,index=False)
    out=original_behavior_alignment_method({'alignment_method':'bv_photodiode_piecewise'},csv)
    assert out['alignment_pathway']=='unknown'
    assert out['alignment_method_conflict']


def test_original_bv_photodiode_method_recovers_from_csv(tmp_path):
    csv=tmp_path/'bonsai_event_log.csv'
    pd.DataFrame({'alignment_method':['bv_photodiode_piecewise']}).to_csv(csv,index=False)
    out=original_behavior_alignment_method({'alignment_method':'existing_corrected_timestamps'},csv)
    assert out['alignment_pathway']=='encoded_photodiode'
