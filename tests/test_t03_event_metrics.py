from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from vip_slap2_analysis.metadata.t03_event_metrics import event_pd_diagnostics, _match_unique


def make_pair(tmp_path,legacy=False):
    dt=.001; tt=np.arange(0,25,dt)
    # Physical PD changes at every half-second. Modern rise once every second.
    signal=(np.floor(tt/.5).astype(int)%2==1).astype(float)
    p=tmp_path/'photodiode.pkl'
    pd.DataFrame({'AnalogInput0':signal},index=tt).to_pickle(p)
    times=np.arange(0,25,1/60)
    frames=np.arange(len(times))
    if legacy:
        vals=['Frame']*len(times)
        rows=pd.DataFrame({'Value':vals,'Frame':frames, 'Timestamp':times,
                           'corrected_timestamps':times+.001})
        method='legacy_frame_modulo'
    else:
        # Rising edges at .5,1.5,2.5, etc; assign states to adjacent frames.
        states=(np.floor(times/.5).astype(int)%2==1).astype(float)
        rows=pd.DataFrame({'Value':['Frame']*len(times),'Frame':frames,'Timestamp':times,
                           'corrected_timestamps':times+.001,'photodiode_state':states})
        method='encoded_photodiode'
    f=tmp_path/'bonsai_event_log.csv';rows.to_csv(f,index=False)
    return f,p,method


def test_encoded_event_pd(tmp_path):
    f,p,method=make_pair(tmp_path)
    r=event_pd_diagnostics(f,p,method,frame_period_ms=1000/60)
    assert r['event_match_n']>=15,r
    assert r['event_coverage']>.9,r
    assert r['event_p95_abs_ms']<1000/60,r
    assert r['event_fraction_within_frame']>.95
    assert r['event_comparison_status'].startswith('modern_')
    assert abs(r['harp_pd_sample_dt_median_ms']-1)<.001


def test_legacy_phase_estimated_but_not_independent(tmp_path):
    f,p,method=make_pair(tmp_path,legacy=True)
    r=event_pd_diagnostics(f,p,method,frame_period_ms=1000/60,saved_phase=0)
    assert r['event_match_n']>=25,r
    assert r['event_comparison_status'].startswith('legacy_')
    assert r['event_phase_source'].startswith('saved_phase')
    assert r['event_p95_abs_ms']<1000/60


def test_ambiguous_duplicate_event_correspondence_is_rejected():
    a,b=_match_unique([1.0,1.01,2.0], [1.005,2.0], .02)
    assert list(a)==[2] and list(b)==[1]


def test_missing_event_clock_columns_fail_explicitly(tmp_path):
    f,p,method=make_pair(tmp_path)
    pd.DataFrame({'Value':['Frame'],'Timestamp':[0],'Frame':[0]}).to_csv(f,index=False)
    with pytest.raises(ValueError,match='corrected_timestamps'):
        event_pd_diagnostics(f,p,method)


def test_modern_reconstructs_photodiode_state_from_original_events(tmp_path):
    f,p,method=make_pair(tmp_path)
    z=pd.read_csv(f)
    z=z.drop(columns='photodiode_state')
    t=np.arange(.5,24.9,.5)
    events=pd.DataFrame({'Value':[f'Photodiode-{int(round(v/.5))%2}' for v in t],
                         'Frame':0,'Timestamp':t-1e-7,'corrected_timestamps':t+.001})
    z=pd.concat([z,events],ignore_index=True).sort_values('Timestamp',kind='stable')
    z.to_csv(f,index=False)
    r=event_pd_diagnostics(f,p,method,frame_period_ms=1000/60)
    assert r['event_match_n']>=15,r
    assert r['event_p95_abs_ms']<1000/60


def test_crossing_di3_phase_is_conditional(tmp_path):
    f,p,method=make_pair(tmp_path)
    clock=np.arange(0,25,.02)
    out=event_pd_diagnostics(f,p,method,frame_period_ms=1000/60,
                             di3_edges_abs=clock,shared_origin_verified=True)
    assert out['event_bv_di3_clock_status']=='conditional_shared_HARP_index_origin'
    assert np.isfinite(out['event_bv_to_di3_p95_abs_phase_ms'])
    assert out['event_same_nearest_di3_cycle_fraction']>=0.9
