from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
# Source repository must be on PYTHONPATH or installed editable.
from vip_slap2_analysis.metadata import t03_cycle_clock as tc


def test_cycle_model_anchors_origin_but_detects_rate_drift():
    x=17+np.arange(15000)*0.0200001
    s=tc.check_epochs(x,0.020)[0]
    assert s['n_pulses']==15000
    assert abs(s['residual_ms'][0])<1e-9
    assert abs(s['drift_end_ms']-1.4999)<0.005
    assert s['irregular_intervals']==0


def test_epoch_gaps_not_counted_as_missing_pulses():
    e=np.r_[np.arange(800)*.02, 25+np.arange(600)*.02]
    assert len(tc.split_di3_epochs(e))==2
    s=tc.check_epochs(e,.02)
    assert sum(x['irregular_intervals'] for x in s)==0
    assert sum(x['n_pulses'] for x in s)==1400


def test_nominal_period_needed_for_lag_metric():
    e=np.arange(100)*.020
    s=tc.check_epochs(e,np.nan)[0]
    assert np.isnan(s['mean_anchored_discrepancy_us'])
    assert len(s['time_nominal_anchored'])==0


def test_file_inventory_one_walk(tmp_path):
    root=tmp_path/'session';root.mkdir()
    for dirname,filename in [('x','bonsai_event_log.csv'),('x','SummaryLoCo-v.mat'),
                             ('sBehavior.harp','HARP_df.csv'),('sBehavior.harp','photodiode.pkl')]:
        d=root/dirname;d.mkdir(exist_ok=True);(d/filename).touch()
    v=tc.file_inventory(root)
    assert v['source_glu'].name=='SummaryLoCo-v.mat'
    assert v['harp_csv'].name=='HARP_df.csv'
    assert v['pd_pkl'].name=='photodiode.pkl'


def test_di3_edge_reader_cross_chunk_boundary_and_cache(tmp_path):
    t=np.arange(25)*.01
    flag=((np.arange(25)//2)%2).astype(int)
    f=tmp_path/'HARP_df.csv';pd.DataFrame({'time':t,'DI3':flag}).to_csv(f,index=False)
    got=tc.get_di3_edges(f,tmp_path/'cache',chunksize=7)
    assert np.array_equal(got, t[np.where((flag==1)&(np.r_[False,flag[:-1]==0]))])
    assert np.array_equal(tc.get_di3_edges(f,tmp_path/'cache',chunksize=3),got)


def test_legacy_method_recovered_from_event_log(tmp_path):
    path=tmp_path/'bonsai_event_log.csv'
    pd.DataFrame({'alignment_method':['frame_modclass_affine_anchored_edges']*2}).to_csv(path,index=False)
    qc=tmp_path/'qc';qc.mkdir()
    (qc/'alignment_meta.json').write_text(json.dumps({'alignment_method':'existing_corrected_timestamps'}))
    pathway,method,_=tc.method_provenance(qc,path)
    assert pathway=='legacy_frame_modulo'
    assert method=='frame_modclass_affine_anchored_edges'


def test_one_session_with_metadata_and_registry(monkeypatch,tmp_path):
    sid='863774_2026-08-04_08-12-47'; root=tmp_path/sid;root.mkdir()
    (root/'dendriticVoltageSummary-xyz.mat').touch()
    d=root/'fooBehavior.harp';d.mkdir()
    t=70+np.arange(2200)*.020672
    flags=np.zeros(len(t)*2,int);flags[::2]=1
    df=pd.DataFrame({'DI3':flags,'time':np.repeat(t,2)+np.tile([0,.0005],len(t))})
    df.to_csv(d/'HARP_df.csv',index=False)
    q=root/'analysis'/'qc'/'behavior';q.mkdir(parents=True)
    (q/'di3_pulse_train_detection.json').write_text(json.dumps({
        'n_rising_edges':2200,'median_period_s':.020672,
        'candidate_epochs':[{'accepted':True,'duration_s':float(t[-1]-t[0]),'n_pulses':len(t)}]}))
    class S:
        n_dmds=1
        def get_dmd_metadata(self,j):return {'linesPerCycle':300,'lineRateHz':300/.020672}
        def get_dmd_epoch_metadata(self,j):return [{'nCycles':2210,'available':True,'epochIdx':1}]
    monkeypatch.setattr(tc,'_summarize_acquisition',lambda path,typ:({'summary':S(),'dmds':[{'dmd':1,'metadata':S().get_dmd_metadata(1),'epochs':S().get_dmd_epoch_metadata(1)}]},'ok'))
    row=pd.Series(dict(session_id=sid,subject_id=863774,session_type='A0',session_dir=root,indicator1='ASAP8'))
    r,detail=tc.session_record(row,tmp_path/'cache')
    assert r['clock_relationship']=='direct_dmd_cycle_clock'
    assert abs(r['period_error_ppm'])<1e-3
    assert r['n_di3_edges']==2200
    assert r['source_n_cycles']==2210
    assert r['duration_overhang_s']>0
    assert np.isnan(r['t03_absolute_slap2_harp_lag_ms'])
    assert len(detail['segments'])==1


def test_integer_line_cadence_not_mistaken_for_dmd_cycle(monkeypatch,tmp_path):
    root=tmp_path/'legacy';root.mkdir()
    (root/'SummaryLoCo-test.mat').touch()
    h=root/'fooBehavior.harp';h.mkdir()
    # Measured DI3 one pulse per 150 lines, DMD source cycle is 300 lines.
    T=0.020
    rise=5+np.arange(2600)*T
    signal=np.zeros(5200,int);signal[::2]=1
    pd.DataFrame({'time':np.repeat(rise,2)+np.tile([0,.0005],len(rise)),
                  'DI3':signal}).to_csv(h/'HARP_df.csv',index=False)
    qc=root/'analysis'/'qc'/'behavior';qc.mkdir(parents=True)
    (qc/'di3_pulse_train_detection.json').write_text(json.dumps({
        'n_rising_edges':len(rise),'median_period_s':T,
        'candidate_epochs':[{'accepted':True,'duration_s':rise[-1]-rise[0],'n_pulses':len(rise)}]}))
    class S:
        n_dmds=1
        def get_dmd_metadata(self,j):return {'linesPerCycle':300,'lineRateHz':7500}
        def get_dmd_epoch_metadata(self,j):return [{'nCycles':1400,'epochIdx':1,'available':True}]
    monkeypatch.setattr(tc,'_summarize_acquisition',lambda p,k:({'summary':S(),
         'dmds':[{'dmd':1,'metadata':S().get_dmd_metadata(1),
                  'epochs':S().get_dmd_epoch_metadata(1)}]},'ok'))
    row=pd.Series({'session_id':'legacy','subject_id':803496,'session_type':'B2',
                   'session_dir':root,'indicator1':'iGluSnFR4f'})
    result,detail=tc.session_record(row,tmp_path/'cache')
    assert result['clock_relationship']=='integer_line_cadence_not_dmd_cycle'
    assert result['source_period_ms']==pytest.approx(40)
    assert result['expected_di3_pulse_period_ms']==pytest.approx(20)
    assert result['period_error_ppm']==pytest.approx(0,abs=.005)
    assert result['source_duration_s']==pytest.approx(56)
    assert len(detail['segments'])==1


def test_missing_source_does_not_invent_clock_latency(monkeypatch,tmp_path):
    root=tmp_path/'no_source';root.mkdir()
    row=pd.Series({'session_id':'no_source','subject_id':111,'session_type':'A0',
                   'session_dir':root})
    result,parts=tc.session_record(row,tmp_path/'cache')
    assert result['source_read_status']=='no_source_summary'
    assert np.isnan(result['t03_absolute_slap2_harp_lag_ms'])
    assert not parts['segments']
