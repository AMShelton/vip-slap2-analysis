"""T0.3 hardware clock and clipping evidence; read-only / explicit clock provenance.

HARP DI3 is the only independently timestamped SLAP2 acquisition-cycle signal.
A nominal SLAP2 clock synthesized from metadata and anchored to the first
DI3 pulse is *not* an independently measured SLAP2 clock. Consequently,
this module reports cadence agreement and anchored residuals but never a
measured absolute SLAP2-versus-HARP edge delay.
"""
from __future__ import annotations
import json
import os
import re
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
from vip_slap2_analysis.common.clock_qc import compare_slap2_harp_clock

SESSION_TYPES = {'A0','A1','A2','B0','B1','B2'}
RECOGNIZED = ('SummaryLoCo','dendriticVoltageSummary','bonsai_event_log','HARP_df','photodiode')

def fnum(x):
    try:
        arr=np.asarray(x).squeeze()
        x=float(arr)
        return x if np.isfinite(x) else np.nan
    except (ValueError,TypeError,OverflowError):
        return np.nan

def read_json(p):
    try: return json.loads(Path(p).read_text()) if p is not None and Path(p).is_file() else {}
    except (OSError,ValueError): return {}

def file_inventory(directory: Path) -> dict:
    """One walk per session, rather than recursive search per asset."""
    out={k:[] for k in ['source_glu','source_voltage','bonsai','harp_csv','pd_pkl']}
    directory=Path(directory)
    if not directory.is_dir():return out
    for base,dirs,files in os.walk(directory):
        dirs[:]=[x for x in dirs if x not in ('.git','__pycache__','.ipynb_checkpoints')]
        bp=Path(base)
        for fn in files:
            low=fn.lower();p=bp/fn
            if low.startswith('summaryloco') and low.endswith('.mat'):out['source_glu'].append(p)
            elif low.startswith('dendriticvoltagesummary') and low.endswith('.mat'):out['source_voltage'].append(p)
            elif low.startswith('bonsai_event_log') and low.endswith('.csv'):out['bonsai'].append(p)
            elif low.startswith('harp_df') and low.endswith('.csv'):out['harp_csv'].append(p)
            elif low.startswith('photodiode') and low.endswith('.pkl'):out['pd_pkl'].append(p)
    def order(k,p):
        parts=str(p).lower()
        canonical=2 if ('source_extraction' in parts and k in ('source_glu','source_voltage')) else 0
        prefer=1 if ('experimentsummary' in parts and k=='source_glu') else 0
        try:st=p.stat();mod=st.st_mtime
        except OSError:mod=0
        return (canonical+prefer,mod)
    return {k:(max(v,key=lambda p:order(k,p)) if v else None) for k,v in out.items()}

def get_di3_edges(csv:Path, cache:Path, chunksize:int=350000) -> np.ndarray:
    """Streaming rising edges; cache invalidated on source filesize/mtime."""
    csv=Path(csv);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    st=csv.stat();version=f'{csv.resolve()}|{st.st_size}|{st.st_mtime_ns}|di3_v2'
    stamp=hashlib.sha1(version.encode()).hexdigest()[:20]
    stored=cache/(f'di3_{stamp}.npz')
    if stored.exists():
        with np.load(stored) as z:return z['edges']
    buf=[];prev=False
    for chunk in pd.read_csv(csv,usecols=['DI3','time'],chunksize=chunksize):
        val=chunk['DI3']
        if pd.api.types.is_numeric_dtype(val) or pd.api.types.is_bool_dtype(val):
            state=val.fillna(False).astype(bool).to_numpy()
        else:state=val.astype(str).str.strip().str.lower().isin(['1','1.0','true','t']).to_numpy()
        tt=pd.to_numeric(chunk['time'],errors='coerce').to_numpy(dtype=float)
        preceding=np.r_[prev,state[:-1]]
        idx=np.flatnonzero(state & ~preceding & np.isfinite(tt))
        if len(idx):buf.append(tt[idx])
        if len(state):prev=bool(state[-1])
    edges=np.concatenate(buf) if buf else np.empty(0,float)
    np.savez_compressed(stored,edges=edges)
    return edges

def split_di3_epochs(edges, *, gap_min_s=0.5, multiplier=10):
    edges=np.asarray(edges,float)
    if len(edges)<2:return []
    diff=np.diff(edges)
    good=diff[np.isfinite(diff)&(diff>0)]
    if not len(good):return []
    T=float(np.median(good));jumps=np.flatnonzero(diff>max(gap_min_s,multiplier*T))
    starts=np.r_[0,jumps+1];stops=np.r_[jumps+1,len(edges)]
    return [(int(i),int(j)) for i,j in zip(starts,stops) if j>i]

def clock_period_from_epochs(edges,epochs):
    num=0;elapsed=0.
    for a,b in epochs:
        if b-a>=10 and edges[b-1]-edges[a]>=30:
            num+=b-a-1;elapsed+=float(edges[b-1]-edges[a])
    return elapsed/num if num else np.nan

def _summarize_acquisition(path:Path, type_):
    """Read metadata only through actual repository summary loaders."""
    if not path:return {},'no_source_summary'
    try:
        if type_=='source_voltage':
            from vip_slap2_analysis.voltage.summary import VoltageSummary as Reader
        else:
            from vip_slap2_analysis.glutamate.summary import GlutamateSummary as Reader
        obj=Reader(path)
        try:
            dm=[]
            for i in range(1,obj.n_dmds+1):
                meta=obj.get_dmd_metadata(i)
                epoch=obj.get_dmd_epoch_metadata(i)
                dm.append({'dmd':i,'metadata':meta,'epochs':epoch})
        finally:
            obj.close()  # VoltageSummary supports with; GlutamateSummary does not.
        # Produce flattened scalar report; avoid loading any traces.
        class SummaryProxy:
            n_dmds=len(dm)
            def get_dmd_metadata(self,j):return dm[j-1]['metadata']
            def get_dmd_epoch_metadata(self,j):return dm[j-1]['epochs']
        return {'dmds':dm,'summary':SummaryProxy()},'ok'
    except Exception as e:return {},f'{type(e).__name__}: {e}'

def check_epochs(edges, nominal_period):
    """Per HARP pulse train segment compare independently recorded DI3 to
    nominal SLAP2 cadence, anchored to segment's first DI3 rising edge.
    Does NOT measure absolute edge latency.
    """
    edges=np.asarray(edges,float);groups=split_di3_epochs(edges)
    segments=[]
    for a,b in groups:
        obs=edges[a:b]; count=len(obs)
        T_nom=float(nominal_period) if np.isfinite(nominal_period) and nominal_period>0 else np.nan
        interval=np.diff(obs)
        T_obs=(obs[-1]-obs[0])/(count-1) if count>1 else np.nan
        if np.isfinite(T_nom):
            nominal=obs[0]+np.arange(count)*T_nom
            residual_ms=(obs-nominal)*1e3
            bad=int(np.sum((interval<0.5*T_nom)|(interval>1.5*T_nom)))
        else:
            nominal=np.array([]);residual_ms=np.array([]);bad=np.nan
        segments.append({'start_index':a,'stop_index':b,'n_pulses':count,
                         'start_time':float(obs[0]),'end_time':float(obs[-1]),
                         'period_observed_s':T_obs,'period_nominal_s':T_nom,
                         'irregular_intervals':bad,
                         'time_harp':obs,'time_nominal_anchored':nominal,
                         'residual_ms':residual_ms,
                         'drift_end_ms':float(residual_ms[-1]) if len(residual_ms) else np.nan,
                         'mean_anchored_discrepancy_us':float(np.mean(residual_ms)*1e3) if len(residual_ms) else np.nan,
                         'p95_anchored_discrepancy_us':float(np.percentile(abs(residual_ms),95)*1e3) if len(residual_ms) else np.nan})
    return segments

def method_from_csv(method: str)->str:
    s=str(method).lower()
    if 'frame_modclass' in s:return 'legacy_frame_modulo'
    if 'bv_photodiode' in s:return 'encoded_photodiode'
    if 'existing_corrected' in s:return 'precorrected_without_method'
    return 'unknown'

def method_provenance(qcdir,log):
    meta=read_json(Path(qcdir)/'alignment_meta.json')
    method=str(meta.get('alignment_method',''))
    original=method
    if log and (not method or method_from_csv(method) in ('unknown','precorrected_without_method')):
        try:
            methods=set()
            for ch in pd.read_csv(log,usecols=['alignment_method'],chunksize=120000):
                methods.update(ch['alignment_method'].dropna().astype(str).unique())
                if len(methods)>1:break
            if len(methods)==1:original=methods.pop()
            elif len(methods)>1:return ('unknown','multiple_alignment_methods',meta)
        except (OSError,ValueError,pd.errors.ParserError):pass
    return method_from_csv(original),original,meta

def qc_source_durations(qcdir):
    """Extract saved per-DMD clipping rows and retain their provenance.

    These are the actual processing decisions, not source-clock anchors.
    """
    qc=Path(qcdir)
    candidates=(qc/'voltage'/'voltage_qc_metadata.json',qc/'glutamate'/'glutamate_qc_metadata.json')
    rows=[]
    def visit(obj,source,path='',depth=0):
        if depth>9:return
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k=='source_epoch_qc' and isinstance(v,list):
                    for entry in v:
                        if isinstance(entry,dict):
                            rows.append({**entry,'qc_source_file':str(source),
                                         'qc_location':(path+'/'+k).strip('/')})
                elif isinstance(v,(list,dict)):visit(v,source,path+'/'+k,depth+1)
        elif isinstance(obj,list):
            for i,x in enumerate(obj):
                if isinstance(x,(dict,list)):visit(x,source,path+f'/{i}',depth+1)
    for f in candidates:
        x=read_json(f)
        if x:visit(x,f)
    # Some processed QC JSONs repeat the same source_epoch_qc under both
    # session_metadata/epoch_integrity/per_dmd and dmd_summary. Avoid treating
    # these copies as distinct acquisition epochs.
    chosen={}
    for entry in rows:
        loc=str(entry.get('qc_location',''))
        match=re.search(r'/DMD(\d+)/', '/'+loc+'/')
        dmd=match.group(1) if match else loc
        key=(str(entry.get('qc_source_file')),dmd,int(fnum(entry.get('source_epoch_index'))) if np.isfinite(fnum(entry.get('source_epoch_index'))) else loc)
        previous=chosen.get(key)
        if previous is None or ('session_metadata/epoch_integrity/' in loc and
                                'session_metadata/epoch_integrity/' not in str(previous.get('qc_location',''))):
            chosen[key]=entry
    return list(chosen.values())

def session_record(row, cache_dir:Path, *, read_di3=True):
    """Fast per-session collection. Registry row is the authority for session identity.

    A single file walk avoids costly repeated session-registry recursive globbing.
    """
    sid=str(row['session_id']); base=Path(row['session_dir'])
    d={'session_id':sid,'subject_id':row.get('subject_id'),
       'session_type':str(row.get('session_type','')),
       'indicator1':str(row.get('indicator1','')),
       'indicator2':str(row.get('indicator2','')),
       'session_dir':str(base),
       'source_period_ms':np.nan,'source_n_cycles':np.nan,'source_n_epochs':np.nan,
       'linesPerCycle':np.nan,'lineRateHz':np.nan,'reference_dmd':np.nan,
       'effective_lines_per_di3_pulse':np.nan,'cadence_minus_cycle_lines':np.nan,
       'di3_clock_lines_per_pulse':np.nan,'expected_di3_pulse_period_ms':np.nan}
    paths=file_inventory(base)
    d.update({f'path_{k}':str(v) if v else '' for k,v in paths.items()})
    qcdir=base/'analysis'/'qc'
    path,method,meta=method_provenance(qcdir/'behavior',paths['bonsai'])
    d.update(alignment_pathway=path,alignment_method=method,
             bv_harp_offset_ms=1000*fnum(meta.get('intercept')),
             bv_harp_interval_rmse_ms=1000*fnum(meta.get('edge_rmse_intervals')),
             bv_harp_matched_edges=fnum(meta.get('matched_edges')),
             display_rate_hz=fnum(meta.get('display_rate_est_hz')))
    # Prefer voltage if it exists, otherwise GIAnT / glutamate metadata.
    preferred='source_voltage' if paths['source_voltage'] else 'source_glu'
    source,status=_summarize_acquisition(paths[preferred],preferred)
    d['source_read_status']=status;d['source_class']=preferred
    di3=np.array([],dtype=float)
    if read_di3 and paths['harp_csv']:
        try:di3=get_di3_edges(paths['harp_csv'],Path(cache_dir)/'edges')
        except Exception as e:d['di3_read_error']=f'{type(e).__name__}: {e}'
    d['n_di3_edges']=len(di3)
    epochs=split_di3_epochs(di3)
    duration=float(sum(di3[b-1]-di3[a] for a,b in epochs)) if epochs else np.nan
    d['harp_di3_duration_s']=duration
    if len(di3)>=2:
        dt=np.concatenate([np.diff(di3[a:b]) for a,b in epochs if b-a>1])
        dt=dt[(dt>0)&np.isfinite(dt)]
        d['di3_period_ms']=1e3*clock_period_from_epochs(di3,epochs)
        d['di3_period_median_ms']=1e3*np.median(dt) if len(dt) else np.nan
        d['di3_interval_p95_jitter_us']=1e6*np.percentile(abs(dt-np.median(dt)),95) if len(dt) else np.nan
    else:d['di3_period_ms']=np.nan;d['di3_period_median_ms']=np.nan;d['di3_interval_p95_jitter_us']=np.nan
    if source:
        try:
            cmp=compare_slap2_harp_clock(source['summary'],behavior_qc_dir=qcdir/'behavior',harp_df_csv=None)
            d['clock_relationship']=cmp.get('relationship','unresolved')
            d['reference_dmd']=cmp.get('reference_dmd')
            # When the saved DI3 QC JSON is absent, classify against the actual
            # HARP edge train already read above (no second CSV scan).
            if cmp.get('reference_dmd') is None and np.isfinite(fnum(d.get('di3_period_ms'))):
                choices=[]
                for item in source['dmds']:
                    md=item['metadata'];ep=item['epochs']
                    lr=fnum(md.get('lineRateHz'))
                    lc=fnum(md.get('linesPerCycle'))
                    if not np.isfinite(lr) and ep:lr=fnum(ep[0].get('metadata',{}).get('lineRateHz'))
                    if not np.isfinite(lc) and ep:lc=fnum(ep[0].get('linesPerCycle'))
                    if np.isfinite(lr) and np.isfinite(lc) and lr>0 and lc>0:
                        choices.append((abs((d['di3_period_ms']/1000)/(lc/lr)-1),item['dmd'],lr,lc,ep))
                if choices:
                    _,dm,lr,lc,ep=min(choices)
                    d.update(reference_dmd=dm,lineRateHz=lr,linesPerCycle=lc,
                             source_period_ms=1000*lc/lr,
                             source_n_cycles=sum(int(z.get('nCycles') or 0) for z in ep),
                             source_n_epochs=len(ep))
                    effective=lr*d['di3_period_ms']/1000
                    d['effective_lines_per_di3_pulse']=effective
                    d['cadence_minus_cycle_lines']=effective-lc
                    d['di3_clock_lines_per_pulse']=round(effective)
                    if abs(effective-lc)<.15:d['clock_relationship']='direct_dmd_cycle_clock'
                    elif abs(effective-round(effective))<.15:d['clock_relationship']='integer_line_cadence_not_dmd_cycle'
                    else:d['clock_relationship']='noninteger_or_unresolved_cadence'
            d['effective_lines_per_di3_pulse']=cmp.get('effective_lines_per_harp_pulse')
            d['cadence_minus_cycle_lines']=cmp.get('line_offset_from_reference_cycle')
            d['di3_clock_lines_per_pulse']=cmp.get('nearest_integer_line_cadence')
            ref=next((x for x in cmp.get('dmds',[]) if x['dmd']==cmp.get('reference_dmd')),None)
            if ref:
                d['source_period_ms']=fnum(ref.get('expected_cycle_period_s'))*1000
                d['linesPerCycle']=ref.get('lines_per_cycle')
                d['lineRateHz']=ref.get('line_rate_hz')
                d['source_n_cycles']=ref.get('source_total_cycles')
                d['source_n_epochs']=ref.get('n_source_epochs')
            d['source_dmds_json']=json.dumps([{'dmd':x['dmd'],'line_rate_hz':x['line_rate_hz'],
                'lines_per_cycle':x['lines_per_cycle'],'source_total_cycles':x['source_total_cycles']}
                for x in cmp.get('dmds',[])])
        except Exception as e:d['clock_comparison_error']=str(e)
    if 'clock_relationship' not in d:d['clock_relationship']='unavailable'
    source_cycle_s=fnum(d.get('source_period_ms'))/1000
    # Early sessions can expose a different integer-line cadence on DI3.
    # Do not force one DI3 rise == one DMD cycle in these recordings.
    nominal_di3_s=source_cycle_s
    if d.get('clock_relationship')=='integer_line_cadence_not_dmd_cycle':
        nlines=fnum(d.get('di3_clock_lines_per_pulse'))
        line_rate=fnum(d.get('lineRateHz'))
        nominal_di3_s=nlines/line_rate if np.isfinite(nlines) and np.isfinite(line_rate) and line_rate>0 else np.nan
    d['expected_di3_pulse_period_ms']=nominal_di3_s*1000
    segs=check_epochs(di3,nominal_di3_s)
    d['di3_n_epochs']=len(segs)
    d['di3_n_irregular_intervals']=int(sum(s['irregular_intervals'] for s in segs if np.isfinite(s['irregular_intervals'])))
    d['relative_anchored_drift_end_ms']=max((abs(s['drift_end_ms']) for s in segs if np.isfinite(s['drift_end_ms'])),default=np.nan)
    d['relative_anchored_discrepancy_mean_us']=float(np.mean([s['mean_anchored_discrepancy_us'] for s in segs if np.isfinite(s['mean_anchored_discrepancy_us'])])) if any(np.isfinite(s['mean_anchored_discrepancy_us']) for s in segs) else np.nan
    d['relative_anchored_discrepancy_p95_us']=float(np.mean([s['p95_anchored_discrepancy_us'] for s in segs if np.isfinite(s['p95_anchored_discrepancy_us'])])) if any(np.isfinite(s['p95_anchored_discrepancy_us']) for s in segs) else np.nan
    if np.isfinite(d['di3_period_ms']) and np.isfinite(nominal_di3_s) and nominal_di3_s>0:
        d['period_error_ppm']=1e6*(d['di3_period_ms']/1000/nominal_di3_s-1)
    else:d['period_error_ppm']=np.nan
    d['source_duration_s']=fnum(d.get('source_n_cycles'))*source_cycle_s if np.isfinite(source_cycle_s) else np.nan
    d['source_duration_basis']='raw_SLAP2_cycle_metadata' if np.isfinite(d['source_duration_s']) else 'unavailable'
    saved_clipping=qc_source_durations(qcdir)
    if not np.isfinite(d['source_duration_s']) and saved_clipping:
        # Only one DMD: avoid summing DMD1 and DMD2 durations together.
        paths=sorted({str(r['qc_location']).split('/source_epoch_qc')[0] for r in saved_clipping})
        chosen=paths[0]
        raw=[fnum(x.get('source_duration_s')) for x in saved_clipping
             if str(x['qc_location']).startswith(chosen+'/source_epoch_qc')]
        raw=[x for x in raw if np.isfinite(x)]
        if raw:
            d['source_duration_s']=float(np.sum(raw))
            d['source_duration_basis']='saved_postprocessing_epoch_qc'
    d['duration_overhang_s']=d['source_duration_s']-duration if np.isfinite(d['source_duration_s']) and np.isfinite(duration) else np.nan
    d['source_epochs_metadata']=json.dumps(source.get('dmds',[]),default=str) if source else ''
    d['saved_clipping_rows_json']=json.dumps(saved_clipping,default=str)
    d['t03_absolute_slap2_harp_lag_ms']=np.nan
    d['t03_absolute_lag_status']='UNIDENTIFIABLE: DI3 is the only HARP-anchored SLAP2 edge clock'
    return d,dict(di3_edges=di3,segments=segs,source=source)

def collect_sessions(summary_xlsx,cache_dir,*, session_ids=None,session_types=SESSION_TYPES,
                     force=False):
    """Registry once, per-session disk cache once. Never read physiology arrays."""
    reg=VIPSessionRegistry.from_excel(summary_xlsx)
    rows=reg.sessions_df.copy()
    if session_types:
        rows=rows.loc[rows['session_type'].astype(str).str.upper().isin({x.upper() for x in session_types})]
    if session_ids is not None:
        selected=set(map(str,session_ids));rows=rows.loc[rows['session_id'].astype(str).isin(selected)]
        missing=selected-set(rows['session_id'].astype(str))
        if missing:raise KeyError(f'Not in registry after session-type selection: {sorted(missing)}')
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    records=[];detail={}
    for i,(_,r) in enumerate(rows.iterrows(),1):
        sid=str(r['session_id']); cachefile=cache_dir/f'{sid}_clock.json'
        if cachefile.is_file() and not force:
            try:
                records.append(json.loads(cachefile.read_text()))
                continue
            except (OSError,ValueError):pass
        print(f'[{i}/{len(rows)}] {sid}',flush=True)
        try:
            record,parts=session_record(r,cache_dir)
            records.append(record);detail[sid]=parts
            cachefile.write_text(json.dumps(record,default=str,indent=2))
        except Exception as e:
            records.append(dict(session_id=sid,session_type=str(r.get('session_type','')),
                                collection_error=f'{type(e).__name__}: {e}'))
    return pd.DataFrame(records),detail

def load_session_details(session_id,session_df,cache_dir):
    """Lazy single-session pulse metadata for plots (uses cached DI3 edges)."""
    r=session_df.loc[session_df['session_id'].astype(str)==str(session_id)]
    if r.empty:raise KeyError(session_id)
    from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
    row=r.iloc[0]
    record,data=session_record(row,cache_dir)
    return record,data


def harp_pd_di3_phase(pd_pickle, di3_edges):
    """Real measured HARP photodiode transition phase within the DI3 cycle.

    Return a CONDITIONAL native-HARP-clock measurement only when the raw
    photodiode index and raw DI3 timestamp intervals overlap. This is
    stimulus-to-nearest-acquisition-CYCLE phase, not a synchronization error.
    Do not shift either signal to make it look aligned.
    """
    from vip_slap2_analysis.metadata.t03_session_sync import _load_pd_local, _pd_edges_local
    blank=dict(status='unavailable',phase_ms=np.array([]),event_time_s=np.array([]),
               phase_median_ms=np.nan,phase_p95_abs_ms=np.nan,n_events=0)
    pulses=np.asarray(di3_edges,float)
    if len(pulses)<3 or not pd_pickle or not Path(pd_pickle).is_file():return blank
    df=_load_pd_local(Path(pd_pickle));raw=np.asarray(df.index,float)
    if len(raw)<4 or np.any(np.diff(raw)<=0):return dict(blank,status='invalid_photodiode_index')
    if min(raw[-1],pulses[-1]) <= max(raw[0],pulses[0]):
        return dict(blank,status='pd_di3_native_clock_origins_do_not_overlap')
    raw_edges,_=_pd_edges_local(df)
    pd_abs=raw[0]+np.asarray(raw_edges,float)
    pstep=np.diff(pulses);pstep=pstep[(pstep>0)&(pstep<.5)]
    if not len(pstep):return blank
    median=float(np.median(pstep))
    inside=pd_abs[(pd_abs>=pulses[0])&(pd_abs<=pulses[-1])]
    idx=np.searchsorted(pulses,inside)
    a=np.clip(idx-1,0,len(pulses)-1);b=np.clip(idx,0,len(pulses)-1)
    best=np.where(abs(inside-pulses[a])<=abs(inside-pulses[b]),a,b)
    delta=inside-pulses[best]
    keep=abs(delta)<.75*median
    phase=1e3*delta[keep]
    if not len(phase):return dict(blank,status='no_matched_events_inside_di3')
    return dict(status='conditional_shared_native_HARP_clock',phase_ms=phase,
                event_time_s=inside[keep],phase_median_ms=float(np.median(phase)),
                phase_p95_abs_ms=float(np.percentile(abs(phase),95)),n_events=len(phase))
