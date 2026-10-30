"""Read-only, registry-driven T0.3 source/HARP acquisition-order audit.

This is a *wall-clock candidate* audit, not hardware timestamp registration.
Session acquisition IDs and launcher timestamps are local PC-clock readings.
Recorded HARP DI3 pulses can constrain pulse counts, not the unknown initial
source-cycle integer by themselves. All inferred offsets have uncertainty.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SESSION_RE = re.compile(r"\b(\d{6}_20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\b")
META_NAME = re.compile(r"acquisition_(?:(20\d{6})_)?(\d{6})_DMD1\.meta$", re.I)
CREATED_RE = re.compile(r"Created on: ([A-Za-z]{3} [A-Za-z]{3}\s+\d+ \d\d:\d\d:\d\d \d{4})")
LOG_STAMP = re.compile(r"^(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)")
HARP_FOLDER_STAMP = re.compile(r"(?:^|[\\/])(?:\d{6}_)?(20\d{6}T\d{6})(?:[\\/]|$)")
DI3_QC_NAMES = ("di3_pulse_train_detection.json",)


def _timestamp(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace('Z', '+00:00')).replace(tzinfo=None)
    except ValueError:
        return None


def _from_path_stamp(p: Path):
    m = META_NAME.search(p.name)
    if not m:
        return None
    date = m.group(1)
    if date is None:
        for part in p.parts:
            mm = SESSION_RE.search(part)
            if mm:
                date = mm.group(1).split('_')[1].replace('-', '')
                break
    if not date:
        return None
    return datetime.strptime(date + m.group(2), '%Y%m%d%H%M%S')


def read_slap2_meta(p: Path) -> dict[str, Any]:
    """Read first 150 bytes plus tiny HDF5 datasets; never load raw samples."""
    p = Path(p)
    result = {'meta_file':str(p), 'meta_filename_time':None,
              'meta_header_time':None, 'meta_duration_s':None,
              'meta_cycles_dmd1':None, 'meta_period_s':None,
              'meta_line_period_s':None}
    start = _from_path_stamp(p)
    if start: result['meta_filename_time'] = start.isoformat()
    try:
        with p.open('rb') as f:
            header = f.read(160).decode('ascii', errors='replace')
        m = CREATED_RE.search(header)
        if m:
            result['meta_header_time'] = datetime.strptime(m.group(1), '%a %b %d %H:%M:%S %Y').isoformat()
    except (OSError, ValueError): pass
    try:
        import h5py
        with h5py.File(p, 'r') as f:
            def scalar(key):
                if key not in f:return None
                v = np.asarray(f[key][()]).reshape(-1)
                if not v.size:return None
                x=float(v[0]);return x if np.isfinite(x) else None
            result['meta_duration_s'] = scalar('acqDuration_s')
            result['meta_cycles_dmd1'] = scalar('acqDurationCycles')
            result['meta_line_period_s'] = scalar('linePeriod_s')
            if result['meta_duration_s'] and result['meta_cycles_dmd1']:
                result['meta_period_s'] = result['meta_duration_s']/result['meta_cycles_dmd1']
    except Exception as exc:
        result['meta_read_warning'] = f'{type(exc).__name__}: {str(exc)[:150]}'
    return result


def read_behavior_launcher(log: Path) -> dict[str, Any]:
    """Select actual workflow launch and HARP data-folder timestamp; exclude launcher UI start."""
    result={'behavior_launcher_log':str(log), 'workflow_launch_time':None,
            'harp_folder_time':None,'workflow_end_time':None,'harp_folder_evidence':None}
    try: lines=Path(log).read_text(encoding='utf-8-sig',errors='replace').splitlines()
    except OSError:return result
    harp_stamps=[]
    for line in lines:
        m=LOG_STAMP.match(line)
        if not m:continue
        stamp = _timestamp(m.group(1))
        if 'Starting Bonsai workflow attempt' in line or 'Starting Bonsai workflow:' in line:
            if result['workflow_launch_time'] is None:
                result['workflow_launch_time'] = stamp.isoformat()
        if ('Bonsai process' in line and 'exited' in line.lower()) or 'Saved end_state' in line:
            result['workflow_end_time'] = stamp.isoformat()
        if ('.harp' in line.lower()) and any(k in line.lower() for k in ('selected harp folder', 'found', 'renamed', 'moved harp folder')):
            for val in HARP_FOLDER_STAMP.findall(line):
                try:harp_stamps.append((datetime.strptime(val, '%Y%m%dT%H%M%S'),line[:260]))
                except ValueError:pass
    if harp_stamps:
        # Prefer a HARP recording-directory time occurring near the Bonsai workflow launch.
        w=_timestamp(result['workflow_launch_time'])
        candidates=sorted(harp_stamps,key=lambda x:abs((x[0]-w).total_seconds()) if w else 0)
        result['harp_folder_time'] = candidates[0][0].isoformat()
        result['harp_folder_evidence'] = candidates[0][1]
    return result


def _list_session_files(session_dir:Path) -> dict[str,list[Path]]:
    """One limited walk per session; skip raw array directories and processed datasets."""
    out={'meta':[], 'behavior_launcher':[], 'qc':[], 'di3_npz':[], 'harp_csv':[]}
    session_dir=Path(session_dir)
    if not session_dir.is_dir():return out
    # Asset paths may be nested in a duplicated session folder; walk once.
    for root, dirs, files in os.walk(session_dir):
        dirs[:]=[d for d in dirs if d.lower() not in {
            '.git','__pycache__','.ipynb_checkpoints','dynamic_data','static_data',
            'source_extraction','derived','figures','plots','reference_stack','extraction'}]
        base=Path(root)
        pathstr=str(base).lower().replace('\\','/')
        for fn in files:
            low=fn.lower();p=base/fn
            if META_NAME.search(fn) and ('slap2' in pathstr or 'dynamic_data' in pathstr):out['meta'].append(p)
            elif low=='launcher.log' and ('/behavior/launcher_metadata' in pathstr or '/behavior/launcher' in pathstr):out['behavior_launcher'].append(p)
            elif low in DI3_QC_NAMES:out['qc'].append(p)
            elif low.endswith('.npz') and 'di3' in low:out['di3_npz'].append(p)
            elif low.startswith('harp_df') and low.endswith('.csv'):out['harp_csv'].append(p)
        # dynamic_data contains .meta alongside >100 .dat chunks. List only meta
        # Since dynamic_data is skipped, collect top-level .meta without recursive scan.
        candidate = base/'dynamic_data'
        if candidate.is_dir():
            out['meta'] += [q for q in candidate.glob('acquisition_*_DMD1.meta')]
    return out


def _read_qc_counts(files:list[Path]):
    """Only accept established detection metadata, not generic estimates."""
    for p in sorted(files,reverse=True):
        try:
            j=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        # Different QC exporter versions use different names.
        def rec_find(obj, keys, depth=0):
            if depth>5:return None
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if k.lower() in keys and isinstance(v,(int,float)):return float(v)
                for v in obj.values():
                    got=rec_find(v,keys,depth+1)
                    if got is not None:return got
            return None
        count=rec_find(j,{'n_rising_edges','rising_edge_count','n_edges','n_pulses','rising_edges_count','pulse_count'})
        duration=rec_find(j,{'duration_s','span_s','recording_duration_s','di3_duration_s','epoch_duration_s'})
        period=rec_find(j,{'median_period_s','median_interval_s','median_rising_edge_period_s','period_s'})
        if count is not None:return count,duration,period,str(p)
    return None,None,None,None


def first_di3_delay_from_csv(path:Path, *, max_rows:int=500_000) -> dict[str, Any]:
    """Optional direct evidence HARP records an initial DI3-low interval.

    The HARP_df.csv `time` is used only for relative duration within that
    recording. No PC-clock correspondence is inferred.
    """
    result={'harp_first_di3_delay_s':None,'harp_initial_di3_state':None,
            'harp_csv_evidence':str(path),'harp_csv_warning':None}
    try:
        seen=0;previous=None;start=None
        for chunk in pd.read_csv(path,usecols=['time','DI3'],chunksize=20_000):
            if not len(chunk):continue
            t=pd.to_numeric(chunk['time'],errors='coerce').to_numpy(float)
            col=chunk['DI3']
            state=(col.fillna(0).astype(bool).to_numpy() if pd.api.types.is_numeric_dtype(col)
                   else col.astype(str).str.strip().str.lower().isin(['true','1','1.0']).to_numpy())
            if start is None:
                good=np.flatnonzero(np.isfinite(t))
                if not len(good):continue
                start=float(t[good[0]])
                previous=bool(state[good[0]])
                result['harp_initial_di3_state']=previous
            for i in range(len(state)):
                if bool(state[i]) and not previous and np.isfinite(t[i]):
                    result['harp_first_di3_delay_s']=float(t[i]-start)
                    return result
                previous=bool(state[i])
            seen+=len(state)
            if seen>=max_rows:break
        result['harp_csv_warning']='No rising edge in first checked HARP rows.'
    except Exception as exc:
        result['harp_csv_warning']=f'{type(exc).__name__}: {str(exc)[:200]}'
    return result


def classify_pair(source:dict, behavior:dict, *, cross_pc_bound_s:float=3,
                  meta_hardware_delay_bound_s:float=3,
                  harp_folder_bound_s:float=2,
                  launcher_to_harp_bound_s:float=6,
                  di3_count:float|None=None,
                  measured_di3_period_s:float|None=None) -> dict[str, Any]:
    """Infer start-order conservatively using bounded software-clock evidence.

    Bounds are analyst-configurable assumptions, NOT calibrated uncertainties.
    A session without an independent anchor must never be marked sub-frame PASS.
    """
    src=_timestamp(source.get('meta_header_time') or source.get('meta_filename_time'))
    folder=_timestamp(behavior.get('harp_folder_time'))
    workflow=_timestamp(behavior.get('workflow_launch_time'))
    harp=folder or workflow
    lag=(harp-src).total_seconds() if src and harp else None
    bound=(float(cross_pc_bound_s)+float(meta_hardware_delay_bound_s)
           +(float(harp_folder_bound_s) if folder else float(launcher_to_harp_bound_s))+1)
    if lag is None:
        classification='missing_timing_evidence';lo=hi=None
    else:
        lo=lag-bound;hi=lag+bound
        classification='slap2_first' if lo>0 else 'harp_first' if hi<0 else 'ambiguous_overlap'
    period=measured_di3_period_s or source.get('meta_period_s')
    result={
      'source_meta_time':src.isoformat() if src else None,
      'harp_anchor_time':harp.isoformat() if harp else None,
      'harp_anchor_type':('harp_recording_folder' if folder else ('workflow_launch' if workflow else 'missing')),
      'estimated_harp_minus_source_s':lag,
      'assumed_timing_bound_s':bound if lag is not None else None,
      'estimated_delta_lower_s':lo, 'estimated_delta_upper_s':hi,
      'order_classification':classification,
      'source_cycles_configured':source.get('meta_cycles_dmd1'),
      'source_duration_configured_s':source.get('meta_duration_s'),
      'cycle_period_estimated_s':period,
      'harp_di3_pulse_count':di3_count,
      'unrecorded_cycle_count_equivalent':None,
      'di3_dmd1_cadence_verified':False,
      'initial_cycle_offset_estimate':None,
      'initial_cycle_offset_lower':None,
      'initial_cycle_offset_upper':None,
      'final_cycle_overhang_estimate':None,
      'alignment_evidence_status':'NOT_HARDWARE_VERIFIED',
    }
    total=source.get('meta_cycles_dmd1')
    nominal_period=source.get('meta_period_s')
    cadence_verified=(measured_di3_period_s is not None and nominal_period is not None
        and np.isfinite(float(measured_di3_period_s)) and float(measured_di3_period_s)>0
        and abs(float(measured_di3_period_s)/float(nominal_period)-1)<0.01)
    result['di3_dmd1_cadence_verified']=bool(cadence_verified)
    if total is not None and di3_count is not None and cadence_verified:
        # Approximation only: rising edges vs configured cycles differ at endpoints.
        missing=max(0,int(round(float(total)-float(di3_count))))
        result['unrecorded_cycle_count_equivalent']=missing
    if lag is not None and period and period>0:
        low_idx=max(0,int(math.floor(max(0,lo)/period))-1)
        high_idx=max(0,int(math.ceil(max(0,hi)/period))+1)
        est=max(0,int(round(max(0,lag)/period)))
        missing=result['unrecorded_cycle_count_equivalent']
        if missing is not None:
            high_idx=min(high_idx,missing+2)
            est=min(est,missing+2)
        result['initial_cycle_offset_estimate']=est
        result['initial_cycle_offset_lower']=min(low_idx,high_idx)
        result['initial_cycle_offset_upper']=high_idx
        if missing is not None:
            result['final_cycle_overhang_estimate']=max(0, missing-est)
    result['reprocess_disposition']=(
       'REVIEW_POTENTIAL_HEAD_TRIM' if classification=='slap2_first' else
       'REVIEW_UNKNOWN_HEAD_TRIM' if classification=='ambiguous_overlap' else
       'NO_HEAD_TRIM_SUGGESTED_BY_WALL_CLOCK' if classification=='harp_first' else
       'NEEDS_TIMING_METADATA')
    return result


def audit_session(session_id:str,session_dir:Path, *, inspect_harp_csv:bool=False, **bounds):
    inventory=_list_session_files(Path(session_dir))
    out={'session_id':str(session_id),'session_dir':str(session_dir),
         'meta_files_found':len(inventory['meta']),
         'behavior_launcher_logs_found':len(inventory['behavior_launcher']),
         'di3_qc_files_found':len(inventory['qc'])}
    behavior={}
    if inventory['behavior_launcher']:
        candidates=[read_behavior_launcher(x) for x in inventory['behavior_launcher']]
        # Prefer a log that actually launched behavior workflow
        behavior=max(candidates,key=lambda x:int(bool(x.get('workflow_launch_time')))+int(bool(x.get('harp_folder_time'))))
    srcs=[read_slap2_meta(m) for m in inventory['meta']]
    candidates=[x for x in srcs if x.get('meta_duration_s') and x['meta_duration_s']>300]
    if not candidates:candidates=srcs
    ref=_timestamp(behavior.get('harp_folder_time') or behavior.get('workflow_launch_time'))
    if ref:
        source=min(candidates,key=lambda x:abs((_timestamp(x.get('meta_header_time') or x.get('meta_filename_time'))-ref).total_seconds()) if _timestamp(x.get('meta_header_time') or x.get('meta_filename_time')) else 1e12) if candidates else {}
    else:source=max(candidates,key=lambda x: x.get('meta_duration_s') or 0) if candidates else {}
    count, duration, period, qc_path=_read_qc_counts(inventory['qc'])
    out.update({**source,**behavior,'di3_qc_path':qc_path,'di3_recording_duration_s':duration})
    out.update(classify_pair(source, behavior, di3_count=count,
                             measured_di3_period_s=period, **bounds))
    if inspect_harp_csv and inventory['harp_csv']:
        evidence=first_di3_delay_from_csv(inventory['harp_csv'][0])
        out.update(evidence)
        # A substantial initial DI3-low interval is direct support for HARP
        # recording before the source cycle clock begins; this only changes the
        # hardware-order evidence, not absolute source-cycle registration.
        period_s=out.get('cycle_period_estimated_s')
        delay=evidence.get('harp_first_di3_delay_s')
        if delay is not None and period_s and delay>max(1.0,10*period_s):
            out['hardware_order_evidence']='harp_before_di3_clock_became_active'
            out['order_classification']='harp_first'
            out['reprocess_disposition']='NO_HEAD_TRIM_SUGGESTED_BY_DI3_ONSET'
        elif delay is not None:
            out['hardware_order_evidence']='DI3_active_at_or_near_harp_onset'
    return out


def manifest_coverage(manifest:Path) -> pd.DataFrame:
    """Manifest-only census. File presence is not a timing determination."""
    by={}
    with Path(manifest).open(encoding='utf-8-sig',newline='') as f:
        for path,typ,*_ in csv.reader(f,delimiter='\t'):
            m=SESSION_RE.search(path)
            if not m:continue
            sid=m.group(1)
            # Count only canonical subject-folder files, not analysis figure references.
            stem=path.replace('\\','/')
            rec=by.setdefault(sid,dict(session_id=sid,source_dmd1_meta=False,
                behavior_launcher_log=False,slap2_launcher_log=False,harp_behavior_bin=False,
                di3_qc=False,source_raw_chunks=False))
            if typ!='file':continue
            lo=stem.lower()
            if META_NAME.search(Path(stem).name):rec['source_dmd1_meta']=True
            if lo.endswith('/launcher_metadata/launcher.log'):
                if '/behavior/' in lo:rec['behavior_launcher_log']=True
                if '/slap2/' in lo:rec['slap2_launcher_log']=True
            if lo.endswith('/behavior_32.bin'):rec['harp_behavior_bin']=True
            if lo.endswith('/di3_pulse_train_detection.json'):rec['di3_qc']=True
            if '-cycle-' in lo and lo.endswith('.dat'):rec['source_raw_chunks']=True
    return pd.DataFrame(by.values()).sort_values('session_id')


def audit_registry(summary_xlsx:Path, output_dir:Path, session_types=None, inspect_harp_csv:bool=False, **bounds):
    from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
    registry=VIPSessionRegistry.from_excel(summary_xlsx)
    df=registry.sessions_df
    if session_types:
        df=df.loc[df['session_type'].astype(str).isin(session_types)].copy()
    # This intentionally bypasses resolve_assets(), which runs several recursive
    # globs per session; use a single limited walk for this audit instead.
    rows=[]
    for rec in df.to_dict('records'):
        sid=str(rec['session_id']);root=rec.get('session_dir')
        if root is None or pd.isna(root):continue
        evidence=audit_session(sid,Path(root),inspect_harp_csv=inspect_harp_csv,**bounds)
        for k in ('session_type','indicator1','session_date','paradigm'):
            if k in rec:evidence[k]=str(rec[k])
        rows.append(evidence)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    out=pd.DataFrame(rows)
    out.to_csv(output_dir/'t03_start_order_evidence.csv',index=False)
    params=dict(summary_xlsx=str(summary_xlsx), session_types=list(session_types or []),
                inspect_harp_csv=inspect_harp_csv,**bounds,
                warning='These are bounded wall-clock estimates; not absolute hardware-cycle registrations.')
    (output_dir/'t03_audit_parameters.json').write_text(json.dumps(params,indent=2))
    return out
