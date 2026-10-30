"""T0.3 session synchronization measurements (READ ONLY).

Three time domains are kept separate:
  BV log -> HARP PD: estimated mapping and held-out predictive residual.
  HARP PD -> HARP DI3: measured event-to-nearest-acquisition-pulse phase.
  SLAP2 source lines -> HARP DI3: only validated with external independently
      established correspondence (never inferred from a generated timebase).

The DI3 clock is not necessarily a sample clock for DMD2 or GIAnT analyzeHz.
HARP PD and DI3 absolute indexes must share the same HARP timestamp origin.
Nothing here modifies original extraction, physiology, or QC assets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from vip_slap2_analysis.io.session_registry import VIPSessionRegistry


def number(value: Any) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else float('nan')
    except (TypeError, ValueError, OverflowError):
        return float('nan')


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    except (OSError, ValueError):
        return {}



def _method_pathway(method: str) -> str:
    """Name the ORIGINAL BonVision-to-HARP alignment method, not a reuse operation."""
    value = str(method or '').strip().lower()
    if 'bv_photodiode' in value:
        return 'encoded_photodiode'
    if 'frame_modclass' in value:
        return 'legacy_frame_modulo'
    if 'existing_corrected' in value:
        return 'precorrected_without_method'
    return 'unknown'


def original_behavior_alignment_method(meta: dict, bonsai_csv: Optional[Path]) -> dict:
    """Recover alignment provenance erased by the 'reuse corrected CSV' code path.

    ``behavior.preprocess`` writes alignment_method=existing_corrected_timestamps
    to QC JSON whenever corrected timestamps are reused, even if the CSV
    preserves the original alignment_method for every event. Prefer the CSV
    method ONLY when the JSON lacks the original method. Detect disagreement
    or mixed methods rather than silently recategorizing affected sessions.
    """
    saved = str(meta.get('alignment_method') or '').strip()
    saved_path = _method_pathway(saved)
    result = dict(alignment_method=saved, alignment_pathway=saved_path,
                  alignment_method_saved=saved,
                  alignment_method_source='alignment_meta.json' if saved else 'unavailable',
                  alignment_reused_corrected=bool(meta.get('reused_existing_corrected_event_log', False))
                      or saved_path == 'precorrected_without_method',
                  alignment_method_conflict='')
    if bonsai_csv is None or not Path(bonsai_csv).is_file():
        return result
    try:
        methods=set()
        # Read only one string column, not the large event-log payload.
        for part in pd.read_csv(bonsai_csv, usecols=['alignment_method'],
                                chunksize=100_000, dtype={'alignment_method':'string'}):
            methods.update(str(x).strip() for x in part['alignment_method'].dropna().unique()
                           if str(x).strip())
            if len(methods)>1:break
    except (ValueError, OSError, pd.errors.ParserError):
        # Legacy *uncorrected* event logs do not have the method column.
        return result
    if len(methods)>1:
        result['alignment_pathway']='unknown'
        result['alignment_method_conflict']='Multiple alignment methods in event log: '+', '.join(sorted(methods))
        return result
    if not methods:
        return result
    csv_method=methods.pop()
    csv_path=_method_pathway(csv_method)
    if saved_path in ('unknown','precorrected_without_method') and csv_path in ('legacy_frame_modulo','encoded_photodiode'):
        result.update(alignment_method=csv_method,alignment_pathway=csv_path,
                      alignment_method_source='bonsai_event_log.csv/alignment_method')
    elif saved_path in ('legacy_frame_modulo','encoded_photodiode') and csv_path in ('legacy_frame_modulo','encoded_photodiode') and saved_path!=csv_path:
        result['alignment_pathway']='unknown'
        result['alignment_method_conflict']=f'QC metadata ({saved}) conflicts with event log ({csv_method})'
    return result


def modalities_for(row: pd.Series, asset) -> str:
    fields = ' '.join(str(row.get(k, '')) for k in ('indicator1', 'indicator2', 'modality', 'session_type')).lower()
    if any(x in fields for x in ('asap', 'voltron', 'voltage')) or asset.get_asset('voltage', 'trace_h5'):
        if 'iglu' in fields:
            return 'glutamate + voltage'
        return 'voltage'
    if 'iglu' in fields and any(x in fields for x in ('rcamp', 'jrg', 'gcamp', 'calcium', 'ca2')):
        return 'glutamate + calcium'
    if 'iglu' in fields:
        return 'glutamate'
    if any(x in fields for x in ('rcamp', 'gcamp', 'jrg', 'calcium')):
        return 'calcium'
    return 'unspecified'


def rising_edges_from_harp_csv(path: Path, chunk_size: int = 300_000) -> np.ndarray:
    """DI3 rising edges in the CSV's original HARP time coordinate.

    Carry digital state across chunks; never infer pulse losses by comparing
    whole-recording pulse counts with different source acquisition windows.
    """
    result = []
    state_prev = False
    for chunk in pd.read_csv(path, usecols=['time', 'DI3'], chunksize=chunk_size):
        if not len(chunk):
            continue
        tt = pd.to_numeric(chunk['time'], errors='coerce').to_numpy(dtype=float)
        vals = chunk['DI3']
        if pd.api.types.is_bool_dtype(vals) or pd.api.types.is_numeric_dtype(vals):
            states = vals.fillna(False).astype(bool).to_numpy()
        else:
            states = vals.astype(str).str.lower().str.strip().isin(('true','t','1','1.0')).to_numpy()
        before = np.r_[state_prev, states[:-1]]
        idx = np.flatnonzero(states & ~before & np.isfinite(tt))
        if len(idx):
            result.append(tt[idx])
        state_prev = bool(states[-1])
    return np.concatenate(result) if result else np.array([], dtype=float)


def di3_clock_diagnostics(edges: np.ndarray, gap_s: float = 0.5) -> dict:
    edges = np.asarray(edges, dtype=float)
    out = dict(di3_n_edges=int(len(edges)), di3_period_ms=np.nan,
               di3_period_mean_ms=np.nan, di3_p95_abs_jitter_us=np.nan,
               di3_n_outlier_intervals=0, di3_n_epoch_gaps=0,
               di3_duration_s=np.nan, di3_n_nonmonotonic=0)
    if len(edges) < 3:
        return out
    delta = np.diff(edges)
    out['di3_n_nonmonotonic'] = int(np.sum(~np.isfinite(delta) | (delta <= 0)))
    out['di3_n_epoch_gaps'] = int(np.sum(delta >= gap_s))
    contiguous = delta[(delta > 0) & np.isfinite(delta) & (delta < gap_s)]
    if not len(contiguous):
        return out
    median = float(np.median(contiguous))
    normal = contiguous[(contiguous > median*.5) & (contiguous < median*1.5)]
    out.update(di3_period_ms=1000*median,
               di3_period_mean_ms=(1000*np.mean(normal) if len(normal) else np.nan),
               di3_p95_abs_jitter_us=(1e6*np.percentile(abs(normal-np.median(normal)),95) if len(normal) else np.nan),
               di3_n_outlier_intervals=int(np.sum((contiguous < median*.5) | (contiguous > median*1.5))),
               di3_duration_s=float(edges[-1]-edges[0]))
    return out


def nearest_pulse_phase(events: np.ndarray, pulses: np.ndarray, max_gap_factor: float = 1.5) -> dict:
    """Signed (event - nearest DI3 rise) with no assumption of sample identity.

    Events outside the pulse recording window, or adjacent to an acquisition-off
    gap, are excluded. This measures phase, NOT a timing registration error.
    """
    events = np.asarray(events, float)
    pulses = np.asarray(pulses, float)
    blank = dict(n=0, median_ms=np.nan, p95_abs_ms=np.nan, p05_ms=np.nan,
                 p95_ms=np.nan, first_quartile_ms=np.nan, last_quartile_ms=np.nan,
                 phase_change_ms=np.nan, time_s=np.array([]), phase_ms=np.array([]))
    if pulses.size < 3 or events.size == 0:
        return blank
    med = np.median(np.diff(pulses)[(np.diff(pulses) > 0) & (np.diff(pulses) < .5)])
    if not np.isfinite(med):
        return blank
    within = events[(np.isfinite(events)) & (events >= pulses[0]) & (events <= pulses[-1])]
    if not len(within):
        return blank
    pos = np.searchsorted(pulses, within)
    lo = np.clip(pos - 1, 0, len(pulses)-1)
    hi = np.clip(pos, 0, len(pulses)-1)
    near = np.where(abs(within-pulses[lo]) <= abs(within-pulses[hi]), lo, hi)
    # Reject edges inside an inter-epoch gap as phase estimates.
    mask = abs(within-pulses[near]) <= max_gap_factor*med/2
    t = within[mask]
    d = 1000*(t-pulses[near[mask]])
    if len(d) == 0:
        return blank
    quarter = max(1, int(np.ceil(len(d)/4)))
    a, b = float(np.median(d[:quarter])), float(np.median(d[-quarter:]))
    return dict(n=int(len(d)),median_ms=float(np.median(d)),p95_abs_ms=float(np.percentile(abs(d),95)),
                p05_ms=float(np.percentile(d,5)), p95_ms=float(np.percentile(d,95)),
                first_quartile_ms=a,last_quartile_ms=b,phase_change_ms=b-a,
                time_s=t,phase_ms=d)


def _load_pd_local(path: Path) -> pd.DataFrame:
    """Analog photodiode pickle loader, matching behavior.preprocess._load_photodiode."""
    df=pd.read_pickle(path)
    if isinstance(df,pd.Series):df=df.to_frame('AnalogInput0')
    if 'AnalogInput0' not in df.columns:df=df.rename(columns={df.columns[0]:'AnalogInput0'})
    df=df.copy();df.index=df.index.astype(float)
    return df


def _pd_edges_local(df: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    """Mirror repo analog photodiode threshold/debounce; no voltage processing.

    This avoids importing optional `harp` binary-reader dependencies when
    the registry already resolves HARP's cached extracted files.
    """
    from scipy.signal import medfilt
    t=np.asarray(df.index,float);t=t-t[0]
    y=np.asarray(df['AnalogInput0'],float)
    y_s=medfilt(y,kernel_size=9) if len(y)>=9 else y.copy()
    threshold=(np.percentile(y_s,95)+np.percentile(y_s,5))/2
    binary=(y_s>threshold).astype(np.int8)
    change=np.diff(binary,prepend=binary[0])
    def debounce(indices):
        tt=t[np.flatnonzero(change==indices)]
        if not len(tt):return tt
        keep=np.r_[True,np.diff(tt)>.005]
        # Match repo's last-kept debounce even for runs of multiple close edges.
        selected=[tt[0]]
        for edge in tt[1:]:
            if edge-selected[-1]>.005:selected.append(edge)
        return np.asarray(selected,float)
    rises=debounce(1);falls=debounce(-1)
    return np.sort(np.r_[rises,falls]),rises


def _load_bv_local(path: Path) -> pd.DataFrame:
    df=pd.read_csv(path)
    df=df.loc[:,~df.columns.astype(str).str.match(r'^Unnamed')].copy()
    req={'Timestamp','Frame','Value'}
    if not req.issubset(df):raise ValueError(f'Bonsai event log missing: {req-set(df.columns)}')
    return df


def _bv_relative(df: pd.DataFrame) -> pd.DataFrame:
    b=df.copy()
    v=b['Value'].astype(str)
    t0=float(b.loc[v.str.lower()=='frame','Timestamp'].iloc[0]) if (v.str.lower()=='frame').any() else float(b['Timestamp'].min())
    state=np.full(len(b),np.nan,dtype=float)
    state[v.str.fullmatch('Photodiode-1',case=False)]=1.
    state[v.str.fullmatch('Photodiode-0',case=False)]=0.
    b['photodiode_state']=pd.Series(state).ffill().to_numpy()
    b['timestamp_bv_rel']=pd.to_numeric(b['Timestamp'],errors='coerce').to_numpy(float)-t0
    return b


def _best_shift_local(bv, hp, max_shift=10):
    da=np.diff(bv); db=np.diff(hp)
    n=min(len(da),len(db),20)
    if n<5:return 0
    da,db=da[:n],db[:n]
    best=(float('inf'),0)
    for shift in range(-max_shift,max_shift+1):
        if shift>=0:a=da[shift:];b=db[:len(a)]
        else:b=db[-shift:];a=da[:len(b)]
        sz=min(len(a),len(b))
        if sz<5:continue
        rmse=float(np.sqrt(np.mean((a[:sz]-b[:sz])**2)))
        if rmse<best[0]:best=rmse,shift
    return int(best[1])


def _shift_local(bv,hp,shift):
    if shift>0:bv=bv[shift:]
    elif shift<0:hp=hp[-shift:]
    n=min(len(bv),len(hp))
    return bv[:n],hp[:n]


def _bv_rises_local(b: pd.DataFrame, hp_rises: np.ndarray) -> Optional[np.ndarray]:
    is_frame=b['Value'].astype(str).str.lower().eq('frame')
    fr=b.loc[is_frame,['timestamp_bv_rel','photodiode_state']]
    state=fr['photodiode_state'].ffill().fillna(0).to_numpy(float)
    t=fr['timestamp_bv_rel'].to_numpy(float)
    ix=np.flatnonzero(np.diff(state,prepend=state[0])==1)
    if not len(ix):return None
    best=None
    for mode in ('previous','midpoint','current'):
        good=ix[ix>0] if mode!='current' else ix
        tt=({'previous':lambda:t[good-1],
             'midpoint':lambda:.5*(t[good-1]+t[good]),
             'current':lambda:t[good]})[mode]()
        if len(tt)<3:continue
        shift=_best_shift_local(tt,hp_rises)
        a,h=_shift_local(tt,hp_rises,shift)
        if len(a)<3:continue
        slope,intercept=np.polyfit(a,h,1)
        rmse=float(np.sqrt(np.mean((slope*a+intercept-h)**2)))
        if best is None or rmse<best[0]:best=(rmse,tt)
    return best[1] if best is not None else None


def photodiode_edges_and_origin(pd_path: Path, harp_csv: Optional[Path]) -> dict:
    """Read measured HARP analog photodiode edges using the repo's detector.

    Do not call data with separately zeroed indexes 'common-clock' unless
    original HARP timestamps establish a shared zero. Full absolute index is
    retained; only subtract a common reference when plotting. Processing the
    photodiode here does not filter or smooth voltage traces.
    """
    pd_df = _load_pd_local(pd_path)
    raw_t = np.asarray(pd_df.index, dtype=float)
    if len(raw_t) < 3 or np.any(np.diff(raw_t) <= 0):
        raise ValueError('Photodiode index not strictly increasing')
    both_rel, rise_rel = _pd_edges_local(pd_df)
    return dict(harp_pd_start_raw=float(raw_t[0]), rises_absolute=rise_rel+raw_t[0],
                edges_absolute=both_rel+raw_t[0], rises_rel=rise_rel, edges_rel=both_rel,
                analog_duration_s=float(raw_t[-1]-raw_t[0]))


def modern_holdout(bonsai_csv: Path, pd_path: Path, stored_intercept_s: float = float('nan')) -> dict:
    """Predict withheld BV photodiode rises from remaining HARP anchors.

    This is out-of-anchor validation, not independent event detection/pairing.
    Reproduces the repo's event parser and photodiode detection conventions
    using cached files, without requiring the optional HARP binary reader.
    """
    b=_bv_relative(_load_bv_local(bonsai_csv))
    pd_df=_load_pd_local(pd_path)
    _,hp=_pd_edges_local(pd_df)
    bv=_bv_rises_local(b,hp)
    blank=dict(status='not_measured',n=0,median_ms=np.nan,p95_abs_ms=np.nan,
               max_abs_ms=np.nan,rmse_ms=np.nan,first_clock_offset_ms=np.nan,
               last_clock_offset_ms=np.nan,clock_offset_change_ms=np.nan,
               time_min=np.array([]),residual_ms=np.array([]),offset_matching_status='unavailable',
               selected_edge_shift=np.nan,n_plausible_shifts=0)
    if bv is None or len(bv)<15 or len(hp)<15:return blank
    # Periodic photodiode trains can pair perfectly at many whole-period shifts.
    # Use the saved preprocessing intercept ONLY to select which association to
    # inspect (not as an independent verification of absolute clock origin).
    candidates=[]
    for candidate_shift in range(-10,11):
        a0,h0=_shift_local(bv,hp,candidate_shift)
        if len(a0)<15:continue
        slope0,intercept0=np.polyfit(a0,h0,1)
        err_intervals=np.diff(a0)-np.diff(h0)
        interval_rmse=float(np.sqrt(np.mean(err_intervals**2)))
        candidates.append((interval_rmse,abs(intercept0-stored_intercept_s) if np.isfinite(stored_intercept_s) else abs(candidate_shift),candidate_shift))
    if not candidates:return blank
    best_interval=min(r[0] for r in candidates)
    plausible=[row for row in candidates if row[0] <= best_interval+0.002]
    chosen=min(plausible,key=lambda r:(r[1],abs(r[2])))
    shift=int(chosen[2])
    matching_status=('periodic_ambiguity_selected_against_stored_fit' if len(plausible)>1 and np.isfinite(stored_intercept_s)
                     else 'periodic_ambiguity_without_absolute_anchor' if len(plausible)>1 else 'unique_interval_pairing')
    x,y=_shift_local(bv,hp,shift)
    n=min(len(x),len(y));x,y=x[:n],y[:n]
    if n<15 or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):return blank
    test=np.arange(1,n-1,5)
    train=np.ones(n,dtype=bool);train[test]=False
    pred=np.interp(x[test],x[train],y[train])
    error=1000*(pred-y[test])
    if not len(error):return blank
    return dict(status='modern_holdout_bv_harp_only',n=int(len(error)),
                median_ms=float(np.median(error)),p95_abs_ms=float(np.percentile(abs(error),95)),
                max_abs_ms=float(np.max(abs(error))),rmse_ms=float(np.sqrt(np.mean(error*error))),
                first_clock_offset_ms=float(1000*(y[0]-x[0])),
                last_clock_offset_ms=float(1000*(y[-1]-x[-1])),
                clock_offset_change_ms=float(1000*((y[-1]-x[-1])-(y[0]-x[0]))),
                time_min=x[test]/60,residual_ms=error,
                offset_matching_status=matching_status,selected_edge_shift=shift,
                n_plausible_shifts=len(plausible))


def fallback_fit_consistency(bonsai_csv: Path, pd_path: Path, meta: dict, modulo_frames: int = 30) -> dict:
    """Legacy modality: fitted frame-modulo transitions to nearest HARP edge.

    Fit-consistency ONLY (may spuriously pass due to periodic cycle ambiguity).
    No held-out claims; less reliable than BV-encoded photodiode.
    """
    out=dict(status='not_measured', n=0,p95_abs_ms=np.nan,median_ms=np.nan,
             time_min=np.array([]),residual_ms=np.array([]))
    phase=number(meta.get('phase'))
    a=number(meta.get('slope')); b=number(meta.get('intercept'))
    if not all(np.isfinite(z) for z in (phase,a,b)):
        return out
    bv=_load_bv_local(bonsai_csv)
    frames_all=bv.loc[bv['Value'].astype(str).str.lower().eq('frame'),'Timestamp']
    t0=float(frames_all.iloc[0]) if len(frames_all) else float(bv['Timestamp'].min())
    isframe=bv['Value'].astype(str).str.lower().eq('frame')
    fr=bv.loc[isframe,['Timestamp','Frame']].dropna()
    idx=pd.to_numeric(fr['Frame'],errors='coerce').to_numpy()
    tt=pd.to_numeric(fr['Timestamp'],errors='coerce').to_numpy()-t0
    sel=np.isfinite(idx)&np.isfinite(tt)&(idx.astype(int)%modulo_frames==int(phase))
    pred=np.sort(a*tt[sel]+b)
    pd_data=photodiode_edges_and_origin(pd_path,None)
    actual=np.asarray(pd_data['edges_rel'],float)
    if not len(actual) or not len(pred):
        return out
    pos=np.searchsorted(actual,pred)
    lo=np.clip(pos-1,0,len(actual)-1); hi=np.clip(pos,0,len(actual)-1)
    close=np.where(abs(pred-actual[lo])<=abs(pred-actual[hi]),lo,hi)
    residual=1000*(pred-actual[close])
    period=np.median(np.diff(actual)) if len(actual)>2 else np.nan
    valid=(pred>=actual[0])&(pred<=actual[-1])&np.isfinite(residual)
    if np.isfinite(period):valid &= abs(residual)<period*1000*.25
    err=residual[valid]; pred=pred[valid]
    if len(err)==0:return out
    return dict(status='legacy_fit_consistency_not_independent',n=int(len(err)),
                median_ms=float(np.median(err)),p95_abs_ms=float(np.percentile(abs(err),95)),
                time_min=pred/60,residual_ms=err)


def source_voltage_boundaries(h5_path: Path) -> dict:
    """Check source-line boundary convention and literal extracted fluorescence equality."""
    import h5py
    out=dict(source_boundaries_tested=0,source_shared_lines=0,source_identical_samples=0,
             source_duplicated_sample_equivalent_ms=np.nan,source_nominal_sample_rate_hz=np.nan)
    with h5py.File(h5_path,'r') as f:
        if 'traces' not in f:return out
        keys=sorted([s for s in f['traces'] if s.startswith('trial_')],key=lambda s:int(s.split('_')[-1]))
        prev=None
        for k in keys:
            ds=f['traces'][k]
            if ds.ndim!=2 or not ds.shape[1] or not ds.shape[0]:continue
            first=number(ds.attrs.get('trialFirstLineGlobal'));last=number(ds.attrs.get('trialLastLineGlobal'))
            start=np.asarray(ds[:min(2,ds.shape[0]),0]);end=np.asarray(ds[:min(2,ds.shape[0]),-1])
            prefix=str(ds.attrs.get('acquisitionPrefix',''))
            epoch=str(ds.attrs.get('epochIdx',''))
            if prev is not None and prev[3:]==(prefix,epoch):
                out['source_boundaries_tested']+=1
                if np.isfinite(prev[1]) and prev[1]==first:out['source_shared_lines']+=1
                if np.array_equal(prev[2],start,equal_nan=True):out['source_identical_samples']+=1
            prev=(first,last,end,prefix,epoch)
    return out


def derived_timebase_qc(h5_path: Path, chunk_size: int = 250_000) -> dict:
    """Chunked monotonicity and DMD equality check, not physical hardware validation."""
    import h5py
    out=dict(derived_dmd1_n=np.nan,derived_dmd2_n=np.nan,
             derived_timestamp_nonmonotonic=0,derived_dmd_max_abs_diff_us=np.nan,
             derived_sample_rate_hz=np.nan)
    with h5py.File(h5_path,'r') as h:
        for name in ('DMD1','DMD2'):
            if name not in h or 'timebase_sec' not in h[name]:continue
            ds=h[name]['timebase_sec'];n=int(ds.shape[0]); out[f'derived_{name.lower()}_n']=n
            prev=None
            for lo in range(0,n,chunk_size):
                x=np.asarray(ds[lo:lo+chunk_size],dtype=float)
                if len(x)==0:continue
                if prev is not None and (not np.isfinite(x[0]) or x[0]<=prev):out['derived_timestamp_nonmonotonic']+=1
                out['derived_timestamp_nonmonotonic']+=int(np.sum((np.diff(x)<=0)|~np.isfinite(np.diff(x))))
                prev=float(x[-1])
        if 'DMD1' in h and 'DMD2' in h and 'timebase_sec' in h['DMD1'] and 'timebase_sec' in h['DMD2']:
            a=h['DMD1/timebase_sec'];b=h['DMD2/timebase_sec']
            if len(a)==len(b):
                mx=0.
                for lo in range(0,len(a),chunk_size):
                    d=abs(np.asarray(a[lo:lo+chunk_size])-np.asarray(b[lo:lo+chunk_size]))
                    if len(d):mx=max(mx,float(np.max(d)))
                out['derived_dmd_max_abs_diff_us']=1e6*mx
    return out


def giant_frame_index_inventory(session_dir: Path) -> dict:
    """Presence and gap/duplicate counts for authentic GIAnT frame_line_idxs.

    Only counts; does not invent absolute HARP mapping or sample equivalence.
    """
    import h5py
    out=dict(giant_experiment_summary_present=False,giant_frame_line_idx_path='',
             giant_frame_line_count=0,giant_nonincreasing_frame_line_steps=0,
             giant_frame_line_gap_p95=np.nan, giant_frame_line_error='')
    matches=list(session_dir.rglob('*experiment_summary*.h5'))+list(session_dir.rglob('*ExperimentSummary*.h5'))+list(session_dir.rglob('*experimentSummary*.h5'))
    if not matches:return out
    path=max(matches,key=lambda p:p.stat().st_mtime)
    out['giant_experiment_summary_present']=True
    try:
        with h5py.File(path,'r') as h:
            for key in ('Path1/frame_info/frame_line_idxs','Path2/frame_info/frame_line_idxs'):
                if key not in h:continue
                ds=h[key];out['giant_frame_line_idx_path']=f'{path.name}:/{key}'
                previous=None;steps=[]
                axis=int(np.argmax(ds.shape)) if ds.ndim else 0
                n=int(ds.shape[axis])
                for lo in range(0,n,250000):
                    slices=[slice(None)]*ds.ndim; slices[axis]=slice(lo,lo+250000)
                    a=np.asarray(ds[tuple(slices)],dtype=float).reshape(-1)
                    if previous is not None:a=np.r_[previous,a]
                    dd=np.diff(a);steps.append(dd)
                    if len(a):previous=a[-1]
                ds_steps=np.concatenate(steps) if steps else np.array([])
                out['giant_frame_line_count']=int(ds.size)
                out['giant_nonincreasing_frame_line_steps']=int(np.sum(ds_steps<=0))
                if len(ds_steps):out['giant_frame_line_gap_p95']=float(np.percentile(ds_steps,95))
                break
    except Exception as e:out['giant_frame_line_error']=f'{type(e).__name__}: {e}'
    return out


def audit_session(registry: VIPSessionRegistry, row: pd.Series, *,
                  detail: bool = False, include_h5: bool = False) -> tuple[dict,dict]:
    """Collect summary and optional plotting arrays for ONE session."""
    asset=registry.resolve_assets(row)
    session_dir=Path(asset.session_dir)
    sid=str(asset.session_id)
    m=dict(session_id=sid, subject_id=asset.subject_id, modality=modalities_for(row,asset),
           source_session_dir=str(session_dir),
           alignment_method='',alignment_pathway='unknown',
           bv_harp_intercept_ms=np.nan,bv_harp_slope=np.nan,
           bv_harp_interval_rmse_ms=np.nan,bv_harp_edges_matched=np.nan,
           bv_harp_holdout_p95_ms=np.nan,bv_harp_holdout_n=0,
           bv_harp_holdout_median_ms=np.nan,bv_harp_clock_offset_first_ms=np.nan,
           bv_harp_clock_offset_last_ms=np.nan,bv_harp_clock_offset_change_ms=np.nan,
           bv_harp_offset_pairing_status='not_evaluated',bv_harp_plausible_edge_shifts=np.nan,
           legacy_fit_consistency_p95_ms=np.nan,legacy_fit_consistency_n=0,
           di3_n_edges=0,di3_period_ms=np.nan,di3_n_outlier_intervals=np.nan,
           harp_photodiode_to_di3_n=0,
           harp_photodiode_to_di3_median_phase_ms=np.nan,
           harp_photodiode_to_di3_p95_abs_phase_ms=np.nan,
           harp_photodiode_to_di3_phase_change_ms=np.nan,
           display_rate_est_hz=np.nan,display_frame_threshold_ms=np.nan,
           display_threshold_origin='not_established',
           imaging_to_stimulus_lag_ms=np.nan,
           imaging_to_stimulus_p95_abs_error_ms=np.nan,
           t03_status='NOT_INDEPENDENTLY_VERIFIED',t03_reason='No verified line-index to HARP correspondence',
           analysis_errors='',evidence_sources='')
    details={}
    src=[];errs=[]
    qc=Path(asset.qc_dir)/'behavior'
    align=read_json(qc/'alignment_meta.json');pd_det=read_json(qc/'di3_pulse_train_detection.json')
    if align:src.append(str(qc/'alignment_meta.json'))
    if pd_det:src.append(str(qc/'di3_pulse_train_detection.json'))
    bv_file=Path(asset.bonsai_event_log_csv) if asset.bonsai_event_log_csv else None
    m.update(original_behavior_alignment_method(align,bv_file))
    if m['alignment_method_conflict']:
        errs.append(m['alignment_method_conflict'])
    if m['alignment_method_source'].startswith('bonsai_event_log.csv') and bv_file:
        src.append(f'{bv_file} [original alignment_method column]')
    for col,field,scale in [('bv_harp_intercept_ms','intercept',1000),('bv_harp_slope','slope',1),
                            ('bv_harp_interval_rmse_ms','edge_rmse_intervals',1000),
                            ('bv_harp_edges_matched','matched_edges',1),
                            ('display_rate_est_hz','display_rate_est_hz',1)]:
        v=number(align.get(field));m[col]=v*scale if np.isfinite(v) else np.nan
    rate=m['display_rate_est_hz']
    if np.isfinite(rate) and 30<=rate<=240:
        m['display_frame_threshold_ms']=1000/rate
        m['display_threshold_origin']='estimated_from_harp_pd_edge_period_modulo30_not_direct_display_measurement'
    if pd_det:
        m['di3_n_edges']=pd_det.get('n_rising_edges',0)
        m['di3_period_ms']=1000*number(pd_det.get('median_period_s'))
    hp_file=Path(asset.harp_df_csv) if asset.harp_df_csv else None
    pd_file=Path(asset.photodiode_pkl) if asset.photodiode_pkl else None
    # For speed, scan raw HARP only in detail mode; single-pass run optional full.
    pulses=np.array([])
    if hp_file and hp_file.is_file() and detail:
        try:
            pulses=rising_edges_from_harp_csv(hp_file)
            m.update(di3_clock_diagnostics(pulses));src.append(str(hp_file))
            details['di3_rising_edges']=pulses
        except Exception as e:errs.append(f'DI3: {type(e).__name__}: {e}')
    if bv_file and pd_file and pd_file.is_file() and detail:
        try:
            if m['alignment_pathway']=='encoded_photodiode':
                b=modern_holdout(bv_file,pd_file,stored_intercept_s=number(align.get('intercept')))
                m.update(bv_harp_holdout_p95_ms=b['p95_abs_ms'],
                         bv_harp_holdout_n=b['n'],bv_harp_holdout_median_ms=b['median_ms'],
                         bv_harp_clock_offset_first_ms=b['first_clock_offset_ms'],
                         bv_harp_clock_offset_last_ms=b['last_clock_offset_ms'],
                         bv_harp_clock_offset_change_ms=b['clock_offset_change_ms'],
                         bv_harp_offset_pairing_status=b['offset_matching_status'],
                         bv_harp_plausible_edge_shifts=b['n_plausible_shifts'])
                details['bv_harp']=b
            elif m['alignment_pathway']=='legacy_frame_modulo':
                b=fallback_fit_consistency(bv_file,pd_file,align)
                m.update(legacy_fit_consistency_p95_ms=b['p95_abs_ms'],legacy_fit_consistency_n=b['n'])
                details['bv_harp_legacy']=b
            src.extend([str(bv_file),str(pd_file)])
        except Exception as e:errs.append(f'BV-HARP: {type(e).__name__}: {e}')
    if pd_file and pd_file.is_file() and pulses.size>=3 and detail:
        try:
            hd=photodiode_edges_and_origin(pd_file,hp_file)
            # `pulses` and PD index must share absolute time coordinates; check
            # physical index origin proximity before computing lag.
            origin_gap_s=abs(float(pulses[0])-hd['harp_pd_start_raw'])
            m['harp_stream_start_difference_ms']=1000*origin_gap_s
            m['harp_stream_origin_plausible']=bool(origin_gap_s<120)
            m['harp_pd_di3_phase_status']='conditional_shared_HARP_timestamp_origin' if origin_gap_s<120 else 'cannot_establish_shared_HARP_origin'
            if origin_gap_s<120 and hd['rises_absolute'].size:
                phase=nearest_pulse_phase(hd['rises_absolute'],pulses)
                m.update(harp_photodiode_to_di3_n=phase['n'],
                         harp_photodiode_to_di3_median_phase_ms=phase['median_ms'],
                         harp_photodiode_to_di3_p95_abs_phase_ms=phase['p95_abs_ms'],
                         harp_photodiode_to_di3_phase_change_ms=phase['phase_change_ms'])
                details['harp_pd_di3']=phase
                details['harp_pd_rising_edges_absolute']=hd['rises_absolute']
            else:errs.append('HARP analog and digital origins not validated; omitted event-to-DI3 phase')
        except Exception as e:errs.append(f'HARP PD-DI3: {type(e).__name__}: {e}')
    if include_h5:
        s=asset.get_asset('voltage','trace_h5')
        if s and Path(s).is_file():
            try:m.update(source_voltage_boundaries(Path(s)));src.append(str(s))
            except Exception as e:errs.append(f'Voltage source: {type(e).__name__}: {e}')
        der=Path(asset.derived_dir)/'voltage'
        hits=list(Path(asset.derived_dir).rglob('voltage_session_traces_dff_robust_f0_trial.h5')) if Path(asset.derived_dir).exists() else []
        if hits:
            try:m.update(derived_timebase_qc(max(hits,key=lambda x:x.stat().st_mtime)));src.append(str(hits[0]))
            except Exception as e:errs.append(f'Derived H5: {type(e).__name__}: {e}')
        # Explicitly opt in: this is a potentially expensive recursive search and read.
        if m['modality'] in ('glutamate','calcium','glutamate + calcium','glutamate + voltage'):
            try:m.update(giant_frame_index_inventory(session_dir))
            except Exception as e:errs.append(f'GIAnT index: {type(e).__name__}: {e}')
    if not np.isfinite(m['display_frame_threshold_ms']):
        m['t03_reason']='Display frame period unavailable; source-to-HARP correspondence not independently validated'
    m['analysis_errors']=' | '.join(errs)
    m['evidence_sources']=' | '.join(src)
    return m,details


def apply_independent_validation(summary: pd.DataFrame, validation_path: Optional[Path]) -> pd.DataFrame:
    """Only externally verified PHYSICAL SLAP2->HARP timing can close T0.3.

    Required input CSV columns:
      session_id,imaging_to_stimulus_lag_ms,p95_abs_error_ms,
      method,evidence_path,independent_validation
    The last must be true. Source line↔DI3 matched origin/phase must be measured.
    The one-frame threshold itself must be independently verified: optional
    `verified_frame_period_ms` is preferred over an estimated display rate.
    """
    if validation_path is None or not Path(validation_path).is_file():return summary
    v=pd.read_csv(validation_path)
    req={'session_id','imaging_to_stimulus_lag_ms','p95_abs_error_ms','method','evidence_path','independent_validation'}
    if not req.issubset(v):raise ValueError(f'Validation CSV missing fields: {sorted(req-set(v.columns))}')
    if v['session_id'].duplicated().any():raise ValueError('Duplicate session IDs in verification CSV')
    d=v.set_index('session_id')
    for i,row in summary.iterrows():
        sid=row['session_id']
        if sid not in d.index:continue
        x=d.loc[sid]
        valid=str(x['independent_validation']).strip().lower() in ('true','1','yes')
        lag=number(x['imaging_to_stimulus_lag_ms']);err=number(x['p95_abs_error_ms'])
        evidence=str(x['evidence_path']).strip() if pd.notna(x['evidence_path']) else ''
        method=str(x['method']).strip() if pd.notna(x['method']) else ''
        frame=number(x.get('verified_frame_period_ms'))
        # Restrict to evidence with an actual independent source-clock association
        if not (valid and np.isfinite(lag) and np.isfinite(err) and err>=0 and evidence and method):
            summary.at[i,'t03_status']='NOT_INDEPENDENTLY_VERIFIED'
            summary.at[i,'t03_reason']='External validation row incomplete/unapproved'
            continue
        if not np.isfinite(frame) or frame<=0:
            summary.at[i,'t03_status']='NOT_INDEPENDENTLY_VERIFIED'
            summary.at[i,'t03_reason']='Verified display frame period needed for one-frame acceptance'
            continue
        summary.at[i,'display_frame_threshold_ms']=frame
        summary.at[i,'display_threshold_origin']='independently_verified'
        summary.at[i,'imaging_to_stimulus_lag_ms']=lag
        summary.at[i,'imaging_to_stimulus_p95_abs_error_ms']=err
        summary.at[i,'t03_status']='PASS' if err<frame else 'FAIL'
        summary.at[i,'t03_reason']=f'Independent {method}: p95 |error| = {err:.3g} ms, frame = {frame:.3g} ms. {evidence}'
    return summary


def run_registry_audit(summary_xlsx: Path, *, selection: Optional[list[str]] = None,
                       detail: bool = False, include_h5: bool = False,
                       validation_csv: Optional[Path] = None,
                       output_dir: Optional[Path] = None) -> tuple[pd.DataFrame,dict]:
    reg=VIPSessionRegistry.from_excel(summary_xlsx)
    sessions=reg.sessions()
    if selection:
        missing=set(selection)-set(sessions['session_id'].astype(str))
        if missing:raise KeyError(f'Session IDs absent from registry: {sorted(missing)}')
        sessions=sessions[sessions['session_id'].isin(selection)]
    records=[];details={}
    if output_dir:Path(output_dir).mkdir(parents=True,exist_ok=True)
    for idx,(_,r) in enumerate(sessions.iterrows(),1):
        sid=str(r['session_id']);print(f'[{idx}/{len(sessions)}] {sid}',flush=True)
        try:
            record,detail_data=audit_session(reg,r,detail=detail,include_h5=include_h5)
        except Exception as e:
            record=dict(session_id=sid,subject_id=r.get('subject_id'),modality='unknown',
                        alignment_pathway='error',t03_status='NOT_INDEPENDENTLY_VERIFIED',
                        t03_reason='Registry or session read error',
                        analysis_errors=f'{type(e).__name__}: {e}')
            detail_data={}
        records.append(record);details[sid]=detail_data
        if output_dir:pd.DataFrame(records).to_csv(Path(output_dir)/'t03_sessions_partial.csv',index=False)
    table=apply_independent_validation(pd.DataFrame(records),validation_csv)
    if output_dir:table.to_csv(Path(output_dir)/'t03_session_results.csv',index=False)
    return table,details
