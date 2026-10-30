"""T0.3 per-event synchronization diagnostics; read-only, modality-independent.

This is intentionally NOT an independent SLAP2 source-sample clock mapper.
The links computed here are corrected BonVision event -> measured HARP analog
photodiode and HARP analog photodiode -> HARP digital DI3 cycle edges.
For modern recordings, corrected BonVision timestamps already used photodiode
anchors; in-sample event differences are therefore fit consistency, not an
independent clock-validation result. Legacy modulo phase may be reconstructed
from the corrected log; such a phase is likewise diagnostic only.

No physiological trace is loaded or filtered. HARP photodiode edges are read
with the detector from the existing vip-slap2-analysis synchronization module.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from vip_slap2_analysis.metadata.t03_session_sync import (
    _load_bv_local, _load_pd_local, _pd_edges_local, nearest_pulse_phase,
    number, read_json, original_behavior_alignment_method,
)


def _blank(status='not_measured'):
    return dict(event_comparison_status=status, event_match_n=0, event_candidate_n=0,
                event_coverage=np.nan, event_median_ms=np.nan,
                event_p95_abs_ms=np.nan, event_rmse_ms=np.nan,
                event_p99_abs_ms=np.nan, event_first_vs_last_median_ms=np.nan,
                event_fraction_within_frame=np.nan, event_phase_source='',
                event_phase_ambiguity_ms=np.nan,
                harp_pd_sample_dt_median_ms=np.nan,
                harp_pd_sample_dt_p95_ms=np.nan,
                harp_pd_edges_rise_n=0, harp_pd_edges_both_n=0,
                event_bv_time_s=np.array([],float),
                event_harp_time_s=np.array([],float),
                event_residual_ms=np.array([],float),
                event_paired_bv_index=np.array([],int),
                event_nearest_di3_phase_ms=np.array([],float),
                event_nearest_di3_time_s=np.array([],float),
                event_bv_to_di3_phase_ms=np.array([],float),
                event_bv_to_di3_p95_abs_phase_ms=np.nan,
                event_pd_to_di3_p95_abs_phase_ms=np.nan,
                event_same_nearest_di3_cycle_fraction=np.nan,
                event_bv_di3_clock_status='not_available')


def _match_unique(predicted, observed, max_abs_seconds):
    """Monotonic nearest-neighbor one-to-one event association in the SAME clock.

    Rejects ambiguous associations if adjacent targets are closer than twice
    the threshold, or if the same target is assigned to two BonVision events.
    A match is *not* a proof that the absolute epoch/origin is correct.
    """
    a=np.asarray(predicted,float)
    b=np.asarray(observed,float)
    if len(a)==0 or len(b)==0:
        return np.array([],int),np.array([],int)
    pos=np.searchsorted(b,a)
    lo=np.clip(pos-1,0,len(b)-1);hi=np.clip(pos,0,len(b)-1)
    tgt=np.where(abs(a-b[lo])<=abs(a-b[hi]),lo,hi)
    dist=abs(a-b[tgt]);good=np.isfinite(a)&(dist<=max_abs_seconds)
    ai=np.flatnonzero(good);bi=tgt[good]
    if not len(ai):return ai,bi
    # If multiple predicted transitions map to same observed edge, exclude ALL
    # ambiguous matches rather than retaining a seemingly precise winner.
    _,inv,counts=np.unique(bi,return_inverse=True,return_counts=True)
    keep=counts[inv]==1
    return ai[keep],bi[keep]


def _frame_dataframe(log):
    b=_load_bv_local(log)
    # Some original event logs contain the encoded Photodiode-* Value rows
    # but not the derived photodiode_state column. Reconstruct only that
    # categorical state (never redo alignment) before selecting Frame rows.
    if 'photodiode_state' not in b or pd.to_numeric(b['photodiode_state'], errors='coerce').notna().sum()==0:
        values=b['Value'].astype(str)
        pd_states=np.full(len(b),np.nan,float)
        pd_states[values.str.fullmatch(r'Photodiode-1',case=False).to_numpy()]=1.0
        pd_states[values.str.fullmatch(r'Photodiode-0',case=False).to_numpy()]=0.0
        b['photodiode_state']=pd.Series(pd_states,index=b.index).ffill().to_numpy()
    b=b[b['Value'].astype(str).str.lower().eq('frame')].copy()
    if 'corrected_timestamps' not in b or 'Frame' not in b:
        raise ValueError('Corrected Frame rows (corrected_timestamps and Frame) are required')
    b['corrected_timestamps']=pd.to_numeric(b['corrected_timestamps'],errors='coerce')
    b['Frame']=pd.to_numeric(b['Frame'],errors='coerce')
    return b.dropna(subset=['corrected_timestamps','Frame'])


def _encoded_candidates(fr):
    if 'photodiode_state' not in fr:
        return []
    states=pd.to_numeric(fr['photodiode_state'],errors='coerce').ffill().fillna(0).to_numpy(float)
    ix=np.flatnonzero(np.diff(states,prepend=states[0])>0.5)
    t=fr['corrected_timestamps'].to_numpy(float)
    ix=ix[ix>0]
    return [(label,x,ix) for label,x in [
        ('previous_frame',t[ix-1]),
        ('midpoint_frames',.5*(t[ix-1]+t[ix])),
        ('current_frame',t[ix])]]


def _fallback_candidates(fr, phase, modulo_frames=30):
    idx=fr['Frame'].to_numpy(float)
    t=fr['corrected_timestamps'].to_numpy(float)
    if np.isfinite(phase):
        phases=[int(phase)%modulo_frames];method='saved_phase'
    else:
        phases=list(range(modulo_frames));method='reconstructed_phase'
    return [(f'{method}_{ph}',t[(idx.astype(int)%modulo_frames)==ph],
             np.flatnonzero((idx.astype(int)%modulo_frames)==ph)) for ph in phases]


def _event_series_summary(pred, observed, ai, bi, frame_ms):
    if not len(ai):return dict(n=0,score=np.inf,coverage=0,p95=np.nan)
    error_ms=1000*(pred[ai]-observed[bi])
    p95=float(np.percentile(abs(error_ms),95))
    # Require coverage to avoid selecting the one candidate that matches
    # only a tiny number of periodic stimulus events.
    coverage=len(ai)/max(len(pred),1)
    score=p95+1000*max(0,.75-coverage)
    return dict(n=len(ai),score=score,coverage=coverage,p95=p95,
                ai=ai,bi=bi,error_ms=error_ms)


def event_pd_diagnostics(bonsai_csv: Path, pd_pkl: Path, method_pathway: str,
                         *, frame_period_ms: float = float('nan'),
                         saved_phase: float = float('nan'),
                         di3_edges_abs=None, shared_origin_verified: bool = False,
                         max_match_ms: float = 80.0) -> dict:
    """Compare corrected BonVision event predictions with measured HARP PD edges.

    Event matching remains conditional on the upstream absolute origin and
    period aliasing. ``shared_origin_verified`` must not be inferred just from
    a small gap between start times of two streams.
    """
    out=_blank()
    pd_frame=_load_pd_local(Path(pd_pkl))
    raw=np.asarray(pd_frame.index,float)
    if len(raw)<3 or np.any(np.diff(raw)<=0):
        return _blank('bad_harp_pd_time_index')
    both_rel,rises_rel=_pd_edges_local(pd_frame)
    deltas=1000*np.diff(raw)
    out.update(harp_pd_sample_dt_median_ms=float(np.median(deltas)),
               harp_pd_sample_dt_p95_ms=float(np.percentile(deltas,95)),
               harp_pd_edges_rise_n=len(rises_rel),harp_pd_edges_both_n=len(both_rel))
    fr=_frame_dataframe(Path(bonsai_csv))
    if method_pathway=='encoded_photodiode':
        candidates=_encoded_candidates(fr)
        observed=rises_rel
        outcome='modern_corrected_event_vs_harp_pd_in_sample'
    elif method_pathway=='legacy_frame_modulo':
        candidates=_fallback_candidates(fr,number(saved_phase))
        observed=both_rel
        outcome='legacy_corrected_event_vs_harp_pd_fit_consistency'
    else:
        out['event_comparison_status']='unknown_alignment_method'
        return out
    if len(observed)<3 or not candidates:
        out['event_comparison_status']='no_usable_event_transitions'
        return out
    # Limit matching to a fraction of the analog photodiode transition interval;
    # this is not equivalent to verifying the correct whole-period cycle match.
    delta=np.diff(observed)
    delta=delta[(delta>0)&(delta<2)]
    tol_s=min(max_match_ms/1000,.25*np.median(delta)) if len(delta) else max_match_ms/1000
    if not np.isfinite(tol_s) or tol_s<=0:return _blank('bad_pd_edge_intervals')
    fits=[]
    for label,pred,idx in candidates:
        ai,bi=_match_unique(pred,observed,tol_s)
        metrics=_event_series_summary(pred,observed,ai,bi,frame_period_ms)
        if metrics['n']>=3:
            fits.append((metrics['score'],label,np.asarray(pred,float),np.asarray(idx,int),metrics))
    if not fits:
        out['event_comparison_status']='no_monotonic_matches'
        return out
    fits.sort(key=lambda x:x[0]);best=fits[0]
    _,label,pred,indices,m=best
    ai,bi=m['ai'],m['bi'];res=m['error_ms']
    phase_ambiguity=float(fits[1][0]-fits[0][0]) if len(fits)>1 else np.nan
    t=pred[ai]
    nquart=max(1,len(res)//4)
    out.update(event_comparison_status=outcome, event_match_n=int(len(res)),
               event_candidate_n=int(len(pred)),event_coverage=float(m['coverage']),
               event_median_ms=float(np.median(res)),
               event_p95_abs_ms=float(np.percentile(abs(res),95)),
               event_p99_abs_ms=float(np.percentile(abs(res),99)),
               event_rmse_ms=float(np.sqrt(np.mean(res**2))),
               event_first_vs_last_median_ms=float(np.median(res[-nquart:])-np.median(res[:nquart])),
               event_fraction_within_frame=float(np.mean(abs(res)<frame_period_ms)) if np.isfinite(frame_period_ms) and frame_period_ms>0 else np.nan,
               event_phase_source=label,event_phase_ambiguity_ms=phase_ambiguity,
               event_bv_time_s=t,event_harp_time_s=observed[bi],
               event_residual_ms=res,event_paired_bv_index=indices[ai])
    if m['coverage']<.75:
        out['event_comparison_status']='low_coverage_'+outcome
    # Native HARP analog and digital timestamps must share the same clock
    # origin. With only the repository's start-time heuristic we report
    # CONDITIONAL, not independently verified, hardware phase.
    if di3_edges_abs is not None and shared_origin_verified:
        pulses=np.asarray(di3_edges_abs,float)
        if len(pulses)>=3:
            med_diffs=np.diff(pulses)
            med_diffs=med_diffs[(med_diffs>0)&(med_diffs<.5)]
            if len(med_diffs):
                med=float(np.median(med_diffs))
                def signed_phases_and_index(times_abs):
                    pos=np.searchsorted(pulses,times_abs)
                    low=np.clip(pos-1,0,len(pulses)-1)
                    high=np.clip(pos,0,len(pulses)-1)
                    chosen=np.where(abs(times_abs-pulses[low])<=abs(times_abs-pulses[high]),low,high)
                    dt=times_abs-pulses[chosen]
                    good=((times_abs>=pulses[0])&(times_abs<=pulses[-1])&
                          (abs(dt)<=.75*med))
                    dt_ms=np.where(good,dt*1000,np.nan)
                    chosen=np.where(good,chosen,-1)
                    return dt_ms,chosen
                t_bv_abs=t+raw[0]
                t_pd_abs=observed[bi]+raw[0]
                bv_ph,bv_idx=signed_phases_and_index(t_bv_abs)
                pd_ph,pd_idx=signed_phases_and_index(t_pd_abs)
                valid=np.isfinite(bv_ph)&np.isfinite(pd_ph)
                same=valid&(bv_idx==pd_idx)
                out.update(event_nearest_di3_time_s=t_pd_abs[valid],
                           event_nearest_di3_phase_ms=pd_ph[valid],
                           event_bv_to_di3_phase_ms=bv_ph[valid],
                           event_bv_to_di3_p95_abs_phase_ms=float(np.percentile(abs(bv_ph[valid]),95)) if np.any(valid) else np.nan,
                           event_pd_to_di3_p95_abs_phase_ms=float(np.percentile(abs(pd_ph[valid]),95)) if np.any(valid) else np.nan,
                           event_same_nearest_di3_cycle_fraction=float(np.mean(same[valid])) if np.any(valid) else np.nan,
                           event_bv_di3_clock_status='conditional_shared_HARP_index_origin')
    return out


def extend_dataset_audit(df: pd.DataFrame, event_detail: dict, summary_xlsx: Path,
                         *, selected_sessions=None, frame_period_ms=1000/60,
                         max_match_ms=80.0, output_dir=None):
    """Registry-based extension: per-session event residuals, not fit reruns.

    Pass ``frame_period_ms`` as a *reference line* for figures; by default
    16.667 ms represents an assumed 60 Hz monitor, not a verified session rate.
    The existing independent validation statuses are NOT promoted from this.
    """
    from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
    reg=VIPSessionRegistry.from_excel(Path(summary_xlsx))
    records=[]; extended={}
    for _,row in df.iterrows():
        sid=str(row['session_id']); asset=reg.resolve_assets(sid)
        out=_blank('missing_assets')
        log=Path(asset.bonsai_event_log_csv) if asset.bonsai_event_log_csv else None
        pd_path=Path(asset.photodiode_pkl) if asset.photodiode_pkl else None
        estimated_frame=number(row.get('display_frame_threshold_ms'))
        session_frame=(estimated_frame if np.isfinite(estimated_frame) and estimated_frame>0 else frame_period_ms)
        if log is not None and pd_path is not None and log.is_file() and pd_path.is_file():
            qc=Path(asset.qc_dir)/'behavior'; metadata=read_json(qc/'alignment_meta.json')
            # Recover original phase only if it is present; do not re-fit the
            # behavior pipeline or overwrite existing corrected timestamps.
            d=event_detail.get(sid,{})
            origin_ok=(str(row.get('harp_pd_di3_phase_status','')).startswith('conditional_shared_HARP')
                       or row.get('harp_pd_di3_phase_status')=='independently_verified_shared_origin')
            pulses=d.get('di3_rising_edges') if origin_ok else None
            try:
                out=event_pd_diagnostics(log,pd_path,row.get('alignment_pathway','unknown'),
                                         frame_period_ms=session_frame,
                                         saved_phase=number(metadata.get('phase')),
                                         di3_edges_abs=pulses,shared_origin_verified=origin_ok,
                                         max_match_ms=max_match_ms)
            except Exception as exc:
                out=_blank('error_'+type(exc).__name__)
                out['event_diagnostic_error']=str(exc)
        detail_fields={k:v for k,v in out.items() if isinstance(v,np.ndarray)}
        summary_fields={k:v for k,v in out.items() if not isinstance(v,np.ndarray)}
        summary_fields['session_id']=sid
        summary_fields['event_frame_reference_ms']=session_frame
        summary_fields['event_frame_reference_type']=(
            'estimated_from_HARP_PD_period_modulo30' if np.isfinite(estimated_frame) and estimated_frame>0
            else 'assumed_display_reference_not_verified')
        records.append(summary_fields)
        extended[sid]=detail_fields
        if output_dir is not None and len(records)%20==0:
            pd.DataFrame(records).to_csv(Path(output_dir)/'t03_event_metrics_partial.csv',index=False)
    diag=pd.DataFrame.from_records(records)
    table=df.merge(diag,on='session_id',how='left',validate='one_to_one')
    table['behavior_event_gate']='NOT_MEASURED'
    measured=(pd.to_numeric(table.get('event_p95_abs_ms'),errors='coerce').notna() &
              table['event_comparison_status'].astype(str).str.startswith(('modern_','legacy_')) &
              (pd.to_numeric(table['event_coverage'],errors='coerce')>=.75))
    table.loc[measured,'behavior_event_gate']=np.where(
        (table.loc[measured,'event_p95_abs_ms'].astype(float)<table.loc[measured,'event_frame_reference_ms'].astype(float)),
        'SUBFRAME_FIT_CONSISTENCY','ABOVE_FRAME_FIT_CONSISTENCY')
    # Important: behavior_event_gate is never the actual T0.3 source-to-HARP PASS.
    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True,parents=True)
        table.to_csv(Path(output_dir)/'t03_session_results_extended.csv',index=False)
    return table,extended
