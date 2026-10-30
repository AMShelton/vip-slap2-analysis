"""Read-only, registry-driven export of SLAP2/HARP synchronization evidence.

From repo root (after `pip install -e .`):
    python -m vip_slap2_analysis.metadata.export_sync_audit_bundle \
        --summary-xlsx "\\\\allen\\aind\\scratch\\ophys\\Andrew\\VIP_synaptic_dynamics\\VIP_SD_summary.xlsx" \
        --session-id 863774_2026-08-04_08-12-47 \
        --output sync_audit_863774.zip

No source .dat, full H5, raw MAT, or HARP dataframe is copied. Derived
voltage timestamps are scanned in chunks; source-extraction coordinates
are retained separately. No traces are filtered, smoothed, interpolated,
or resampled. Compatible with Python 3.9+.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

from vip_slap2_analysis.io.session_registry import VIPSessionRegistry


BEHAVIOR_QC_FILES = (
    'imaging_epochs.csv', 'imaging_epochs_summary.json',
    'di3_pulse_train_detection.json', 'di3_pulse_train_epoch_qc.csv',
    'di3_pulse_train_gaps.csv', 'alignment_meta.json',
)
VOLTAGE_QC_FILES = ('voltage_qc_metadata.json',)
QC_FILE_MAX_BYTES = 5 * 1024 * 1024


def json_safe(v: Any, depth: int = 0) -> Any:
    """JSON-convert small metadata without expanding large arrays / references."""
    if depth > 8:
        return '<nested metadata omitted>'
    if isinstance(v, np.generic):
        return json_safe(v.item(), depth + 1)
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        return bytes(v).decode('utf-8', errors='replace').rstrip('\x00')[:2000]
    if isinstance(v, Path):
        return v.name
    if isinstance(v, np.ndarray):
        if v.size > 10000:
            return {'shape': list(v.shape), 'dtype': str(v.dtype), 'not_exported': 'large metadata array'}
        return json_safe(v.tolist(), depth + 1)
    if isinstance(v, dict):
        return {str(k): json_safe(val, depth + 1) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        if len(v) > 10000:
            return {'length': len(v), 'not_exported': 'large metadata list'}
        return [json_safe(x, depth + 1) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v[:100000] if isinstance(v, str) else v
    return str(v)[:250]


def zip_json(z: zipfile.ZipFile, name: str, value: Any) -> None:
    z.writestr(name, json.dumps(json_safe(value), indent=2, allow_nan=False))


def zip_npy(z: zipfile.ZipFile, name: str, x: np.ndarray) -> None:
    buff = io.BytesIO()
    np.save(buff, np.asarray(x), allow_pickle=False)
    z.writestr(name, buff.getvalue())


def file_record(p: Optional[Path], base: Path) -> Dict[str, Any]:
    if p is None:
        return {'exists': False, 'path': None}
    p = Path(p)
    try:
        rel = p.relative_to(base).as_posix()
    except ValueError:
        rel = p.name
    return {'exists': p.is_file(), 'path': rel, 'bytes': p.stat().st_size if p.is_file() else None}


def attr_record(node: Any) -> Dict[str, Any]:
    return {str(k): json_safe(v) for k, v in node.attrs.items()}


def write_qc_files(z: zipfile.ZipFile, asset: Any, status: Dict[str, Any]) -> None:
    for sub, names in [('behavior', BEHAVIOR_QC_FILES), ('voltage', VOLTAGE_QC_FILES)]:
        qdir = asset.qc_dir / sub
        for name in names:
            p = qdir / name
            rel = 'qc/' + sub + '/' + name
            if not p.is_file():
                status[rel] = 'missing'
            elif p.stat().st_size > QC_FILE_MAX_BYTES:
                status[rel] = 'skipped: exceeds 5 MiB'
            else:
                z.write(p, rel)
                status[rel] = 'copied'


def harp_edges_csv(path: Path, z: zipfile.ZipFile) -> Dict[str, Any]:
    """Extract every DI3 rising edge across CSV chunk boundaries; never copy full CSV."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(['time_sec'])
    previous = False
    nrows, nedge, invalid = 0, 0, 0
    first, last = None, None
    # Use text streaming to bound RAM independently of dataframe size.
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'DI3', 'time'}.issubset(reader.fieldnames or []):
            raise ValueError('HARP_df.csv requires columns: DI3, time')
        for row in reader:
            nrows += 1
            try:
                x = str(row['DI3']).strip().lower()
                state = x in ('true', 't') or (float(x) != 0.0 if x not in ('true', 'false', 't', 'f', '') else False)
                t = float(row['time'])
                if not math.isfinite(t):
                    invalid += 1
                    previous = state
                    continue
                if state and not previous:
                    w.writerow([format(t, '.17g')])
                    nedge += 1
                    if first is None:
                        first = t
                    last = t
                previous = state
            except (ValueError, TypeError):
                invalid += 1
    z.writestr('harp/di3_rising_edges.csv', out.getvalue())
    return {'source': 'registry.harp_df_csv', 'rows': nrows, 'rising_edges': nedge,
            'invalid_rows': invalid, 'first_edge_sec': first, 'last_edge_sec': last,
            'full_edge_train_exported': True}


def select_signal_ranges(n: int, width: int) -> Dict[str, tuple]:
    w = min(n, width)
    if not w:
        return {}
    return {'start': (0, w), 'middle': (max(0, (n-w)//2), max(0, (n-w)//2)+w), 'end': (n-w, n)}


def source_h5(z: zipfile.ZipFile, path: Path, *, roi_limit: int = 2,
              boundary_samples: int = 32, trace_window_samples: int = 2048) -> Dict[str, Any]:
    """Export exact source trial attributes and raw boundary samples without constructing timestamps."""
    info: Dict[str, Any] = {'file': path.name, 'authority': 'source_extraction',
                            'timestamps_explicit': False, 'datasets': [], 'windows': [], 'errors': []}
    with h5py.File(path, 'r') as h:
        info['file_attrs'] = attr_record(h)
        if 'traces' not in h:
            info['errors'].append('No /traces group')
            return info
        trace_group = h['traces']
        trial_paths = sorted('traces/' + k for k in trace_group if re.fullmatch(r'trial_\d+', k))
        extra_paths = []
        if 'continuous' in trace_group:
            extra_paths.extend('traces/continuous/' + k for k in trace_group['continuous'].keys())
        if 'epochs' in trace_group:
            for epoch in sorted(trace_group['epochs']):
                extra_paths.extend('traces/epochs/' + epoch + '/' + k for k in trace_group['epochs'][epoch].keys())
        for ds_path in trial_paths + extra_paths:
            ds = h[ds_path]
            if not isinstance(ds, h5py.Dataset):
                continue
            info['datasets'].append({'path': ds_path, 'shape': list(ds.shape), 'dtype': str(ds.dtype),
                                     'attrs': attr_record(ds)})
            if ds.ndim != 2 or ds.dtype.kind not in 'fiu':
                continue
            # Split-H5 voltage extractor stores (global_roi, samples).
            n_roi, n_time = ds.shape
            n_roi = min(roi_limit, n_roi)
            if n_roi == 0 or n_time == 0:
                continue
            is_trial = ds_path in trial_paths
            ranges = select_signal_ranges(n_time, boundary_samples if is_trial else trace_window_samples)
            for location, (start, stop) in ranges.items():
                arr = ds[:n_roi, start:stop]  # literal, unfiltered source samples
                key = f'source/{ds_path.replace("/", "_")}_{location}.npy'
                zip_npy(z, key, arr)
                info['windows'].append({'dataset': ds_path, 'n_rois': int(n_roi),
                                        'sample_start': int(start), 'sample_stop_exclusive': int(stop),
                                        'archive_path': key, 'orientation': 'roi_by_sample'})
    return info


def source_summary_metadata(asset: Any, z: zipfile.ZipFile, status: Dict[str, Any]) -> None:
    mat = asset.get_asset('voltage', 'summary_mat')
    h5 = asset.get_asset('voltage', 'trace_h5')
    if mat is not None and mat.is_file() and h5 is not None and h5.is_file():
        try:
            from vip_slap2_analysis.voltage.summary import VoltageSummary
            with VoltageSummary(mat, trace_path=h5) as vs:
                data = {
                    'authority': 'source_extraction', 'format': vs.layout,
                    'n_trials': int(vs.n_trials), 'n_dmds': int(vs.n_dmds),
                    'n_rois_per_dmd': vs.n_rois,
                    'analysis_rate_hz': vs.analyze_hz(),
                    'source_trial_epoch': vs.trial_epoch,
                    'summary_metadata': vs.metadata,
                    'dmds': {},
                }
                for d in range(1, vs.n_dmds+1):
                    data['dmds'][f'DMD{d}'] = {
                        'dmd_metadata': vs.get_dmd_metadata(d),
                        'source_epoch_metadata': vs.get_dmd_epoch_metadata(d),
                        'source_epoch_durations_sec': vs.get_dmd_epoch_durations_sec(d),
                        'trial_line_ranges': vs.get_trial_line_ranges(d),
                        'trial_timing_table': vs.get_trial_timing_table(d),
                    }
                zip_json(z, 'source/voltage_summary_metadata.json', data)
                # Existing hardware-cadence QC works on metadata; errors should not block export.
                try:
                    from vip_slap2_analysis.common.clock_qc import compare_slap2_harp_clock
                    zip_json(z, 'qc/existing_clock_comparison.json', compare_slap2_harp_clock(
                        vs, behavior_qc_dir=asset.qc_dir / 'behavior', harp_df_csv=asset.harp_df_csv))
                except Exception as exc:
                    status['qc/existing_clock_comparison.json'] = f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            status['source/voltage_summary_metadata.json'] = f'{type(exc).__name__}: {exc}'
    else:
        status['source/voltage_summary_metadata.json'] = 'missing source voltage summary or H5'


def maybe_glu_summary(asset: Any, z: zipfile.ZipFile, status: Dict[str, Any]) -> None:
    """Optional analogous source *acquisition* timing metadata for dual-color sessions."""
    mat = asset.summary_mat
    if mat is None or not mat.is_file():
        status['source/glutamate_calcium_acquisition_metadata.json'] = 'SummaryLoCo missing'
        return
    try:
        from vip_slap2_analysis.glutamate.summary import GlutamateSummary
        with __import__('contextlib').closing(GlutamateSummary(mat)) as gs:
            data = {'authority': 'source_extraction', 'file': mat.name, 'n_dmds': gs.n_dmds,
                    'dmds': {}}
            for d in range(1, gs.n_dmds + 1):
                data['dmds'][f'DMD{d}'] = {
                    'dmd_metadata': gs.get_dmd_metadata(d),
                    'source_epoch_metadata': gs.get_dmd_epoch_metadata(d),
                    'source_epoch_durations_sec': gs.get_dmd_epoch_durations_sec(d),
                }
            zip_json(z, 'source/glutamate_calcium_acquisition_metadata.json', data)
    except Exception as exc:
        status['source/glutamate_calcium_acquisition_metadata.json'] = f'{type(exc).__name__}: {exc}'


def scan_timebase(t: h5py.Dataset, epoch: Optional[h5py.Dataset], *, chunk: int,
                  expected_dt: Optional[float]) -> Dict[str, Any]:
    """Exact streaming continuity check of EVERY retained timestamp and epoch label."""
    n = int(t.shape[0]); first, last = None, None
    previous_t = None; previous_epoch = None
    n_nonfinite = n_nonpositive = n_unexpected = n_epoch_changes = n_backwards_across_epochs = 0
    max_dt = -math.inf; min_dt = math.inf; max_abs_error = 0.
    bad_examples = []
    per_epoch = {}
    for lo in range(0, n, chunk):
        hi = min(n, lo+chunk)
        ts = np.asarray(t[lo:hi], dtype=np.float64)
        labs = np.asarray(epoch[lo:hi], dtype=np.int64) if epoch is not None else np.ones(ts.size, np.int64)
        if ts.size == 0:
            continue
        if first is None:
            first = float(ts[0])
        last = float(ts[-1])
        prev_ts = np.r_[previous_t if previous_t is not None else np.nan, ts[:-1]]
        prev_labs = np.r_[previous_epoch if previous_epoch is not None else labs[0], labs[:-1]]
        dt = ts-prev_ts
        same = labs == prev_labs
        positive = np.isfinite(ts) & np.isfinite(prev_ts) & np.isfinite(dt)
        check = same & positive
        if previous_t is None:
            check[0] = False
        invalid = ~np.isfinite(ts)
        n_nonfinite += int(invalid.sum())
        n_nonpositive += int((check & (dt <= 0)).sum())
        positive_dt = dt[check & (dt > 0)]
        if positive_dt.size:
            max_dt = max(max_dt, float(np.max(positive_dt)))
            min_dt = min(min_dt, float(np.min(positive_dt)))
            if expected_dt is not None:
                err = np.abs(positive_dt - expected_dt)
                max_abs_error = max(max_abs_error, float(err.max()))
        if expected_dt is not None:
            anomalous = check & (np.abs(dt - expected_dt) > max(1e-8, 1e-3 * expected_dt))
            n_unexpected += int(anomalous.sum())
        else:
            anomalous = check & (dt <= 0)
        changes = (labs != prev_labs)
        if previous_t is None:
            changes[0] = False
        n_epoch_changes += int(changes.sum())
        n_backwards_across_epochs += int((changes & positive & (dt <= 0)).sum())
        for idx in np.flatnonzero(anomalous | (check & (dt <= 0)) | ~np.isfinite(ts))[:max(0, 25 - len(bad_examples))]:
            bad_examples.append({'index': int(lo + idx), 'timestamp_sec': float(ts[idx]),
                                 'epoch': int(labs[idx]), 'delta_sec': json_safe(float(dt[idx]))})
        for label in np.unique(labs):
            idxs = np.flatnonzero(labs == label)
            entry = per_epoch.setdefault(str(int(label)), {'n_samples': 0, 'first_sec': None, 'last_sec': None})
            entry['n_samples'] += int(idxs.size)
            if entry['first_sec'] is None:
                entry['first_sec'] = float(ts[idxs[0]])
            entry['last_sec'] = float(ts[idxs[-1]])
        previous_t, previous_epoch = float(ts[-1]), int(labs[-1])
    return {'n_samples_scanned': n, 'first_sec': first, 'last_sec': last,
            'expected_dt_sec': expected_dt, 'n_nonfinite': n_nonfinite,
            'n_nonpositive_steps_within_epoch': n_nonpositive,
            'n_uneven_steps_within_epoch': n_unexpected if expected_dt is not None else None,
            'n_epoch_transitions': n_epoch_changes,
            'n_nonpositive_steps_across_epoch_transitions': n_backwards_across_epochs,
            'minimum_positive_dt_sec': min_dt if math.isfinite(min_dt) else None,
            'maximum_positive_dt_sec': max_dt if math.isfinite(max_dt) else None,
            'max_abs_error_vs_nominal_dt_sec': max_abs_error if expected_dt else None,
            'per_epoch': per_epoch, 'first_25_anomalies': bad_examples}


def sample_times(z: zipfile.ZipFile, t: h5py.Dataset, e: Optional[h5py.Dataset],
                 prefix: str, widths: int = 64) -> list:
    ranges = select_signal_ranges(t.shape[0], widths)
    results = []
    for name, (a,b) in ranges.items():
        key = f'derived/{prefix}_time_{name}.npy'
        zip_npy(z, key, t[a:b])
        out = {'position': name, 'start_index': a, 'stop_index_exclusive': b, 'file': key}
        if e is not None:
            ek = f'derived/{prefix}_epoch_{name}.npy'
            zip_npy(z, ek, e[a:b]); out['epoch_file'] = ek
        results.append(out)
    return results


def derived_h5(z: zipfile.ZipFile, path: Path, *, chunk: int, roi_limit: int,
               boundary_samples: int, trace_window_samples: int,
               skip_full_scan: bool = False) -> Dict[str, Any]:
    """Targeted derived metadata, full clock scans, independent DMD comparison."""
    info: Dict[str, Any] = {'file': path.name, 'authority': 'downstream_derived',
                            'datasets': {}, 'dmds': {}, 'dmd_comparison': {}, 'warnings': []}
    with h5py.File(path, 'r') as h:
        info['file_attrs'] = attr_record(h)
        groups = sorted([k for k in h if re.fullmatch(r'DMD\d+', k)], key=lambda k: int(k[3:]))
        dmd_t = {}
        for dmd in groups:
            g = h[dmd]
            record = {'attrs': attr_record(g), 'arrays': {}, 'selected_windows': [], 'trace_windows': []}
            for key, ds in g.items():
                if isinstance(ds, h5py.Dataset):
                    record['arrays'][key] = {'shape': list(ds.shape), 'dtype': str(ds.dtype),
                                             'attrs': attr_record(ds)}
            if 'timebase_sec' not in g:
                record['error'] = 'Missing timebase_sec'
                info['dmds'][dmd] = record
                continue
            time = g['timebase_sec']; dmd_t[dmd] = time
            if len(time.shape) != 1:
                record['error'] = 'timebase_sec is not 1-D'; info['dmds'][dmd] = record; continue
            ep = g.get('sample_epoch')
            if ep is not None and (ep.ndim != 1 or ep.shape[0] != time.shape[0]):
                record['error'] = 'sample_epoch shape mismatch'; info['dmds'][dmd] = record; continue
            rate = None
            meta = record['attrs'].get('reconstruction_metadata_json')
            if isinstance(meta, str):
                try:
                    md = json.loads(meta)
                    rate = md.get('nominal_sample_rate_hz', md.get('sample_rate_hz'))
                    rate = float(rate) if rate is not None else None
                    if rate and math.isfinite(rate) and rate > 0:
                        record['nominal_rate_source'] = 'reconstruction_metadata_json'
                    else:
                        rate = None
                except (ValueError, TypeError):
                    rate = None
            expected_dt = 1.0 / rate if rate else None
            if expected_dt is None and not skip_full_scan and time.shape[0] > 1:
                # Derive a diagnostic reference cadence from a bounded prefix.
                # This is *not* promoted to an authoritative acquisition rate.
                limit = min(int(time.shape[0]), 100000)
                prefix_t = np.asarray(time[:limit], dtype=np.float64)
                prefix_e = (np.asarray(ep[:limit], dtype=np.int64) if ep is not None
                            else np.ones(limit, dtype=np.int64))
                valid = ((prefix_e[1:] == prefix_e[:-1]) &
                         np.isfinite(prefix_t[1:]) & np.isfinite(prefix_t[:-1]))
                positive = np.diff(prefix_t)[valid]
                positive = positive[positive > 0]
                if positive.size:
                    expected_dt = float(np.median(positive))
                    record['nominal_rate_source'] = 'prefix_median_diagnostic_only'
            if expected_dt is not None:
                record['reference_rate_hz'] = 1.0 / expected_dt
            record['timebase_scan'] = ({'skipped': True} if skip_full_scan else
                                        scan_timebase(time, ep, chunk=chunk, expected_dt=expected_dt))
            record['selected_windows'] = sample_times(z, time, ep, dmd)
            n = len(time)
            # Small original, unfiltered data snippets for physiology plots.
            for signal in ('raw_f', 'dff'):
                if signal not in g:
                    continue
                ds = g[signal]
                if ds.ndim != 2 or ds.shape[1] != n:
                    info['warnings'].append(f'{dmd}/{signal}: expected (roi,time); got {ds.shape}')
                    continue
                for location,(a,b) in select_signal_ranges(n, trace_window_samples).items():
                    key=f'derived/{dmd}_{signal}_{location}.npy'
                    zip_npy(z, key, ds[:min(roi_limit, ds.shape[0]), a:b])
                    tk = f'derived/{dmd}_trace_time_{location}.npy'
                    if signal == 'raw_f':
                        zip_npy(z, tk, time[a:b])
                        if ep is not None:
                            zip_npy(z, f'derived/{dmd}_trace_epoch_{location}.npy', ep[a:b])
                    record['trace_windows'].append({'path': key, 'signal': signal,
                                                    'sample_start': a, 'sample_stop_exclusive': b,
                                                    'n_rois': min(roi_limit, ds.shape[0]),
                                                    'time_file': tk})
            # Export the complete compact trial-index arrays (not the full timebase).
            for key in ('trial_lengths_samples','trial_starts_sec','trial_valid_mask','trial_epoch','roi_ids'):
                if key in g and g[key].ndim == 1 and g[key].size < 100000:
                    out = f'derived/{dmd}_{key}.npy'
                    zip_npy(z, out, g[key][:])
                    record[key] = out
            if 'trial_lengths_samples' in g:
                lengths = np.asarray(g['trial_lengths_samples'][:], dtype=np.int64)
                if np.any(lengths < 0) or lengths.sum() != n:
                    record['warning'] = ('trial_lengths_samples has negative values or its sum '
                                         'does not equal retained timebase length')
                else:
                    offsets = np.r_[0, np.cumsum(lengths)]
                    # Source and derived samples at EVERY retained trial boundary.
                    boundary = []
                    for i, (a,b) in enumerate(zip(offsets[:-1], offsets[1:]), start=1):
                        if b <= a:
                            continue
                        for label, k in [('start', int(a)), ('end', int(b-1))]:
                            clip_a=max(int(a), k-boundary_samples+1) if label == 'end' else k
                            clip_b=min(int(b), k+boundary_samples) if label == 'start' else k+1
                            obj={'trial': i, 'edge': label, 'sample_idx': k,
                                 'range_start': clip_a, 'range_stop_exclusive': clip_b,
                                 'time_sec': float(time[k]),
                                 'prior_sample_time_sec': float(time[k-1]) if k > 0 else None,
                                 'next_sample_time_sec': float(time[k+1]) if k+1 < n else None}
                            if ep is not None:
                                obj['epoch'] = int(ep[k])
                            for signal in ('raw_f', 'dff'):
                                if signal in g and g[signal].ndim == 2 and g[signal].shape[1] == n:
                                    arr=g[signal][:min(roi_limit,g[signal].shape[0]),clip_a:clip_b]
                                    archive=f'derived/boundaries/{dmd}_trial_{i:04d}_{label}_{signal}.npy'
                                    zip_npy(z, archive, arr)
                                    obj[signal+'_path']=archive
                            boundary.append(obj)
                    record['trial_boundaries'] = boundary
            info['dmds'][dmd] = record
        if len(dmd_t) >= 2:
            a,b = dmd_t[groups[0]],dmd_t[groups[1]]
            if a.ndim != 1 or b.ndim != 1:
                info['dmd_comparison'] = {'error': 'non-1D timebase'}
            elif a.shape != b.shape:
                info['dmd_comparison'] = {'shape_equal': False,
                                          groups[0]+'_shape': list(a.shape), groups[1]+'_shape': list(b.shape)}
            elif skip_full_scan:
                info['dmd_comparison'] = {'skipped': True}
            else:
                unequal=0; mx=0.; first_difference=None
                for lo in range(0,a.shape[0],chunk):
                    x=np.asarray(a[lo:lo+chunk]); y=np.asarray(b[lo:lo+chunk])
                    neq = ~(x == y)
                    unequal += int(neq.sum())
                    if neq.any() and first_difference is None:
                        first_difference = lo + int(np.flatnonzero(neq)[0])
                    diff=np.abs(x.astype(float)-y.astype(float))
                    if np.isfinite(diff).any():
                        mx=max(mx,float(np.nanmax(diff)))
                info['dmd_comparison']={'shape_equal':True, 'n_compared':int(a.shape[0]),
                                        'n_nonidentical':unequal, 'first_nonidentical_index':first_difference,
                                        'max_abs_diff_sec':mx}
    return info


def export_session(asset: Any, output: Path, *, chunk: int=262144,
                   roi_limit: int=2, boundary_samples: int=32,
                   trace_window_samples: int=2048, skip_full_scan: bool=False,
                   include_harp_edges: bool=True, include_glu_ca: bool=False,
                   overwrite: bool=False) -> Dict[str, Any]:
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; pass --overwrite if intentional')
    if chunk < 1024 or roi_limit < 1 or boundary_samples < 1 or trace_window_samples < 1:
        raise ValueError('chunk>=1024 and ROI/sample limits must be positive')
    if not asset.session_dir.is_dir():
        raise FileNotFoundError(f'Session directory unavailable: {asset.session_dir}')
    output.parent.mkdir(parents=True, exist_ok=True)
    path_v = asset.get_asset('voltage', 'trace_h5')
    summary_v = asset.get_asset('voltage', 'summary_mat')
    derived_dir = asset.derived_dir / 'voltage'
    derived_list = sorted(derived_dir.glob('voltage_session_traces_*.h5')) if derived_dir.is_dir() else []
    if len(derived_list) > 1:
        # Prefer robust-F0 trial, then continuous; report all candidates.
        order = ['voltage_session_traces_dff_robust_f0_trial.h5',
                 'voltage_session_traces_dff_robust_f0_continuous.h5',
                 'voltage_session_traces_dff_static_f0_trial.h5']
        derived = sorted(derived_list, key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name))[0]
    else:
        derived = derived_list[0] if derived_list else None
    inventory = {'session_id': asset.session_id, 'subject_id': asset.subject_id,
                 'source_voltage_trace': file_record(path_v, asset.session_dir),
                 'source_voltage_summary': file_record(summary_v, asset.session_dir),
                 'derived_voltage_selected': file_record(derived, asset.session_dir),
                 'derived_voltage_candidates': [file_record(p, asset.session_dir) for p in derived_list],
                 'source_glutamate_calcium_summary': file_record(asset.summary_mat, asset.session_dir),
                 'harp_df': file_record(asset.harp_df_csv, asset.session_dir),
                 'mode': {'scan_all_derived_timestamps': not skip_full_scan,
                          'harp_edges_exported': include_harp_edges,
                          'glutamate_calcium_acquisition_metadata': include_glu_ca}}
    manifest: Dict[str, Any]={'schema_version':'1.0', 'export_utc':datetime.now(timezone.utc).isoformat(),
                            'session_id':asset.session_id, 'inventory':inventory,
                            'notes':['No recordings, raw .dat, full MATLAB .mat or full physiology H5 copied.',
                                     'Source extraction line indices are not downstream HARP timestamps.',
                                     'Only H5-derived timebase_sec is an explicit sampled downstream timebase.',
                                     'Trace windows are exact stored samples; no filtering, smoothing or interpolation.',
                                     'Source dataset position and DMD location must be interpreted using source summary metadata.',
                                     'DI3 duration can differ from entire SLAP2 acquisition due to acquisition-window clipping.',
                                     'Do NOT infer missing clock pulses from unequal total source/HARP cycle counts.'],
                            'status': {}}
    # Build as temporary file, rename only once zip is successfully complete.
    tmp=output.with_name(output.name + '.partial')
    if tmp.exists():
        raise FileExistsError(f'Partial output already exists: {tmp}')
    try:
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as z:
            write_qc_files(z, asset, manifest['status'])
            if include_harp_edges:
                if asset.harp_df_csv is not None and asset.harp_df_csv.is_file():
                    try:
                        manifest['harp_edges'] = harp_edges_csv(asset.harp_df_csv, z)
                    except Exception as exc:
                        manifest['status']['harp/di3_rising_edges.csv']=f'{type(exc).__name__}: {exc}'
                else:
                    manifest['status']['harp/di3_rising_edges.csv'] = 'HARP CSV missing'
            if path_v is not None and path_v.is_file():
                try:
                    zip_json(z, 'source/voltage_trace_inventory.json',
                             source_h5(z, path_v, roi_limit=roi_limit,
                                       boundary_samples=boundary_samples,
                                       trace_window_samples=trace_window_samples))
                except Exception as exc:
                    manifest['status']['source/voltage_trace_inventory.json'] = f'{type(exc).__name__}: {exc}'
            source_summary_metadata(asset,z,manifest['status'])
            if include_glu_ca:
                maybe_glu_summary(asset,z,manifest['status'])
            if derived is not None and derived.is_file():
                try:
                    zip_json(z, 'derived/voltage_audit.json', derived_h5(
                        z, derived, chunk=chunk, roi_limit=roi_limit,
                        boundary_samples=boundary_samples,
                        trace_window_samples=trace_window_samples,
                        skip_full_scan=skip_full_scan))
                except Exception as exc:
                    manifest['status']['derived/voltage_audit.json'] = f'{type(exc).__name__}: {exc}'
            else:
                manifest['status']['derived/voltage_audit.json'] = 'Derived voltage H5 missing'
            zip_json(z, 'manifest.json', manifest)
            z.writestr('README.txt', '''SLAP2–HARP sync audit, read-only export.
Open manifest.json first. Exported arrays are native float/boolean .npy and are
indexed by source/voltage_trace_inventory.json and derived/voltage_audit.json.
Sample axes and positions are explicit; do not concatenate source-trial and
downstream timestamps without validating clipping and trial layout.
The HARP edge train contains DI3 rises within recorded HARP time only.
The source acquisition may start earlier or end later.
No signal filtering or smoothing was performed.
''')
        tmp.replace(output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {'output': str(output), 'zip_bytes':output.stat().st_size,
            'status': manifest['status'], 'inventory': inventory}


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary-xlsx', type=Path, required=True, help='Path to VIP_SD_summary.xlsx')
    p.add_argument('--session-id', required=True, help='Exact registry session_id')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk',type=int,default=262144)
    p.add_argument('--roi-limit',type=int,default=2)
    p.add_argument('--boundary-samples',type=int,default=32)
    p.add_argument('--trace-window-samples',type=int,default=2048)
    p.add_argument('--skip-full-scan',action='store_true',help='Skip full continuity and DMD equality scans')
    p.add_argument('--no-harp-edges',action='store_true',help='Do not extract DI3 edge train')
    p.add_argument('--include-glu-ca',action='store_true',help='Export available SummaryLoCo acquisition metadata')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    registry=VIPSessionRegistry.from_excel(args.summary_xlsx)
    asset=registry.resolve_assets(args.session_id)
    result=export_session(asset,args.output,chunk=args.chunk,roi_limit=args.roi_limit,
                          boundary_samples=args.boundary_samples,
                          trace_window_samples=args.trace_window_samples,
                          skip_full_scan=args.skip_full_scan,
                          include_harp_edges=not args.no_harp_edges,
                          include_glu_ca=args.include_glu_ca,overwrite=args.overwrite)
    print(json.dumps(json_safe(result),indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
