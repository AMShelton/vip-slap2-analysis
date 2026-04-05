import os
import glob
import json
import numpy as np
import pandas as pd
from read_harp import HarpReader
from scipy.signal import butter, filtfilt


"""
Utilities for SLAP2 time-series and image helpers.

This module contains small, focused helpers that are reused across preprocessing,
analysis, and plotting steps. Keep functions **pure** (no implicit file I/O)
except for `save_figure`.

Functions
---------
lowpass_filter : Zero-phase Butterworth low-pass filtering along time.
downsample     : Downsample time-first arrays by integer factor via mean.
normalize      : Clip to [0, max] and scale to [0, 1] with NaN safety.
save_figure    : Save a Matplotlib figure in multiple formats with stable PDF fonts.

Notes
-----
- Always pass sampling rates and sizes explicitly; avoid hidden constants.
- Prefer ndarray inputs with time on axis 0 (T, ...).
"""

def lowpass_filter(data, cutoff, fs=1.0, order=4):

    """
    Apply a zero-phase Butterworth low-pass filter along the time axis (axis=0).

    Parameters
    ----------
    data : ndarray
        Time-first array, e.g., shape (T, N) or (T, H, W).
    cutoff : float
        Cutoff frequency in Hz. Must satisfy 0 < cutoff < fs/2.
    fs : float, default=1.0
        Sampling rate in Hz.
    order : int, default=4
        Filter order.

    Returns
    -------
    ndarray
        Filtered array with the same shape as `data`.

    Raises
    ------
    ValueError
        If `cutoff` is not in (0, fs/2).
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def downsample(data, factor):

    """
    Downsample along the time axis by averaging non-overlapping windows.

    Truncates trailing frames that don't complete a full window.

    Parameters
    ----------
    data : ndarray
        Time-first array with shape (T, ...).
    factor : int
        Integer downsampling factor (>= 1).

    Returns
    -------
    ndarray
        Downsampled array with shape (T//factor, ...).
    """

    frames = data.shape[0]
    trimmed_frames = (frames // factor) * factor
    data = data[:trimmed_frames]
    return data.reshape(-1, factor, data.shape[1], data.shape[2]).mean(axis=1)
    
def normalize_image(image, max_val=None):

    """
    Normalize an image/array to [0, 1] with clipping and NaN safety.

    Parameters
    ----------
    image : ndarray
        Input array; NaNs are preserved (treated as zeros during scaling).
    max_val : float or None, optional
        Maximum value to normalize by. If None, uses `np.nanmax(image)`.
        If the maximum is <= 0 or NaN, returns zeros.

    Returns
    -------
    ndarray
        Float array in [0, 1] of same shape as input.
    """

    image = np.clip(image, 0, max_val)
    image = image / max_val
    return image

def normalize(arr):
    
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))

    return arr

def save_figure(fig, fname, formats = ['.pdf'],transparent=False,dpi=300,facecolor=None,**kwargs):

    """
    Save a Matplotlib figure in one or more formats with consistent PDF fonts.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    fname : str
        Output path **without** extension (extensions are taken from `formats`).
    formats : list[str] or tuple[str, ...], default (".pdf",)
        File extensions to write, e.g., [".pdf", ".png"].
    transparent : bool, default False
        Save with transparent background.
    dpi : int, default 300
        Dots per inch (for raster formats).
    figsize : tuple of (width, height) inches or None, default None
        If provided, sets figure size before saving.
    facecolor : str or None, default None
        Facecolor to apply when saving.

    Notes
    -----
    - Embeds fonts as Type 42 in PDFs for Illustrator compatibility.
    """

    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])

    elif 'figsize' in kwargs.keys():
        fig.set_size_inches(kwargs['figsize'])
    for f in formats:
        fig.savefig(fname + f, transparent = transparent,dpi=dpi)

def get_HARP_data(harp_path):
    
    harp_handler = HarpReader(harp_path)
    harp_df = harp_handler.reader.DigitalInputState.read()
    harp_df['time']=harp_df.index
    harp_df = harp_df.sort_values('time')
    harp_df = harp_df.reset_index(drop=True)
    acq_time = harp_df['time']-harp_df['time'].iloc[0]
    acq_time = acq_time.values
    
    return harp_df,acq_time

def get_stim_data(harp_dir,behavior_files = ['encoder.pkl','photodiode.pkl','licks.pkl','rewards.pkl']):
    stimulus_df = pd.read_csv(glob.glob(os.path.join(harp_dir,'**.csv'))[0])
    behavior_dfs = []
    for filename in behavior_files:
        behavior_dfs.append(pd.read_pickle(glob.glob(os.path.join(harp_dir,'**',filename),recursive=True)[0])) 
    encoder_df, photodiode_df, licks_df, rewards_df = behavior_dfs
    
    return stimulus_df,encoder_df, photodiode_df, licks_df, rewards_df

def tolerant_mean(arrs, avg='mean', ddof=0, return_counts=False):
    """
    NaN-aware averaging across a ragged list of 1D arrays by aligning along their first axis.
    At each time index, only arrays that have a value at that index contribute.
    Any NaNs in the contributing arrays are excluded (masked) as well.

    Parameters
    ----------
    arrs : list[np.ndarray]
        Ragged list of 1D arrays (possibly containing NaNs) to average across.
    avg : {'mean','median'}
        Aggregation type across arrays at each index.
    ddof : int
        Delta degrees of freedom for std (0 by default; use 1 for sample std).
    return_counts : bool
        If True, also return the effective sample count per index (after masking).

    Returns
    -------
    mean : np.ndarray (shape: [max_len])
    std  : np.ndarray (shape: [max_len])
    (optional) counts : np.ndarray (shape: [max_len]) number of non-masked entries per index
    """
    if len(arrs) == 0:
        out = (np.array([]), np.array([]))
        return (*out, np.array([])) if return_counts else out

    # Ensure float dtype and compute sizes
    arrs = [np.asarray(a, dtype=float) for a in arrs]
    lens = [len(a) for a in arrs]
    max_len = max(lens)
    n = len(arrs)

    # Build masked array [max_len, n], initially fully masked
    M = np.ma.empty((max_len, n), dtype=float)
    M.mask = True  # everything masked to start

    # Fill columns; mask positions beyond current array length and NaNs within-range
    for j, a in enumerate(arrs):
        L = len(a)
        if L == 0:
            continue
        M[:L, j] = a
        # additionally mask any NaNs that were written
        nan_mask = np.isnan(a)
        if nan_mask.any():
            M.mask[:L, j] = M.mask[:L, j] | nan_mask

    # Aggregate across arrays (axis=1) with mask awareness
    if avg == 'mean':
        mean = np.ma.mean(M, axis=1).filled(np.nan)
        std  = np.ma.std(M,  axis=1, ddof=ddof).filled(np.nan)
    elif avg == 'median':
        mean = np.ma.median(M, axis=1).filled(np.nan)
        # std here is still the (masked) standard deviation; swap for robust spread if desired
        std  = np.ma.std(M, axis=1, ddof=ddof).filled(np.nan)
    else:
        raise ValueError("avg must be 'mean' or 'median'")

    if return_counts:
        counts = (~M.mask).sum(axis=1).filled(0).astype(int)
        return mean, std, counts
    else:
        return mean, std
    
def load_pkg_from_npz(npz_path, json_path=None):
    """
    Load a session 'pkg' dict from an .npz saved with allow_pickle=True.
    If a sibling .json is provided (or found automatically), merge top-level keys
    (e.g., 'im_rate_Hz', 'prepost_sec', 'stim_times', etc.) into the pkg.

    Returns
    -------
    pkg : dict
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=True) as z:
        # Common patterns:
        # 1) A single 'pkg' object stored
        if 'pkg' in z.files:
            pkg = z['pkg'].item()
        else:
            # 2) Multiple arrays/objects—try to build a dict
            pkg = {}
            for k in z.files:
                arr = z[k]
                if arr.dtype == object and arr.size == 1:
                    # likely a pickled dict
                    try:
                        pkg[k] = arr.item()
                    except Exception:
                        pkg[k] = arr
                else:
                    pkg[k] = arr

    # Merge JSON metadata if provided or auto-detected
    if json_path is None:
        base, _ = os.path.splitext(npz_path)
        guess = base + ".json"
        if os.path.isfile(guess):
            json_path = guess
    if json_path is not None and os.path.isfile(json_path):
        with open(json_path, "r") as f:
            meta = json.load(f)
        # Shallow merge JSON → pkg (don’t overwrite existing keys unless needed)
        for k, v in meta.items():
            if k not in pkg:
                pkg[k] = v

    return pkg

def infer_ids_from_filename(path):
    """
    Heuristic: parse mouse/session from filename like '810196_2025-07-31_810196.npz'
    Returns (session_id, mouse_id)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    # You can make this smarter for your own naming scheme:
    session_id = stem
    # try to pull a leading integer as mouse_id
    parts = stem.split('_')
    mouse_id = None
    for p in parts:
        if p.isdigit():
            mouse_id = p
            break
    return session_id, mouse_id
