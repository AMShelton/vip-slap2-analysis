import os
import glob
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from read_harp import HarpReader
from scipy.signal import medfilt,find_peaks
from NPUltra_analysis.plotting import normalize

def process_harp_sessions(harp_root_dir, save=True, overwrite=False):
    """
    Process all subdirectories in `harp_root_dir` that contain HARP binary data.

    Parameters:
    -----------
    harp_root_dir : str or Path
        Path to the parent directory containing session folders.
    save : bool
        If True, saves .pkl files for encoder, photodiode, licks, and rewards.
    overwrite : bool
        If True, existing extracted files will be overwritten.
    """
    harp_root_dir = Path(harp_root_dir)
    session_dirs = [d for d in harp_root_dir.iterdir() if d.is_dir()]

    for session in session_dirs:
        try:
            print(f"Processing {session}...")
            reader = HarpReader(session)
            extracted_dir = session / "extracted_files"
            if extracted_dir.exists() and not overwrite:
                print(f"→ Skipping {session.name}: already processed.")
                continue

            if save:
                extracted_dir.mkdir(exist_ok=True)
                reader.get_encoder.to_pickle(extracted_dir / 'encoder.pkl')
                reader.get_photodiode.to_pickle(extracted_dir / 'photodiode.pkl')
                reader.get_licks.to_pickle(extracted_dir / 'licks.pkl')
                reader.get_rewards.to_pickle(extracted_dir / 'rewards.pkl')
                print(f"→ Saved data to {extracted_dir}")
            else:
                print(reader.get_encoder.head())
                print(reader.get_photodiode.head())
                print(reader.get_licks.head())
                print(reader.get_rewards.head())
        except Exception as e:
            print(f"❌ Error processing {session.name}: {e}")

def process_single_harp_session(session_path, save=True, overwrite=False):
    """
    Process a single HARP session folder.

    Parameters:
    -----------
    session_path : str or Path
        Path to the directory containing HARP binary files.
    save : bool
        If True, saves .pkl files for encoder, photodiode, licks, and rewards.
    overwrite : bool
        If True, existing extracted files will be overwritten.
    """
    session_path = Path(session_path)
    try:
        print(f"Processing {session_path}...")
        reader = HarpReader(session_path)
        extracted_dir = session_path / "extracted_files"
        
        if extracted_dir.exists() and not overwrite:
            print(f"→ Skipping {session_path.name}: already processed.")
            return

        if save:
            extracted_dir.mkdir(exist_ok=True)
            reader.get_encoder.to_pickle(extracted_dir / 'encoder.pkl')
            reader.get_photodiode.to_pickle(extracted_dir / 'photodiode.pkl')
            reader.get_licks.to_pickle(extracted_dir / 'licks.pkl')
            reader.get_rewards.to_pickle(extracted_dir / 'rewards.pkl')
            print(f"→ Saved data to {extracted_dir}")
        else:
            print(reader.get_encoder.head())
            print(reader.get_photodiode.head())
            print(reader.get_licks.head())
            print(reader.get_rewards.head())
    except Exception as e:
        print(f"❌ Error processing {session_path.name}: {e}")

def get_signal_edges(signal,time,est_rate = 30):
    # Light smoothing against noise; adjust kernel if your PD is already clean.
    y_s = medfilt(signal, kernel_size=5) if len(signal)>=5 else signal.copy()

    # Auto-threshold at mid of bimodal distribution
    thr = (np.percentile(y_s, 95) + np.percentile(y_s, 5)) / 2.0
    binary = (y_s > thr).astype(int)

    # Rising edges: transitions 0->1
    db = np.diff(binary, prepend=binary[0])
    rise_idx = np.where(db==1)[0]
    t_rise = time[rise_idx]

    # Falling edges: transitions 1->0
    fall_idx = np.where(-db==1)[0]
    t_fall = time[fall_idx]

    # Optional: collapse spurious multiple edges within a frame (debounce)
    # Merge edges closer than, say, 5 ms (typical if photodiode bounces)
    if len(t_rise)>1:
        keep = [0]
        for i in range(1, len(t_rise)):
            if (t_rise[i] - t_rise[keep[-1]]) > 0.005:
                keep.append(i)
        t_rise = t_rise[keep]
        
    if len(t_fall)>1:
        keep = [0]
        for i in range(1, len(t_fall)):
            if (t_fall[i] - t_fall[keep[-1]]) > 0.005:
                keep.append(i)
        t_fall = t_fall[keep]
        
    estimated_rate = est_rate #Hz

    cycle_rates = []

    for start,stop in zip(t_rise,t_fall):
        dt = np.diff([start,stop])[0]
        n_frames = estimated_rate/dt
        cycle_rates.append(n_frames)
    avg_rate = np.median(cycle_rates)    

    print(f'Median frame rate of LCD screen: {avg_rate:.6} Hz')

    return (rise_idx,fall_idx,t_rise,t_fall,avg_rate)

def get_time_offset(photodiode_df,modulo=60):

    #Calculate the offset time between the first photodiode flip and the first image presentation
    pd_time = photodiode_df.index - photodiode_df.index[0]
    pd_signal = photodiode_df['AnalogInput0'].values

    signal_metrics = get_signal_edges(pd_signal,pd_time,est_rate=30)

    flip_time = signal_metrics[2][0]
    avg_rate = signal_metrics[-1]

    offset_time =  flip_time - (1/avg_rate)*modulo
    print(f'Time offset of image presentation from photodiode signal: {offset_time:.4} seconds')

    return offset_time

def correct_event_log(stimulus_df,photodiode_df,savepath = None):

    #Update stimulus_df to contain information about the first stimulus shown (for whatever reason these aren't logged in the trial log)
    tif_row = stimulus_df[stimulus_df['Value'].str.contains('.tif', case=False, na=False)].iloc[0]
    tif_value = tif_row['Value']
    new_row = pd.DataFrame([{'Frame': -1, 'Timestamp': 0.0, 'Value': 'Frame'}])
    new_row_1 = pd.DataFrame([{'Frame': -1, 'Timestamp': 0.0, 'Value': tif_value}])
    stimulus_df.index = stimulus_df.index + 1
    stimulus_df = pd.concat([new_row, new_row_1, stimulus_df]).reset_index(drop=True)  

    offset_time = get_time_offset(photodiode_df)  

    stimulus_df['corrected_timestamp'] = stimulus_df['Timestamp'] + offset_time

    if savepath:
        stimulus_df.to_csv(savepath)
        print('Saved stimulus table to savepath')

    return stimulus_df

