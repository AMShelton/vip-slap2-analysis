import numpy as np

def concat_rois_across_trials(vs, dmd: int = 1, drop_discarded: bool = True, dtype=np.float32):
    """
    Concatenate ROI traces across all valid trials for a given DMD.

    Assumes vs.get_roi_traces(dmd, trial) returns shape (n_samples, n_rois)
    i.e. samples first, ROI second (as your screenshot shows).

    Returns
    -------
    roi_traces : list[np.ndarray]
        List length n_rois. Each element is a 1D array of concatenated samples.
    trial_slices : list[tuple[int, slice]]
        (trial_index_1based, slice_into_concatenated_time) for bookkeeping.
    """
    dmd0 = dmd - 1
    valid_trials = vs.valid_trials[dmd0]  # should be 1-indexed
    n_rois = vs.n_rois[dmd0]

    chunks_by_roi = [[] for _ in range(n_rois)]
    trial_slices = []
    t_cursor = 0

    for trial in valid_trials:
        X = vs.get_roi_traces(dmd=dmd, trial=trial, drop_discarded=False, dtype=dtype)
        # X: (n_samples, n_rois)

        if drop_discarded:
            df = vs.get_discard_frames(dmd=dmd, trial=trial)
            df = np.asarray(df).astype(bool).squeeze()
            if df.ndim != 1:
                df = df.reshape(-1)

            # Align just in case of minor shape weirdness
            if df.size != X.shape[0]:
                df = df[:X.shape[0]] if df.size > X.shape[0] else np.pad(df, (0, X.shape[0] - df.size), constant_values=False)

            X = X[~df, :]  # keep non-discarded samples

        seg_len = X.shape[1]
        trial_slices.append((trial, slice(t_cursor, t_cursor + seg_len)))
        t_cursor += seg_len

        # Append per ROI (column)
        for r in range(n_rois):
            chunks_by_roi[r].append(X[r,:])

    roi_traces = [
        np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=dtype)
        for chunks in chunks_by_roi
    ]

    return roi_traces, trial_slices