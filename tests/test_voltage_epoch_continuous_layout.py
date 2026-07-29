import h5py
import numpy as np

from vip_slap2_analysis.voltage.summary import VoltageSummary


def test_epoch_continuous_layout_discovery(tmp_path):
    p = tmp_path / "traces.h5"
    with h5py.File(p, "w") as h5:
        h5.create_dataset("traces/epochs/epoch_0001/DMD1", data=np.zeros((10, 2)))
        h5.create_dataset("traces/epochs/epoch_0002/DMD1", data=np.zeros((20, 2)))

    vs = VoltageSummary.__new__(VoltageSummary)
    vs._h5 = h5py.File(p, "r")
    vs._summary_layout = "split_h5"
    vs.n_rois = [2]
    vs.n_dmds = 1
    try:
        assert vs._epoch_continuous_dataset_keys() == [
            (1, 1, "traces/epochs/epoch_0001/DMD1"),
            (2, 1, "traces/epochs/epoch_0002/DMD1"),
        ]
        assert vs.available_trace_modes() == ["epoch_continuous"]
    finally:
        vs._h5.close()
