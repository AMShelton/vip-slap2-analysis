import json

from vip_slap2_analysis.common.clock_qc import compare_slap2_harp_clock


class FakeSummary:
    def __init__(self, dmds):
        self._dmds = dmds
        self.n_dmds = len(dmds)

    def get_dmd_metadata(self, dmd):
        return dict(self._dmds[dmd - 1]["metadata"])

    def get_dmd_epoch_metadata(self, dmd):
        return [dict(row) for row in self._dmds[dmd - 1].get("epochs", [])]


def _write_detection(tmp_path, *, n_edges, period_s, accepted_counts=None):
    candidates = []
    for i, count in enumerate(accepted_counts or [], start=1):
        candidates.append(
            {
                "source_epoch_index": i,
                "accepted": True,
                "n_pulses": int(count),
                "duration_s": float(max(count - 1, 0) * period_s),
            }
        )
    payload = {
        "n_rising_edges": int(n_edges),
        "median_period_s": float(period_s),
        "candidate_epochs": candidates,
    }
    (tmp_path / "di3_pulse_train_detection.json").write_text(json.dumps(payload))


def test_modern_voltage_direct_dmd1_cycle_clock_and_session_count(tmp_path):
    line_rate = 10686.6669921875
    dmd1_cycles = [11881, 6748, 23749, 24188, 2445, 4294, 6003]
    harp_accepted = [11689, 6748, 23750, 24189, 2446, 4295, 5643]
    summary = FakeSummary(
        [
            {
                "metadata": {"lineRateHz": line_rate, "linesPerCycle": 227},
                "epochs": [
                    {"epochIdx": i, "nCycles": n, "linesPerCycle": 227, "available": True}
                    for i, n in enumerate(dmd1_cycles, start=1)
                ],
            },
            {
                "metadata": {"lineRateHz": line_rate, "linesPerCycle": 233},
                "epochs": [
                    {"epochIdx": i, "nCycles": 1, "linesPerCycle": 233, "available": True}
                    for i in range(1, 8)
                ],
            },
        ]
    )
    period = 227.0 / line_rate
    _write_detection(
        tmp_path,
        n_edges=79309,
        period_s=period,
        accepted_counts=harp_accepted,
    )

    report = compare_slap2_harp_clock(summary, behavior_qc_dir=tmp_path)

    assert report["available"]
    assert report["relationship"] == "direct_dmd_cycle_clock"
    assert report["reference_dmd"] == 1
    assert report["direct_cycle_count_comparison_valid"]
    assert report["session_cycle_count"]["slap2_cycles"] == 79308
    assert report["session_cycle_count"]["harp_pulses"] == 79309
    assert report["session_cycle_count"]["difference_cycles"] == 1
    assert [row["difference_cycles"] for row in report["accepted_epoch_cycle_counts"]][1:6] == [0, 1, 1, 1, 1]


def test_historical_glutamate_reports_integer_line_cadence_not_cycle_loss(tmp_path):
    line_rate = 10686.6669921875
    summary = FakeSummary(
        [
            {
                "metadata": {"lineRateHz": line_rate, "linesPerCycle": 210},
                "epochs": [{"epochIdx": 1, "nCycles": 88547, "linesPerCycle": 210, "available": True}],
            },
            {
                "metadata": {"lineRateHz": line_rate, "linesPerCycle": 205},
                "epochs": [{"epochIdx": 1, "nCycles": 90707, "linesPerCycle": 205, "available": True}],
            },
        ]
    )
    period = 211.0 / line_rate
    _write_detection(tmp_path, n_edges=88128, period_s=period)

    report = compare_slap2_harp_clock(summary, behavior_qc_dir=tmp_path)

    assert report["available"]
    assert report["reference_dmd"] == 1
    assert report["relationship"] == "integer_line_cadence_not_dmd_cycle"
    assert report["nearest_integer_line_cadence"] == 211
    assert not report["direct_cycle_count_comparison_valid"]
    assert "session_cycle_count" not in report


def test_modern_glutamate_calcium_session_matches_dmd1_cycle_clock(tmp_path):
    line_rate = 10686.6669921875
    summary = FakeSummary(
        [
            {
                "metadata": {"lineRateHz": line_rate, "linesPerCycle": 130},
                "epochs": [{"epochIdx": 1, "nCycles": 147970, "linesPerCycle": 130, "available": True}],
            },
            {
                "metadata": {"lineRateHz": line_rate, "linesPerCycle": 138},
                "epochs": [{"epochIdx": 1, "nCycles": 139392, "linesPerCycle": 138, "available": True}],
            },
        ]
    )
    # The observed HARP timestamp grid from this session corresponds to roughly
    # 129.95 line periods per pulse, close enough to the 130-line DMD1 cycle.
    period = 0.012159999925643206
    _write_detection(tmp_path, n_edges=147970, period_s=period)

    report = compare_slap2_harp_clock(summary, behavior_qc_dir=tmp_path)

    assert report["available"]
    assert report["reference_dmd"] == 1
    assert report["relationship"] == "direct_dmd_cycle_clock"
    assert report["session_cycle_count"]["difference_cycles"] == 0
