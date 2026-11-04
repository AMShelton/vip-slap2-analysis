"""Print the key fields of saved cycle_registration_DMD*.json files.

    python scripts/show_cycle_registration.py "Z:\\VIP_synaptic_dynamics\\ASAP8\\863774\\863774_2026-08-0*\\analysis\\qc\\sync\\cycle_registration_DMD*.json"
"""
import glob
import json
import sys

TOP = ("session_id", "dmd", "status", "relationship", "acquisition_start_order", "start_order_evidence",
       "effective_lines_per_pulse", "lines_per_cycle", "line_rate_hz", "nominal_cycle_period_s",
       "observed_cycle_period_s", "period_error_ppm", "harp_n_rising_edges", "harp_record_start_s",
       "harp_record_end_s", "n_source_epochs", "n_di3_epochs", "limitations")
EPOCH = ("source_epoch_index", "di3_epoch_index", "n_source_cycles", "n_observed_pulses",
         "n_internal_missing_pulses", "internal_gaps", "lead_s", "tail_s", "anchor", "status",
         "cycle_index_offset", "candidate_offset_range", "first_pulse_harp_s", "last_pulse_harp_s",
         "residual_rms_s", "notes")

for pattern in sys.argv[1:]:
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        out = {k: d.get(k) for k in TOP}
        out["inputs"] = {k: v for k, v in d.get("inputs", {}).items() if k != "summary_epochs"}
        out["summary_epochs"] = d.get("inputs", {}).get("summary_epochs")
        out["epochs"] = [{k: e.get(k) for k in EPOCH} for e in d.get("epochs", [])]
        print("=" * 100)
        print(f)
        print(json.dumps(out, indent=1, default=str))
