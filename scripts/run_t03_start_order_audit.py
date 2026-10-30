"""CLI for read-only T0.3 acquisition-order audit (run from repo root)."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser(description='Read-only SLAP2-versus-HARP start-order census')
    p.add_argument('--summary-xlsx',type=Path,help='VIP_SD_summary.xlsx on VAST')
    p.add_argument('--manifest',type=Path,help='Optional directory_manifest.tsv file-presence census')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--all-registry-sessions',action='store_true',help='Default filters to A0,A1,A2,B0,B1,B2')
    p.add_argument('--inspect-harp-csv',action='store_true',help='Optionally read first HARP_df rows for initial DI3-low interval')
    p.add_argument('--cross-pc-bound-s',type=float,default=3.0)
    p.add_argument('--meta-hardware-delay-bound-s',type=float,default=3.0)
    p.add_argument('--harp-folder-bound-s',type=float,default=2.0)
    p.add_argument('--launcher-to-harp-bound-s',type=float,default=6.0)
    args=p.parse_args()
    if args.manifest is None and args.summary_xlsx is None:p.error('Supply --summary-xlsx and/or --manifest')
    # Local development fallback if script copied without installing the package.
    home=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(home/'src'))
    from vip_slap2_analysis.metadata.t03_start_order import audit_registry,manifest_coverage
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.manifest:
        cover=manifest_coverage(args.manifest)
        cover.to_csv(args.output_dir/'t03_manifest_presence.csv',index=False)
        print('Manifest-only session identifiers:',len(cover))
        print(cover.drop(columns='session_id').sum().to_string())
        print('The manifest does NOT contain the timing values needed to classify order.')
    if args.summary_xlsx:
        audit=audit_registry(args.summary_xlsx,args.output_dir,inspect_harp_csv=args.inspect_harp_csv,
            session_types=None if args.all_registry_sessions else ['A0','A1','A2','B0','B1','B2'],
            cross_pc_bound_s=args.cross_pc_bound_s,
            meta_hardware_delay_bound_s=args.meta_hardware_delay_bound_s,
            harp_folder_bound_s=args.harp_folder_bound_s,
            launcher_to_harp_bound_s=args.launcher_to_harp_bound_s)
        print('\nStart-order audit:',len(audit),'registry sessions')
        if not audit.empty:
            print(audit['order_classification'].value_counts(dropna=False).to_string())
            print('Wrote:',args.output_dir/'t03_start_order_evidence.csv')

if __name__=='__main__':main()
