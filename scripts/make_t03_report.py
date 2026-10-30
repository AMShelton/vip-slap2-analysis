"""Plots/report from a local T0.3 start-order audit CSV. No new data reads."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate(csv:Path,out:Path):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(csv)
    if 'order_classification' not in df.columns:
        raise ValueError('Input must be the full t03_start_order_evidence.csv; manifest-only presence is not timing evidence')
    fig_paths=[]
    def finish(fig,name,notes):
        fig.tight_layout(rect=(0,0.09,1,0.98))
        fig.text(.02,.035,notes,fontsize=8,ha='left',va='bottom',wrap=True)
        for ext in ('png','svg'):
            p=out/(name+'.'+ext);fig.savefig(p,dpi=250,bbox_inches='tight')
        fig_paths.append(name);plt.close(fig)
    statuses=['slap2_first','harp_first','ambiguous_overlap','missing_timing_evidence']
    counts=df['order_classification'].fillna('missing_timing_evidence').value_counts()
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.bar(range(len(statuses)),[counts.get(k,0) for k in statuses])
    ax.set_xticks(range(len(statuses)),['SLAP2 first','HARP first','Ambiguous','Missing'],rotation=15,ha='right')
    ax.set_ylabel('Registry sessions');ax.set_title('T0.3 acquisition start order: evidence classification')
    for i,k in enumerate(statuses):ax.text(i,counts.get(k,0)+.15,str(counts.get(k,0)),ha='center')
    finish(fig,'01_order_coverage','SOURCE: session registry + SLAP2 .meta header + behavior launcher log / HARP recording-folder name.\n'
           'MEANING: conservative acquisition-order classification with configurable assumptions about PC clock skew and software delay.')
    good=df.dropna(subset=['estimated_harp_minus_source_s','assumed_timing_bound_s']).copy()
    if len(good):
        good['date']=pd.to_datetime(good['session_id'].str.extract(r'_(\d{4}-\d{2}-\d{2})_')[0])
        good=good.sort_values('date')
        fig,ax=plt.subplots(figsize=(11,5))
        x=np.arange(len(good));y=good['estimated_harp_minus_source_s'].to_numpy(float)
        e=good['assumed_timing_bound_s'].to_numpy(float)
        ax.errorbar(x,y,yerr=e,fmt='.',capsize=1.5,elinewidth=.7,markersize=5)
        ax.axhline(0,color='gray',lw=1);ax.set_ylabel('HARP start − SLAP2 metadata time (s)')
        ax.set_xlabel('Sessions ordered chronologically');ax.set_title('Estimated start-time difference and uncertainty bounds')
        finish(fig,'02_timing_estimates','SOURCE: .meta MAT-file creation time and Bonsai launcher / HARP folder timestamp.\n'
             'MEANING: positive = SLAP2 nominally first. Error bars represent analyst-configured clock/delay bounds, NOT statistical confidence intervals.')
        fig,ax=plt.subplots(figsize=(9,5))
        for status,group in good.groupby('order_classification'):
            vals=group['estimated_harp_minus_source_s'].dropna()
            if len(vals):ax.hist(vals,bins=min(max(len(vals)//2,5),25),alpha=.45,label=status)
        ax.axvline(0,color='gray',lw=1);ax.set_xlabel('HARP start − SLAP2 .meta header (s)')
        ax.set_ylabel('Sessions');ax.set_title('Start offset distributions across sessions');ax.legend(fontsize=8)
        finish(fig,'03_delta_hist','SOURCE: same wall-clock candidates as Figure 2.\n'
             'MEANING: distribution is NOT the measured SLAP2–HARP hardware registration error.')
    if 'initial_cycle_offset_estimate' in df.columns:
        avail=df.dropna(subset=['initial_cycle_offset_estimate','initial_cycle_offset_lower','initial_cycle_offset_upper'])
        if len(avail):
            fig,ax=plt.subplots(figsize=(10,5))
            x=avail['initial_cycle_offset_estimate'].to_numpy(float)
            widths=avail['initial_cycle_offset_upper'].to_numpy(float)-avail['initial_cycle_offset_lower'].to_numpy(float)
            ax.scatter(x,widths,alpha=.6)
            ax.set_xlabel('Initial source cycles preceding first HARP pulse: estimated count')
            ax.set_ylabel('Number of candidate cycle indices within assumed timing bounds')
            ax.set_title('Integer-cycle ambiguity in retrospective alignment')
            finish(fig,'04_integer_cycle_uncertainty',
                'SOURCE: estimated PC-clock delta / acquisition cycle period from DMD1 metadata.\n'
                'MEANING: wide vertical range means integer cycle remains unidentifiable to one display frame; requires independent validation.')
    report=(out/'t03_report_summary.md')
    report.write_text('# T0.3 acquisition start order — audit summary\n\n'
        f'Sessions evaluated: {len(df)}\n\n'
        + '\n'.join(f'- {k}: {int(counts.get(k,0))}' for k in statuses)
        + '\n\nThis reports inferred start order, not validated source-to-stimulus absolute alignment.\n'
        'Reprocessing requires a reviewed offset file and a verified source-sample correspondence.\n'
        '\nFigures: '+', '.join(fig_paths)+'\n')
    return fig_paths

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence-csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    print('Generated:',generate(args.evidence_csv,args.output_dir))
if __name__=='__main__':main()
