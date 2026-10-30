"""Collect SLAP2 synchronization metadata without copying raw .dat/large trace arrays.

Usage (PowerShell):
 python collect_slap2_sync_metadata.py --session "\\\\allen\\aind\\scratch\\ophys\\Andrew\\VIP_synaptic_dynamics\\ASAP8\\863774\\863774_2026-08-04_08-12-47" --output "$env:USERPROFILE\\Desktop\\sync_metadata_863774.zip"

Only file names/sizes, metadata-like attributes, and HDF5 dataset schemas are exported.
This is *not* yet a validated cycle-count extractor; raw MAT v5 summaries are inventoried,
not opened. To protect privacy and keep outputs small, arbitrary HDF5 dataset contents
and user files are never copied. Review the zip before sharing.
"""
from __future__ import annotations
import argparse, json, os, re, zipfile
from pathlib import Path

EXCLUDE_DIRS={'.git','figures','notebooks','__pycache__'}
CANDIDATE_SUFFIXES={'.h5','.hdf5','.mat','.json','.csv','.dat','.meta','.txt'}
MAX_FILES=300000
KEYS=re.compile(r'(cycle|line|epoch|clock|tim|frame|rate|dmd|duration|offset|start|end|sync|trial)',re.I)

def safe_value(v):
 try:
  import numpy as np
  if isinstance(v,(bytes,bytearray)):return v.decode('utf-8','replace')[:300]
  if isinstance(v,np.ndarray):
   if v.size>30:return {'shape':list(v.shape),'dtype':str(v.dtype)}
   return safe_value(v.tolist())
  if isinstance(v,(np.integer,np.floating,np.bool_)):return v.item()
 except ImportError:pass
 if isinstance(v,(int,float,str,bool)) or v is None:return str(v)[:300] if isinstance(v,str) else v
 if isinstance(v,(list,tuple)):return [safe_value(i) for i in v[:30]]
 return str(v)[:300]

def inspect_h5(path,limit=3000):
 try:import h5py
 except ImportError:return {'error':'h5py missing; install with python -m pip install h5py'}
 output={'objects':[]}
 try:
  with h5py.File(path,'r') as f:
   def walk(k,obj):
    if len(output['objects'])>=limit:return
    d={'name':k,'type':'dataset' if isinstance(obj,h5py.Dataset) else 'group'}
    if isinstance(obj,h5py.Dataset):
     d.update(shape=list(obj.shape),dtype=str(obj.dtype),compression=obj.compression)
     # Tiny numeric metadata datasets only; do not extract arbitrary physiology traces
     if KEYS.search(k) and obj.size <= 30 and obj.dtype.kind in 'iufb':
      try:d['small_metadata_value']=safe_value(obj[()])
      except Exception:pass
    attrs={k:safe_value(v) for k,v in obj.attrs.items() if KEYS.search(k)}
    if attrs:d['sync_attrs']=attrs
    output['objects'].append(d)
   walk('/',f)
   f.visititems(walk)
 except Exception as e: output['error']=f'{type(e).__name__}: {e}'
 return output

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--session',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);pa.add_argument('--h5-max-mb',type=int,default=4096)
 args=pa.parse_args();root=args.session
 if not root.is_dir():pa.error(f'Not a directory: {root}')
 inventory=[];schemas={};count=0
 for dirpath,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if d.lower() not in EXCLUDE_DIRS]
  for name in files:
   count+=1
   if count>MAX_FILES:raise RuntimeError('File safety limit exceeded')
   path=Path(dirpath)/name;rel=path.relative_to(root).as_posix()
   try: size=path.stat().st_size
   except OSError:continue
   if path.suffix.lower() not in CANDIDATE_SUFFIXES:continue
   inventory.append({'path':rel,'bytes':size})
   if path.suffix.lower() in {'.h5','.hdf5','.mat'} and size <= args.h5_max_mb*1024**2:
    # h5py supports MATLAB v7.3 HDF5 .mat only. MAT v5 is inventoried.
    try:
     import h5py
     if h5py.is_hdf5(str(path)):schemas[rel]=inspect_h5(path)
    except ImportError:schemas[rel]={'error':'h5py not installed'}
 if not any('/analysis/qc' in '/'+i['path'] for i in inventory):print('WARNING: no analysis/qc assets found')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('inventory.json',json.dumps({'session_name':root.name,'files':inventory},indent=2))
  z.writestr('hdf5_schemas.json',json.dumps(schemas,indent=2))
  z.writestr('README.txt','Metadata inventory; no raw trace arrays or raw .dat bytes included. Check for paths/metadata you do not wish to share.\n')
 print(f'Created {args.output}: {len(inventory)} file records, {len(schemas)} HDF5 schemas')
if __name__=='__main__':main()
