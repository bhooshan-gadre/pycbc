#!/usr/bin/env python
"""Attribute replay costs using per-rank stage profiles, never GPU dispatch alone."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def analyze(manifest, warmup=1):
    report=dict(runs=[], notes=[
        'Per-rank times overlap; do not add ranks or detectors to process wall time.',
        'Outside-loop time includes startup, imports, bank/device setup and shutdown; it is not exclusively compilation.',
        'Profiling synchronizes JAX stages and can reduce overlap. Use unprofiled repeats for final speedups.',
        'Rank 0 CPU preprocessing/coincidence may remain the critical path after worker GPU offload.'])
    for run in manifest['runs']:
        row={k:run[k] for k in ['mode','repeat','returncode','wall_s']}
        row['ranks']={}
        for path in sorted((Path(run['directory'])/'stages').glob('rank*.jsonl')):
            entries=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            blocks=[e for e in entries if e.get('phase','block')=='block']
            startup=next((e for e in entries if e.get('phase')=='startup'),{})
            metrics={}
            def append(key,value):
                metrics.setdefault(key,[]).append(float(value))
            for entry in blocks:
                append('conditioning_all_detectors_s',sum(d.get('advance_s',0)+d.get('psd_check_s',0) for d in entry.get('detectors',{}).values()))
                append('filtering_all_detectors_s',sum(d.get('filter_s',0) for d in entry.get('detectors',{}).values()))
                for key in ['block_s','mpi_s','postprocess_s']:
                    if key in entry:append(key,entry[key])
                for ifo,detector in entry.get('detectors',{}).items():
                    for key in ['advance_s','psd_check_s','filter_s']:
                        if key in detector:append(ifo+'/'+key,detector[key])
                    for key,value in detector.get('jax',{}).items():append(ifo+'/jax/'+key,value)
                    for key,value in detector.get('preprocessing',{}).items():append(ifo+'/preprocessing/'+key,value)
            rank=dict(startup=startup,blocks=len(blocks),stages={})
            for name,values in metrics.items():
                tail=values[warmup:]
                rank['stages'][name]=dict(samples=len(values),first_s=values[0],
                    total_s=sum(values),median_s=float(np.median(tail)) if tail else None,
                    p95_s=float(np.percentile(tail,95)) if tail else None,
                    max_s=max(values))
            coarse={name:rank['stages'][name]['median_s'] for name in ['conditioning_all_detectors_s','filtering_all_detectors_s','mpi_s','postprocess_s'] if name in rank['stages'] and rank['stages'][name]['median_s'] is not None}
            rank['dominant_serial_stage']=max(coarse,key=coarse.get) if coarse else None
            # The last replay block may be inactive: retain the last recorded
            # layout/counters for each detector, not just the final block.
            rank['buffer_transfers']={}
            rank['batching']={}
            for entry in blocks:
                for ifo,d in entry.get('detectors',{}).items():
                    if 'device_buffer' in d:rank['buffer_transfers'][ifo]=d['device_buffer']
                    if 'batching' in d:rank['batching'][ifo]=d['batching']
            tuning_path=path.with_name(path.stem+'-tuning.json')
            if tuning_path.exists():rank['tuning']=json.loads(tuning_path.read_text())
            row['ranks'][path.stem]=rank
        root=row['ranks'].get('rank0')
        if root and 'block_s' in root['stages'] and run['wall_s'] is not None:
            row['outside_rank0_loop_s']=max(0.,run['wall_s']-root['stages']['block_s']['total_s'])
        row['profile_available']=bool(row['ranks'])
        report['runs'].append(row)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--warmup-blocks',type=int,default=1)
    args=p.parse_args()
    if args.warmup_blocks<0:p.error('warmup-blocks must be nonnegative')
    manifest=json.loads((args.run_dir/'runs.json').read_text())
    report=analyze(manifest,args.warmup_blocks)
    out=args.run_dir/'performance';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    rows=[]
    for run in report['runs']:
        for rank,info in run['ranks'].items():
            for stage,values in info['stages'].items():
                rows.append(dict(mode=run['mode'],repeat=run['repeat'],rank=rank,stage=stage,**values))
    if rows:
        with (out/'stages.csv').open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Median per-detector/per-rank values, not a sum of overlapping ranks.
        selected=[r for r in rows if r['rank']=='rank1' and r['stage'].startswith('H1/') and r['median_s'] is not None]
        names=sorted({r['stage'] for r in selected})
        fig,ax=plt.subplots(figsize=(13,6),constrained_layout=True)
        groups=sorted({(r['mode'],r['repeat']) for r in selected})
        width=.8/max(len(groups),1);x=np.arange(len(names))
        for i,group in enumerate(groups):
            lookup={r['stage']:r['median_s'] for r in selected if (r['mode'],r['repeat'])==group}
            ax.bar(x+i*width,[lookup.get(n,0) for n in names],width,label=f'{group[0]}_{group[1]}')
        ax.set_xticks(x,names,rotation=35,ha='right');ax.set_ylabel('Median seconds per call')
        ax.set_title('H1 rank 1 stages (inclusive totals overlap sub-stages)');ax.legend()
        fig.savefig(out/'stages.png',dpi=140);plt.close(fig)
    for run in report['runs']:
        print(run['mode'],run['repeat'],'wall_s=',run['wall_s'],
              'outside_loop_s=',run.get('outside_rank0_loop_s'),
              'profiles=',run['profile_available'])
        for name,rank in run['ranks'].items():
            print(' ',name,'dominant stage:',rank['dominant_serial_stage'])
            if rank['startup'].get('native_bank_generation_s') is not None:
                print('    native bank generation_s=',rank['startup']['native_bank_generation_s'])
            if rank['startup'].get('jax_waveforms'):
                print('    JAX waveform generation:',json.dumps(rank['startup']['jax_waveforms']))
            if rank['startup'].get('reference_waveforms'):
                reference=rank['startup']['reference_waveforms']
                print('    CPU reference waveform source:',rank['startup'].get('bank_waveform_source'))
                print('    CPU reference generation/JIT:',json.dumps({k:v for k,v in reference.items() if k != 'waveforms'}))
            for stage,values in rank['stages'].items():
                if stage in ['conditioning_all_detectors_s','filtering_all_detectors_s','mpi_s','postprocess_s'] or '/jax/' in stage or '/preprocessing/' in stage:
                    print(f"    {stage}: total={values['total_s']:.6f}s median={values['median_s']}s p95={values['p95_s']}s")
            for ifo,batching in rank['batching'].items():
                print('   ',ifo,'batching:',json.dumps(batching,sort_keys=True))
            if 'tuning' in rank:
                print('    tuning setup_s=',rank['tuning']['setup_s'])
    parity_path=args.run_dir/'comparison'/'summary.json'
    if parity_path.exists():
        parity=json.loads(parity_path.read_text())
        print('Output parity:',parity.get('status'))
        for label,result in parity.get('comparisons',{}).items():
            if result.get('passed'):continue
            print(' ',label,'FAILED')
            for key in ['missing_files','extra_files','missing_triggers','extra_triggers','row_errors']:
                if result.get(key):print('   ',key,':',json.dumps(result[key][:5]))
            for field,value in result.get('fields',{}).items():
                if not value['passed']:print('   ',field,':',json.dumps(value))
            for key,value in result.get('other_products',{}).items():
                if value:print('    other_products/',key,':',json.dumps(value[:5]))
            candidates=result.get('candidates',{})
            if candidates and not candidates.get('passed'):print('    candidates:',json.dumps(candidates))
    if not rows:
        print('No stage profiles: rerun with --profile-stages.')
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
