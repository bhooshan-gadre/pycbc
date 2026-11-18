#!/usr/bin/env python
"""Compare native/JAX live trigger products, candidate XMLs and measured latency.

Reads runs.json from run_backend_comparison.py. Exits nonzero for failed jobs,
missing blocks/rows/candidates, unequal results, or incomplete latency records.
Historical GPS lag is retained for diagnosis and never called replay latency.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import h5py

IFOS = {'H1','L1','V1','K1','G1'}


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def collect(directory):
    rows, extras, files = {}, {}, []
    for path in sorted(directory.rglob('*.hdf')):
        relative = str(path.relative_to(directory)); files.append(relative)
        with h5py.File(path,'r') as f:
            for ifo in IFOS & set(f):
                group = f[ifo]
                if 'template_id' not in group:
                    continue
                ids = group['template_id'][:]
                columns = {k:v[:] for k,v in group.items() if isinstance(v,h5py.Dataset)
                           and v.ndim == 1 and len(v) == len(ids) and k != 'loudest'}
                for i,tid in enumerate(ids):
                    key = (relative,ifo,int(tid))
                    if key in rows:
                        raise ValueError(f'Duplicate single-trigger identity: {key}')
                    rows[key] = {k:v[i] for k,v in columns.items()}
            def visit(name, value):
                # PSDs, coincidence/ranking products and stored gate arrays.
                # Compare these separately from per-template trigger rows.
                if not isinstance(value,h5py.Dataset): return
                top = name.split('/')[0]
                if top in IFOS and name.count('/') == 1 and name.split('/')[-1] != 'gates': return
                extras[(relative,name)] = value[()]
            f.visititems(visit)
    return rows,extras,files


def difference(a,b,field,rate):
    a,b = np.asarray(a),np.asarray(b)
    if a.shape != b.shape:
        return dict(passed=False,reason=f'shape {a.shape} versus {b.shape}')
    if a.dtype.kind not in 'iufc' or b.dtype.kind not in 'iufc':
        equal = np.array_equal(a,b)
        return dict(passed=bool(equal),reason=None if equal else 'non-numeric values differ')
    if a.dtype.kind in 'iu' and b.dtype.kind in 'iu':
        equal = np.array_equal(a,b)
        return dict(passed=bool(equal),reason=None if equal else 'integer values differ')
    delta = b.astype(np.complex128 if a.dtype.kind == 'c' else np.float64)-a
    if field == 'coa_phase':
        delta = np.angle(np.exp(1j*delta))
    tolerance = 1.01/rate if field == 'end_time' else .001 if field == 'coa_phase' else .001+.0001*np.abs(a)
    finite = np.isfinite(a).all() and np.isfinite(b).all()
    return dict(passed=bool(finite and np.all(np.abs(delta)<=tolerance)),
                max_absolute_error=float(np.max(np.abs(delta))) if delta.size and finite else None,
                reason=None if finite else 'nonfinite data')


def candidates(directory):
    """Parse scientific LIGO-LW tables, ignoring generated process/event IDs."""
    result = {}
    for path in sorted(directory.rglob('*.xml*')):
        if not (path.name.endswith('.xml') or path.name.endswith('.xml.gz')): continue
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path,'rb') as handle:
            root = ET.parse(handle).getroot()
        tables = {}
        for table in root.iter('Table'):
            name = table.attrib.get('Name','')
            if name not in ['sngl_inspiral:table','coinc_inspiral:table','coinc_event:table']: continue
            columns = [c.attrib['Name'].split(':')[-1] for c in table.findall('Column')]
            streams = table.findall('Stream'); rows = []
            for stream in streams:
                for line in csv.reader((stream.text or '').splitlines(), delimiter=stream.attrib.get('Delimiter',','), skipinitialspace=True):
                    if not line or not any(v.strip() for v in line): continue
                    if len(line) == len(columns)+1 and not line[-1].strip(): line = line[:-1]
                    if len(line) != len(columns): raise ValueError(f'Malformed XML table {path}/{name}')
                    row = {}
                    for key,value in zip(columns,line):
                        if key.endswith('_id') or key in ['creator_db','search','channel']: continue
                        value = value.strip()
                        try: row[key] = float(value)
                        except ValueError: row[key] = value
                    rows.append(row)
            rows.sort(key=lambda r:tuple(str(r.get(k,'')) for k in ['ifo','mass1','mass2','spin1z','spin2z','end_time','end_time_ns']))
            tables[name] = rows
        result[str(path.relative_to(directory))] = tables
    return result


def compare_xml(a,b,rate):
    report = dict(missing=sorted(a.keys()-b.keys()),extra=sorted(b.keys()-a.keys()),errors=[])
    for name in sorted(a.keys() & b.keys()):
        if a[name].keys() != b[name].keys():
            report['errors'].append(name+': table set differs'); continue
        for table,rows in a[name].items():
            other = b[name][table]
            if len(rows) != len(other):
                report['errors'].append(name+'/'+table+': row count differs'); continue
            for i,(r,s) in enumerate(zip(rows,other)):
                # Combine integer seconds/nanoseconds before applying a sample
                # tolerance. Comparing nanoseconds independently is misleading.
                r,s = dict(r),dict(s)
                for row in (r,s):
                    if 'end_time' in row and 'end_time_ns' in row:
                        row['end_time'] += row.pop('end_time_ns')*1e-9
                if r.keys() != s.keys():
                    report['errors'].append(f'{name}/{table}/{i}: columns differ'); continue
                for field in r:
                    if not difference(r[field],s[field],field,rate)['passed']:
                        report['errors'].append(f'{name}/{table}/{i}/{field}')
    report['passed'] = not any(report[k] for k in ['missing','extra','errors'])
    report['reference_candidates'],report['actual_candidates'] = len(a),len(b)
    return report


def latency(run, warmup):
    directory = Path(run['directory'])
    path = directory/'latency.jsonl'
    samples = []
    if path.exists():
        samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    source = 'rank-0 monotonic JSONL'
    if not samples:
        source = 'rounded legacy rank-0 Took log (lower precision)'
        # Rank must be exactly 0; never combine worker and coordinator timings.
        pattern = re.compile(r'^\S+\s+\S+\s+0\s+Took ([\d.]+), duty factor of ([\d.]+), lag ([\d.\-]+) s, (\d+) live detectors')
        for line in (directory/'search.log').read_text().splitlines():
            m = pattern.search(line)
            if m:
                samples.append(dict(elapsed_s=float(m[1]),live_detectors=['unknown']*int(m[4])))
    usable = [s for s in samples if s['live_detectors']]
    warm = usable[warmup:]
    seconds = [float(s['elapsed_s']) for s in warm]
    if any(not np.isfinite(s['elapsed_s']) or s['elapsed_s'] < 0 for s in samples):
        raise ValueError('Invalid latency values in '+str(directory))
    return dict(source=source,blocks_total=len(samples),active_blocks=len(usable),
        excluded_warmup_blocks=min(warmup,len(usable)),warm_blocks=len(seconds),
        median_s=float(np.median(seconds)) if seconds else None,
        p95_s=float(np.percentile(seconds,95)) if seconds else None,
        max_s=max(seconds) if seconds else None,
        first_active_block_s=float(usable[0]['elapsed_s']) if usable else None,
        wall_s=run['wall_s']),samples


def compare(ref,other,rate,directory,label):
    r,re,rf = collect(ref); a,ae,af = collect(other)
    keys = sorted(r.keys() & a.keys())
    report = dict(missing_files=sorted(set(rf)-set(af)),extra_files=sorted(set(af)-set(rf)),
        missing_triggers=[list(k) for k in sorted(r.keys()-a.keys())],
        extra_triggers=[list(k) for k in sorted(a.keys()-r.keys())],
        matched=len(keys),fields={},row_errors=[],other_products={})
    csv_rows = []
    for k in keys:
        row = dict(block=k[0],detector=k[1],template_id=k[2])
        if r[k].keys() != a[k].keys(): report['row_errors'].append(str(k)+': columns differ')
        for field in r[k].keys() & a[k].keys():
            m = difference(r[k][field],a[k][field],field,rate)
            report['fields'].setdefault(field,dict(passed=True,max_absolute_error=0.,failed_rows=0))
            agg = report['fields'][field]
            agg['passed'] &= m['passed']; agg['failed_rows'] += int(not m['passed'])
            agg['max_absolute_error'] = max(agg['max_absolute_error'],m.get('max_absolute_error') or 0.)
            if field in ['snr','chisq','sg_chisq','end_time','coa_phase']:
                rv,av = float(r[k][field]),float(a[k][field]); dv=av-rv
                if field == 'coa_phase': dv=float(np.angle(np.exp(1j*dv)))
                row.update({field+'_native':rv,field+'_actual':av,field+'_difference':dv})
        csv_rows.append(row)
    report['other_products']['missing'] = [list(k) for k in sorted(re.keys()-ae.keys())]
    report['other_products']['extra'] = [list(k) for k in sorted(ae.keys()-re.keys())]
    report['other_products']['errors'] = [dict(file=k[0],dataset=k[1],**difference(re[k],ae[k],k[1].split('/')[-1],rate))
        for k in sorted(re.keys() & ae.keys()) if not difference(re[k],ae[k],k[1].split('/')[-1],rate)['passed']]
    report['candidates'] = compare_xml(candidates(ref),candidates(other),rate)
    report['passed'] = bool(r) and bool(a) and not any(report[k] for k in
        ['missing_files','extra_files','missing_triggers','extra_triggers','row_errors']) and all(
        m['passed'] for m in report['fields'].values()) and not any(report['other_products'].values()) and report['candidates']['passed']
    if csv_rows:
        with (directory/(label+'_triggers.csv')).open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=sorted(set().union(*(x.keys() for x in csv_rows))),lineterminator='\n')
            writer.writeheader();writer.writerows(csv_rows)
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(5,2,figsize=(14,14),constrained_layout=True)
        for i,field in enumerate(['snr','chisq','sg_chisq','end_time','coa_phase']):
            selected=[x for x in csv_rows if field+'_native' in x]
            axes[i,0].plot([x[field+'_native'] for x in selected],'.',label='native')
            axes[i,0].plot([x[field+'_actual'] for x in selected],'.',alpha=.6,label=label)
            axes[i,1].plot([x[field+'_difference'] for x in selected],'.'); axes[i,0].set_ylabel(field)
            axes[i,1].set_ylabel('Difference'); axes[i,1].set_xlabel('Block / detector / template row')
        axes[0,0].legend();fig.savefig(directory/(label+'_results.png'),dpi=140);plt.close(fig)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--warmup-blocks',type=int,default=1,
                   help='Exclude this many active blocks per run from warm latency only')
    args=p.parse_args()
    if args.warmup_blocks < 0: p.error('warmup-blocks must be nonnegative')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=dict(status='ERROR',failures=[],runs=[],comparisons={})
    out=args.run_dir.resolve()/'comparison';out.mkdir(exist_ok=True)
    try:
        manifest=json.loads((args.run_dir/'runs.json').read_text())
        if manifest.get('plan_only'): raise ValueError('Plan-only manifest has no executed searches')
        if manifest.get('input_integrity') is not True: report['failures'].append('Input integrity not verified')
        runs=manifest['runs']; native=[r for r in runs if r['mode']=='native' and r['returncode']==0]
        if not native: raise ValueError('No successful native run')
        options=manifest['native_options']; rate=float(options[options.index('--sample-rate')+1])
        reference=Path(native[0]['directory'])/'output'
        reference_latency,reference_samples = latency(native[0],args.warmup_blocks)
        latency_series = []
        for run in runs:
            label=f'{run["mode"]}_{run["repeat"]}'
            if run['returncode'] != 0:
                report['failures'].append(label+': search failed; see search.log');continue
            stats,samples=latency(run,args.warmup_blocks)
            if not stats['warm_blocks']: report['failures'].append(label+': no warm active latency samples')
            stats['coverage_verified'] = False
            if stats['source'] == reference_latency['source'] == 'rank-0 monotonic JSONL':
                stats['coverage_verified'] = True
                ref_blocks = [(s['gps_start'],s['gps_end'],s['live_detectors']) for s in reference_samples]
                actual_blocks = [(s['gps_start'],s['gps_end'],s['live_detectors']) for s in samples]
                if ref_blocks != actual_blocks:
                    report['failures'].append(label+': analysis-block/detector coverage differs')
            if stats['source'] == 'rank-0 monotonic JSONL':
                log_path=Path(run['directory'])/'search.log'
                if log_path.exists():
                    count=sum(bool(re.search(r'^\S+\s+\S+\s+0\s+Took ',line)) for line in log_path.read_text().splitlines())
                    if count != len(samples):
                        report['failures'].append(label+': JSONL/log block counts differ')
            latency_series.append((label,samples))
            report['runs'].append(dict(label=label,mode=run['mode'],repeat=run['repeat'],**stats))
            actual=Path(run['directory'])/'output'
            result=compare(reference,actual,rate,out,label)
            report['comparisons'][label]=result
            if not result['passed']: report['failures'].append(label+': output parity failed')
        groups={m:[r for r in report['runs'] if r['mode']==m] for m in {r['mode'] for r in report['runs']}}
        baseline=float(np.median([r['wall_s'] for r in groups.get('native',[])]))
        native_blocks=[r['median_s'] for r in groups.get('native',[]) if r['median_s'] is not None]
        native_block_median=float(np.median(native_blocks)) if native_blocks else None
        report['timings']={}
        for mode,rows in groups.items():
            wall=float(np.median([r['wall_s'] for r in rows]))
            blocks=[r['median_s'] for r in rows if r['median_s'] is not None]
            block_median=float(np.median(blocks)) if blocks else None
            report['timings'][mode]=dict(median_wall_s=wall,wall_speedup_vs_native=baseline/wall,
                median_of_run_block_medians_s=block_median,
                block_median_speedup_vs_native=(native_block_median/block_median
                    if native_block_median is not None and block_median else None))
        names=[r['label'] for r in report['runs']];x=np.arange(len(names))
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        axes[0].bar(x,[r['wall_s'] for r in report['runs']]);axes[0].set_ylabel('Complete process wall time (s)')
        axes[1].bar(x,[r['median_s'] or 0 for r in report['runs']],label='median')
        axes[1].plot(x,[r['p95_s'] or 0 for r in report['runs']],'o',label='p95');axes[1].legend()
        axes[1].set_ylabel('Active-block processing latency (s), warm')
        for ax in axes: ax.set_xticks(x,names,rotation=30)
        fig.savefig(out/'latency.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
        for label,samples in latency_series:
            active=[s for s in samples if s['live_detectors']]
            ax.plot([s['elapsed_s'] for s in active],'.-',label=label)
        ax.set_xlabel('Active analysis block (includes initial block)')
        ax.set_ylabel('Processing latency (s)');ax.legend();ax.grid(alpha=.2)
        fig.savefig(out/'latency_blocks.png',dpi=140);plt.close(fig)
        with (out/'latency.csv').open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(report['runs'][0]),lineterminator='\n');writer.writeheader();writer.writerows(report['runs'])
        report['status']='FAIL' if report['failures'] else 'PASS'
    except Exception as exc:
        report['error']=str(exc)
    dump(out/'summary.json',report)
    print(json.dumps(dict(status=report['status'],failures=report['failures'],error=report.get('error'))))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
