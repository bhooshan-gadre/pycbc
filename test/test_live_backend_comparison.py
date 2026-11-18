"""Fixed native inputs, exact bank waveforms, and latency/result acceptance."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'examples/live'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


runner=load('run_backend_comparison')
comparison=load('compare_backend_runs')


def test_native_options_preserve_models_and_search_settings():
    args=runner.options()
    assert runner.values(args,'--approximant') == ['SPAtmplt:mtotal<4','SEOBNRv4_ROM:else']
    assert runner.values(args,'--sample-rate') == ['2048']
    assert runner.values(args,'--analysis-chunk') == ['8']
    assert runner.values(args,'--max-triggers-in-batch') == ['30']
    assert '--enable-background-estimation' in args
    assert runner.values(args,'--frame-src') == ['H1:strain/H1/*','L1:strain/L1/*','V1:strain/V1/*']


@pytest.mark.parametrize('phenom',[False,True])
def test_plan_reuses_bank_and_only_changes_backend(tmp_path,phenom):
    (tmp_path/'bank.hdf').write_bytes(b'bank')
    (tmp_path/'H1-test.gwf').write_bytes(b'frame')
    base=['--bank-file','bank.hdf','--approximant','SPAtmplt:mtotal<4','SEOBNRv4_ROM:else',
          '--frame-src','H1:H1-*.gwf','--sample-rate','512',
          '--output-path','output','--run-snr-optimization','--snr-opt-extra-opts','--foo bar']
    path=tmp_path/'args.json';path.write_text(json.dumps(base))
    out=tmp_path/'runs'
    result=subprocess.run([sys.executable,str(ROOT/'examples/live/run_backend_comparison.py'),
        '--example-dir',str(tmp_path),'--args-json',str(path),'--output',str(out),'--plan-only',
        '--profile-stages','--jax-preprocessing','condition','--coordinator-gpu',
        '--allow-gpu-sharing',*(['--waveform-model','IMRPhenomD','--reference-waveforms','ripple'] if phenom else [])],
        text=True,capture_output=True)
    assert result.returncode == 0,result.stdout+result.stderr
    m=json.loads((out/'runs.json').read_text())
    assert m['input_integrity'] is True and m['plan_only'] is True
    assert len(m['inputs']) == 2
    assert len({r['directory'] for r in m['runs']}) == 3
    for row in m['runs']:
        assert '--run-snr-optimization' not in row['command']
        assert '--foo bar' not in row['command']
        assert runner.values(row['command'],'--bank-file') == ['bank.hdf']
        assert runner.values(row['command'],'--approximant') == (['IMRPhenomD'] if phenom else ['SPAtmplt:mtotal<4','SEOBNRv4_ROM:else'])
        assert runner.values(row['command'],'--bank-waveform-source') == ['ripple' if phenom else 'lal']
        assert '--stage-profile-dir' in row['command']
        assert ('--jax-coordinator-preprocessing' in row['command']) == (row['mode'] == 'jax_gpu')
        assert ('--jax-allow-gpu-sharing' in row['command']) == (row['mode'] == 'jax_gpu')
        assert ('--jax-autotune' in row['command']) == (row['mode'] == 'jax_gpu')
        if row['mode'] != 'native':
            assert runner.values(row['command'],'--jax-waveform-source') == ['ripple' if phenom else 'bank']
            assert runner.values(row['command'],'--jax-preprocessing') == ['condition']
    # Preparation/comparison must never delete the native example output.
    (tmp_path/'output').mkdir();(tmp_path/'output'/'keep.txt').write_text('keep')
    assert (tmp_path/'output'/'keep.txt').read_text()=='keep'


def write_output(path,chisq=1.):
    path.mkdir(parents=True,exist_ok=True)
    with h5py.File(path/'block.hdf','w') as f:
        g=f.create_group('H1')
        for name,value in dict(template_id=0,snr=20.,chisq=chisq,chisq_dof=14,
                               sg_chisq=1.,coa_phase=.2,end_time=100.).items():
            g[name]=np.array([value])


def test_compare_retains_missing_blocks_veto_errors_and_large_integer_ids(tmp_path):
    native,actual=tmp_path/'native',tmp_path/'actual'
    write_output(native);write_output(actual)
    assert comparison.compare(native,actual,2048,tmp_path,'good')['passed']
    write_output(actual,1.1)
    report=comparison.compare(native,actual,2048,tmp_path,'bad')
    assert not report['passed'] and report['fields']['chisq']['failed_rows']==1
    (actual/'block.hdf').unlink()
    assert comparison.compare(native,actual,2048,tmp_path,'missing')['missing_files']==['block.hdf']
    assert not comparison.difference(np.uint64(2**63),np.uint64(2**63+1),'template_hash',2048)['passed']


def test_latency_does_not_use_historical_lag_or_worker_times(tmp_path):
    run=dict(directory=str(tmp_path),wall_s=10)
    (tmp_path/'search.log').write_text(
        '2026-11-18T01:00:00 node 1 Took 99.99, duty factor of 1, lag 200000000.00 s, 2 live detectors\n'
        '2026-11-18T01:00:00 node 0 Took 0.25, duty factor of 0.03, lag 200000000.00 s, 2 live detectors\n')
    stats,_=comparison.latency(run,0)
    assert stats['median_s']==.25 and stats['blocks_total']==1
    samples=[dict(elapsed_s=x,live_detectors=['H1']) for x in [1.,.123456,.223456]]
    (tmp_path/'latency.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    stats,_=comparison.latency(run,1)
    assert stats['source']=='rank-0 monotonic JSONL'
    assert stats['first_active_block_s']==1.
    assert stats['median_s']==pytest.approx(.173456)


def test_candidate_xml_ignores_generated_ids_but_checks_snr(tmp_path):
    for name,event_id,snr in [('native',1,20),('actual',2,20)]:
        d=tmp_path/name;d.mkdir()
        (d/'candidate.xml').write_text(f'''<LIGO_LW><Table Name="sngl_inspiral:table">
<Column Name="sngl_inspiral:event_id"/><Column Name="sngl_inspiral:ifo"/>
<Column Name="sngl_inspiral:snr"/><Stream Delimiter=",">
  {event_id},H1,{snr},
  </Stream></Table></LIGO_LW>''')
    a=comparison.candidates(tmp_path/'native');b=comparison.candidates(tmp_path/'actual')
    assert comparison.compare_xml(a,b,2048)['passed']
    b['candidate.xml']['sngl_inspiral:table'][0]['snr']=21
    assert not comparison.compare_xml(a,b,2048)['passed']


def test_explicit_native_bank_source_bypasses_waveform_regeneration():
    pytest.importorskip('jax')
    from types import SimpleNamespace
    from pycbc.filter.jax_matchedfilter import JaxLiveBatchMatchedFilter
    engine=object.__new__(JaxLiveBatchMatchedFilter);engine.waveform_source='bank'
    class Template:
        approximant='SEOBNRv4_ROM'
        id=7
        def __array__(self,dtype=None,copy=None): return np.ones(3,dtype=dtype)
    value,report=engine._generate_and_compare(Template())
    assert value is None and report['implementation']=='native-bank'
    assert not report['use_generated'] and report['match_template']==1.


def test_random_candidate_directories_match_but_products_still_checked(tmp_path):
    native,actual=tmp_path/'native',tmp_path/'actual'
    folders=[]
    for root,suffix in [(native,'kwvk'),(actual,'rvnv')]:
        write_output(root)
        folder=root/'2020_05_06'/f'candidate_1272790260.107_{suffix}'
        folder.mkdir(parents=True);folders.append(folder)
        with h5py.File(folder/'coinc-1272790260.107_H1_psd.hdf','w') as f:
            f['psd']=np.ones(16)
        (folder/'candidate.xml').write_text('''<LIGO_LW><Table Name="sngl_inspiral:table">
<Column Name="sngl_inspiral:ifo"/><Column Name="sngl_inspiral:snr"/>
<Stream Delimiter=",">H1,20,</Stream></Table></LIGO_LW>''')
    assert comparison.compare(native,actual,2048,tmp_path,'suffix')['passed']
    with h5py.File(folders[1]/'coinc-1272790260.107_H1_psd.hdf','r+') as f:
        f['psd'][0]=2
    result=comparison.compare(native,actual,2048,tmp_path,'changed')
    assert not result['passed'] and result['other_products']['errors']
    xml=comparison.candidates(actual)
    key=next(iter(xml));xml[key]['sngl_inspiral:table'][0]['snr']=21
    assert not comparison.compare_xml(comparison.candidates(native),xml,2048)['passed']
    import shutil
    shutil.copytree(folders[1],folders[1].with_name('candidate_1272790260.107_abcd'))
    with pytest.raises(ValueError,match='Ambiguous candidate product'):
        comparison.collect(actual)
    with pytest.raises(ValueError,match='Ambiguous candidate XML'):
        comparison.candidates(actual)


def test_structured_statistics_use_field_tolerances_and_exact_counts():
    dtype=[('snr','f4'),('coa_phase','f4'),('end_time','f8'),('template_id','u8')]
    a=np.array([(10.,.5,1272790260.,2**63)],dtype=dtype)
    b=a.copy();b['snr']+=1e-5;b['coa_phase']+=1e-5
    assert comparison.difference(a,b,'stat',2048)['passed']
    b['template_id']+=1
    assert not comparison.difference(a,b,'stat',2048)['passed']
    b=a.copy();b['snr']+=1
    assert not comparison.difference(a,b,'stat',2048)['passed']
    result=comparison.difference(100,101,'count',2048)
    assert not result['passed']
    assert result['native_values']==[100] and result['actual_values']==[101]
