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


def test_plan_reuses_bank_and_only_changes_backend(tmp_path):
    (tmp_path/'bank.hdf').write_bytes(b'bank')
    (tmp_path/'H1-test.gwf').write_bytes(b'frame')
    base=['--bank-file','bank.hdf','--frame-src','H1:H1-*.gwf','--sample-rate','512',
          '--output-path','output','--run-snr-optimization','--snr-opt-extra-opts','--foo bar']
    path=tmp_path/'args.json';path.write_text(json.dumps(base))
    out=tmp_path/'runs'
    result=subprocess.run([sys.executable,str(ROOT/'examples/live/run_backend_comparison.py'),
        '--example-dir',str(tmp_path),'--args-json',str(path),'--output',str(out),'--plan-only'],
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
        if row['mode'] != 'native':
            assert runner.values(row['command'],'--jax-waveform-source') == ['bank']
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


@pytest.mark.parametrize('mpi_version,openmpi', [('HYDRA build details',False),
                                                ('mpiexec (OpenRTE) 4.1.6',True)])
def test_native_example_launcher_accepts_hydra_and_openmpi(tmp_path,mpi_version,openmpi):
    # Reach the real run.sh launch path using pre-existing fixture names. The
    # recording launcher stops before the example's post-search checks.
    for name in ['template_bank.hdf','injections.hdf','single_significance_fits.hdf',
                 'H1-fit_coeffs.hdf','L1-fit_coeffs.hdf','V1-fit_coeffs.hdf','statHL.hdf']:
        (tmp_path/name).touch()
    (tmp_path/'strain').mkdir()
    launcher=tmp_path/'launcher'
    record=tmp_path/'launch.json'
    launcher.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
if '--version' in sys.argv:
    print(os.environ['TEST_MPI_VERSION'])
    sys.exit(0)
with open(os.environ['TEST_MPI_RECORD'],'w') as handle:
    json.dump(dict(args=sys.argv[1:],token=os.environ.get('TEST_EXPORT_TOKEN')),handle)
sys.exit(37)
''')
    launcher.chmod(0o755)
    (tmp_path/'pycbc_live').write_text('#!/bin/sh\nexit 0\n')
    (tmp_path/'pycbc_live').chmod(0o755)
    env=dict(os.environ,PYCBC_MPIEXEC=str(launcher),TEST_MPI_VERSION=mpi_version,
             TEST_MPI_RECORD=str(record),TEST_EXPORT_TOKEN='inherited',
             PATH=str(tmp_path)+os.pathsep+os.environ['PATH'])
    result=subprocess.run(['bash',str(ROOT/'examples/live/run.sh')],cwd=tmp_path,
                           env=env,capture_output=True,text=True)
    assert result.returncode==37,result.stdout+result.stderr
    actual=json.loads(record.read_text())
    assert actual['token']=='inherited'
    args=actual['args']
    assert args[:2]==['-n','2']
    assert '-x' not in args and '-host' not in args
    assert ('--bind-to' in args)==openmpi
    assert runner.values(args,'--approximant')==['SPAtmplt:mtotal<4','SEOBNRv4_ROM:else']
