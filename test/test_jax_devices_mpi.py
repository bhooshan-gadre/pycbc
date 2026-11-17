"""Optional real MPI collective tests using simulated visibility, no GPUs.

Run explicitly with PYCBC_TEST_MPI=1. These validate communicator behavior,
not CUDA execution; use the full-live benchmark for actual GPU acceptance.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('PYCBC_TEST_MPI') != '1',
                                reason='set PYCBC_TEST_MPI=1 for real MPI tests')


@pytest.mark.parametrize('mask,sharing,should_fail', [
    ('GPU-testA,GPU-testB',False,False), ('GPU-testA',False,True),
    ('GPU-testA',True,False)])
def test_collective_binding(tmp_path,mask,sharing,should_fail):
    pytest.importorskip('mpi4py')
    if not shutil.which('mpiexec'):
        pytest.skip('MPI launcher unavailable')
    script = tmp_path/'binding.py'
    script.write_text('''import json
from mpi4py import MPI
from pycbc.filter.jax_devices import bind_mpi_gpu
comm = MPI.COMM_WORLD
try:
    value = bind_mpi_gpu(comm, comm.rank > 0, allow_sharing=SHARING)
    result = dict(value=value, error=None)
except RuntimeError as exc:
    result = dict(value=None, error=str(exc))
rows = comm.gather(result, root=0)
if comm.rank == 0:
    print('BINDING_JSON=' + json.dumps(rows), flush=True)
'''.replace('SHARING',repr(sharing)))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=mask)
    result = subprocess.run(['mpiexec','-n','3',sys.executable,str(script)],
        env=env,text=True,capture_output=True,timeout=40)
    assert result.returncode == 0, result.stdout+result.stderr
    rows = json.loads(next(line[len('BINDING_JSON='):] for line in
                           result.stdout.splitlines() if line.startswith('BINDING_JSON=')))
    if should_fail:
        assert all('Not enough distinct' in r['error'] for r in rows)
    else:
        assert all(r['error'] is None for r in rows)
        assert rows[0]['value'] is None
        assert all(r['value']['device_index'] == 0 for r in rows[1:])
        if not sharing:
            assert len({r['value']['selected_gpu'] for r in rows[1:]}) == 2
