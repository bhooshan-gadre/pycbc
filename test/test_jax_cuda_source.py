"""Check CUDA bin-loop source against independent native Cython arithmetic.

Compile the same kernel bodies as host C++ to test indexing and arithmetic
without claiming GPU execution. GPU integration is tested by validate_cpu_gpu.
"""
import ctypes
import os
import shutil
import subprocess
import numpy as np
import pytest


def test_cuda_source_selected_and_repeated_points_match_native(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    from pycbc.vetoes.jax_cuda import SOURCE
    from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
    from pycbc.scheme import CPUScheme
    from pycbc.types import Array
    header='''#include <cmath>
#define __global__
struct float2 {float x,y;};
struct dim {int x;};
dim blockDim={1},blockIdx={0},threadIdx={0};
'''
    wrapper='''
extern "C" void evaluate(const float2* q,const int* rows,const int* edges,
const float* shifts,const float2* snr,const float* norms,const float* counts,
float* pieces,float* result,int n,int ne,int stride,int nt,int points) {
for(int i=0;i<n*(ne-1);i++){blockIdx.x=i;binsum(q,rows,edges,shifts,pieces,n,ne,stride,nt,points);}
for(int i=0;i<n;i++){blockIdx.x=i;finish(pieces,rows,snr,norms,counts,result,n,ne-1,points);}
}
'''
    source=tmp_path/'kernel.cpp';source.write_text(header+SOURCE+wrapper)
    library=tmp_path/'kernel.so'
    subprocess.run([compiler,'-shared','-fPIC','-O2','-ffp-contract=off',str(source),'-o',str(library)],check=True)
    function=ctypes.CDLL(str(library)).evaluate
    function.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*5
    rng=np.random.default_rng(981);ntime=262144;stride=131073
    q=((rng.normal(size=(3,stride))+1j*rng.normal(size=(3,stride)))*.01).astype('complex64')
    bins=np.tile(np.array([30,8192,32768,65536,131000],dtype='int32'),(3,1))
    norms=np.array([.1,.2,.3],dtype='float32');counts=np.full(3,4,dtype='float32')
    rows=np.array([2,0,2],dtype='int32')
    for points in [False,True]:
        shifts=np.array([244113,131001,178913],dtype='float32')
        snr=np.array([.1j,.5+.2j,-.3j],dtype='complex64')
        partial=np.empty((3,4),dtype='float32');actual=np.empty(3,dtype='float32')
        arrays=(q,rows,bins,shifts,snr,norms,counts,partial,actual)
        function(*[a.ctypes.data for a in arrays],3,5,stride,ntime,int(points))
        expected=[]
        with CPUScheme(num_threads=1):
            for c,row in enumerate(rows):
                corr=np.zeros(ntime,dtype='complex64');corr[:stride]=q[row]
                slot=c if points else row
                expected.append(power_chisq_at_points_from_precomputed(
                    Array(corr),snr[slot:slot+1],float(norms[row]),bins[row],[shifts[slot]])[0])
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-5)


@pytest.mark.skipif(os.environ.get('PYCBC_JAX_TEST_PLATFORM') != 'gpu',reason='Set PYCBC_JAX_TEST_PLATFORM=gpu for required GPU integration')
def test_cuda_dlpack_selected_and_candidate_points_on_gpu():
    import jax
    import jax.numpy as jnp
    from pycbc.vetoes.jax_chisq import power_chisq_selected,power_chisq_candidate_points
    from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
    from pycbc.types import Array
    from pycbc.scheme import CPUScheme
    device=jax.devices('gpu')[0]
    rng=np.random.default_rng(78);ntime=65536
    q=((rng.normal(size=(3,ntime))+1j*rng.normal(size=(3,ntime)))*.01).astype('complex64')
    q[:,ntime//2+1:]=0
    bins=np.tile(np.array([100,8000,16000,24000,32000],dtype='int32'),(3,1))
    norms=np.array([.1,.2,.3],dtype='float32');counts=np.full(3,4,dtype='float32')
    shifts=np.array([60113,31001,28913],dtype='int32')
    snr=np.array([.1j,.5+.2j,-.3j],dtype='complex64');rows=np.array([2,0,2],dtype='int32')
    with jax.default_device(device):
        qd,sd,nd,bd,td,cd=[jnp.asarray(a) for a in (q,snr,norms,bins,shifts,counts)]
        for points in [False,True]:
            if points:
                value=power_chisq_candidate_points(qd,nd,bd,cd,rows,shifts,
                    snr*norms[rows],32000)
            else:value=power_chisq_selected(qd,sd,nd,bd,td,cd,rows,32000)
            assert value.devices()=={device}
            actual=np.asarray(value)
            with CPUScheme(num_threads=1):
                expected=[]
                for c,row in enumerate(rows):
                    slot=c if points else row
                    expected.append(power_chisq_at_points_from_precomputed(Array(q[row]),
                        snr[slot:slot+1],float(norms[row]),bins[row],[shifts[slot]])[0])
            np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-5)
