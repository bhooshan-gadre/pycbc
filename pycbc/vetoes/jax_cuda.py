"""Native-compatible point chi-squared CUDA kernels for JAX device arrays.

CuPy provides NVRTC compilation and DLPack stream interop, without copying
spectra through host memory. Each CUDA thread performs one native bin loop;
using an XLA while loop here would launch kernels once per frequency sample.
"""
from functools import lru_cache
import numpy as np

SOURCE = r'''
extern "C" __global__ void binsum(
    const float2* q, const int* rows, const int* edges, const float* shifts,
    float* pieces, int candidates, int nedge, int stride, int ntime, int points) {
    int work = blockDim.x*blockIdx.x + threadIdx.x;
    int c=work/(nedge-1), b=work%(nedge-1);
    if(c>=candidates) return;
    int row=rows[c], shiftrow=points?c:row;
    int start=edges[row*nedge+b], end=edges[row*nedge+b+1];
    double phase=2.0*3.141592653*(double)shifts[shiftrow];
    float pr=(float)cos(phase*start/ntime), pi=(float)sin(phase*start/ntime);
    float vsr=(float)cos(phase/ntime), vsi=(float)sin(phase/ntime);
    float real=0.0f,imag=0.0f;
    for(int k=start;k<end;k++) {
        float2 value=q[(long long)row*stride+k];
        float vr=value.x, vi=value.y, t1=pr, t2=pi;
        float vs=vr+vi, va=vi-vr;
        float k1=vr*(t1+t2), k2=t1*va, k3=t2*vs;
        real+=k1-k3; imag+=k1+k2;
        pr=t1*vsr-t2*vsi; pi=t1*vsi+t2*vsr;
    }
    pieces[work]=real*real+imag*imag;
}
extern "C" __global__ void finish(
    const float* pieces,const int* rows,const float2* snr,const float* norms,
    const float* counts,float* result,int candidates,int nbin,int points) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=candidates)return;
    int row=rows[c];
    float sum=0.0f;
    for(int b=0;b<nbin;b++)sum+=pieces[c*nbin+b];
    float2 peak=snr[points?c:row];
    float power=sum*counts[row]-(peak.x*peak.x+peak.y*peak.y);
    result[c]=power*(norms[row]*norms[row]);
}
'''


@lru_cache(None)
def _kernels(device_id):
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError('Native-compatible GPU vetoes require CuPy/NVRTC; install cupy-cuda12x for CUDA 12 (or the matching CUDA package). No CPU fallback is used.') from exc
    # Prevent fused operations from changing the native float32 recurrence.
    with cp.cuda.Device(device_id):
        module=cp.RawModule(code=SOURCE,options=('--fmad=false',),name_expressions=('binsum','finish'))
        return cp,module.get_function('binsum'),module.get_function('finish')


def power_chisq_cuda(q,snr,norms,bins,shifts,counts,rows,ntime,points=False):
    """Raw chi-squared; points=True supplies candidate-indexed peaks/shifts."""
    import jax
    device=next(iter(q.devices()))
    cp,bin_kernel,finish=_kernels(device.local_hardware_id)
    with cp.cuda.Device(device.local_hardware_id):
        def view(a):
            if hasattr(a,'devices') and a.devices() != {device}:
                a=jax.device_put(a,device)
            return cp.from_dlpack(a) if hasattr(a,'__dlpack__') and not isinstance(a,np.ndarray) else cp.asarray(a)
        # DLPack honors producer/consumer stream dependencies in both directions.
        types=(cp.complex64,cp.complex64,cp.float32,cp.int32,cp.float32,cp.float32)
        arrays=[cp.ascontiguousarray(view(a),dtype=t) for a,t in zip((q,snr,norms,bins,shifts,counts),types)]
        qd,sd,nd,bd,td,cd=arrays
        rd=cp.ascontiguousarray(view(rows),dtype=cp.int32)
        count=int(len(rows));nedge=int(bins.shape[1]);nbin=nedge-1
        pieces=cp.empty((count,nbin),dtype=cp.float32)
        result=cp.empty(count,dtype=cp.float32)
        if count:
            i32=np.int32
            bin_kernel(((count*nbin+127)//128,),(128,),
                (qd,rd,bd,td,pieces,i32(count),i32(nedge),i32(q.shape[1]),i32(ntime),i32(points)))
            finish(((count+127)//128,),(128,),
                (pieces,rd,sd,nd,cd,result,i32(count),i32(nbin),i32(points)))
        return jax.dlpack.from_dlpack(result)
