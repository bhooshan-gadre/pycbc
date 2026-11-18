"""Device-resident preprocessing for JAX live filtering.

LAL frame I/O, state/DQ decisions, glitch clustering, and the host strain view
used by coincidence/followup remain native. The device mirror receives only
changed samples after its initial upload. Do not use global processing schemes
or change native FFT normalization to enable this backend.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass
class SpectrumMetadata:
    length: int
    delta_f: float
    epoch: object
    psd: object

    def __len__(self):
        return self.length


@lru_cache(None)
def whitening_kernel(length, trim, rate):
    import jax
    import jax.numpy as jnp
    @jax.jit
    def run(x, psd, window):
        spectrum = jnp.fft.rfft(x) * jnp.float32(1/rate)
        spectrum = spectrum / psd
        if trim:
            white = jnp.fft.irfft(spectrum, n=length) * jnp.float32(rate)
            white = white[trim:length-trim] * window
            spectrum = jnp.fft.rfft(white) * jnp.float32(1/rate)
        return spectrum.astype(jnp.complex64)
    return run


@lru_cache(None)
def ring_update(length, step, width):
    import jax
    import jax.numpy as jnp
    @jax.jit(donate_argnums=(0,))
    def run(buffer, head, tail):
        head = (head + step) % length
        index = (head + length-width + jnp.arange(width)) % length
        return buffer.at[index].set(tail), head
    return run


@lru_cache(None)
def ring_slice(length, count):
    import jax
    import jax.numpy as jnp
    @jax.jit
    def run(buffer, head):
        return buffer[(head + length-count + jnp.arange(count)) % length]
    return run


@lru_cache(None)
def inverse_kernel(length, max_filter_len, kmin, hann):
    import jax
    import jax.numpy as jnp
    n = (length-1)*2
    half = max_filter_len//2
    if max_filter_len <= 0 or 2*half > n:
        raise ValueError('invalid inverse-PSD filter length')
    window = np.hanning(max_filter_len)
    @jax.jit
    def run(psd):
        inv = jnp.zeros(length, dtype=jnp.complex64)
        inv = inv.at[kmin:n//2].set(jnp.sqrt(1/psd[kmin:n//2]))
        time = jnp.fft.irfft(inv, n=n)
        if hann:
            time = time.at[:half].multiply(jnp.asarray(window[-half:], dtype=time.dtype))
            time = time.at[n-half:].multiply(jnp.asarray(window[:half], dtype=time.dtype))
        time = time.at[half:n-half].set(0)
        freq = jnp.fft.rfft(time)
        return 1/jnp.abs(freq*jnp.conj(freq))
    return run


@lru_cache(None)
def welch_kernel(seg_len, stride, segments, rate, dtype):
    import jax
    import jax.numpy as jnp
    from pycbc.psd.estimate import median_bias
    window = np.hanning(seg_len).astype(dtype)
    scale = np.float32(2*rate/(window*window).sum())
    bias = np.float32(median_bias(segments))
    @jax.jit
    def run(x):
        indices = jnp.arange(segments)[:,None]*stride+jnp.arange(seg_len)[None,:]
        frames = x[indices]*jnp.asarray(window)
        freq = jnp.fft.rfft(frames) * jnp.float32(1/rate)
        power = jnp.abs(freq*jnp.conj(freq))
        power = power.at[:,0].multiply(.5).at[:,-1].multiply(.5)
        return (jnp.median(power, axis=0)/bias)*scale
    return run


@lru_cache(None)
def condition_kernel(high_order, down_order, factor, scale):
    import jax
    import jax.numpy as jnp
    @jax.jit
    def run(raw, high_response, down_response):
        def zero_phase(x, response, order):
            y = jnp.fft.irfft(jnp.fft.rfft(x)*response, n=x.shape[0])
            y = y.at[:order*2].set(0)
            # Preserve the native expression -len(coeff)//2 (odd lengths).
            return jnp.roll(y, -(2*order+1)//2)
        x = zero_phase(raw, high_response, high_order)
        x = (x*scale).astype(jnp.float32)
        if factor != 1:
            x = zero_phase(x, down_response, down_order)[::factor]
        return x
    return run


@lru_cache(None)
def glitch_kernel(count, padded_length, rate, low):
    import jax
    import jax.numpy as jnp
    start = int(padded_length/2-count/2)
    kmin = int(low/(rate/padded_length))
    @jax.jit
    def run(value,psd):
        padded = jnp.pad(value,(start,padded_length-count-start))
        freq = jnp.fft.rfft(padded)*jnp.float32(1/rate)
        factor = jnp.where(jnp.arange(len(psd)) >= kmin,
                           (psd*jnp.float32(rate/2-low))**(-.5),0)
        time = jnp.fft.irfft(freq*factor,n=padded_length)*jnp.float32(rate)
        return jnp.abs(time[start:start+count])
    return run


class JaxLiveConditioner:
    """Per-detector GPU/CPU JAX backend; no hidden CPU device fallback."""
    def __init__(self, reader, device, mode='whiten'):
        if mode not in ('whiten','condition'):
            raise ValueError('mode must be whiten or condition')
        self.reader, self.device, self.mode = reader, device, mode
        self.buffer = None
        self.head = 0
        self.epoch = None
        self.source_psd = None
        self.spectra, self.weights, self.windows, self.responses = {}, {}, {}, {}
        self.metrics = dict(uploaded_bytes=0, full_buffer_uploads=0, incremental_uploads=0)

    def put(self, x):
        import jax
        return jax.device_put(np.asarray(x), self.device)

    def sync(self):
        """Mirror the post-gating host ring, updating only changed samples."""
        r = self.reader
        epoch = float(r.strain.start_time)
        length = len(r.strain)
        if self.buffer is not None and self.epoch == epoch:
            return
        step = 0 if self.epoch is None else round((epoch-self.epoch)*r.sample_rate)
        if self.buffer is None or step <= 0 or step >= length or len(self.buffer) != length:
            self.buffer = self.put(np.asarray(r.strain, dtype=np.float32))
            self.head = np.int32(0)
            self.metrics['uploaded_bytes'] += length*4
            self.metrics['full_buffer_uploads'] += 1
        else:
            width = min(length, step + r.corruption)
            for center,zero,pad in r.gate_params:
                first = max(0, int((center-zero-pad-epoch)*r.sample_rate)-1)
                width = max(width, length-first)
            tail = self.put(np.asarray(r.strain)[-width:].astype(np.float32,copy=False))
            self.buffer,self.head = ring_update(length,step,width)(self.buffer,self.head,tail)
            self.metrics['uploaded_bytes'] += width*4
            self.metrics['incremental_uploads'] += 1
        self.epoch = epoch
        self.spectra = {}

    def inverse(self, psd, max_filter_len, low_frequency_cutoff, hann=False):
        import jax
        from pycbc.types import FrequencySeries
        kmin = int(low_frequency_cutoff/psd.delta_f) if low_frequency_cutoff else 1
        with jax.default_device(self.device):
            value = inverse_kernel(len(psd),int(max_filter_len),kmin,hann)(self.put(np.asarray(psd,dtype=np.float32)))
        host = FrequencySeries(np.asarray(value),delta_f=psd.delta_f,epoch=psd.epoch)
        return host,value

    def overwhitened(self, delta_f):
        import jax
        from pycbc import psd as psd_module
        from pycbc.strain import gate_data
        from pycbc.types import TimeSeries
        r = self.reader
        with jax.default_device(self.device):
            self.sync()
            if self.source_psd is not r.psd:
                self.source_psd = r.psd
                self.spectra = {}
            if delta_f in self.spectra:
                return self.spectra[delta_f]
            count = int(1/delta_f)*r.sample_rate + 2*r.reduced_pad
            if count > len(r.strain):
                raise ValueError('strain buffer too short for GPU whitening')
            padded_df = r.sample_rate/count
            if delta_f not in r.psds:
                p = psd_module.interpolate(r.psd,padded_df)
                short = psd_module.interpolate(r.psd,delta_f)
                if self.mode == 'condition':
                    padded,weight = self.inverse(p,int(r.sample_rate*r.psd_inverse_length),r.low_frequency_cutoff)
                    short,_ = self.inverse(short,int(r.sample_rate*r.psd_inverse_length),r.low_frequency_cutoff)
                else:
                    padded = psd_module.inverse_spectrum_truncation(p,int(r.sample_rate*r.psd_inverse_length),low_frequency_cutoff=r.low_frequency_cutoff)
                    short = psd_module.inverse_spectrum_truncation(short,int(r.sample_rate*r.psd_inverse_length),low_frequency_cutoff=r.low_frequency_cutoff)
                    weight = self.put(np.asarray(padded,dtype=np.float32))
                short.psdt = padded
                r.psds[delta_f] = short
                self.weights[delta_f] = (short,weight)
            short = r.psds[delta_f]
            if delta_f not in self.weights or self.weights[delta_f][0] is not short:
                self.weights[delta_f] = (short,self.put(np.asarray(short.psdt,dtype=np.float32)))
            key = (count,r.reduced_pad,r.trim_padding)
            if key not in self.windows:
                n = count-2*r.reduced_pad
                taper = r.trim_padding/2/r.sample_rate
                w = TimeSeries(np.ones(n,dtype=np.float32),delta_t=1/r.sample_rate)
                gate_data(w,[(w.start_time,0,taper),(w.end_time,0,taper)])
                window = self.put(np.asarray(w))
                self.windows[key] = window
            x = ring_slice(len(r.strain),count)(self.buffer,self.head)
            data = whitening_kernel(count,r.reduced_pad,r.sample_rate)(x,self.weights[delta_f][1],self.windows[key])
            epoch = r.strain.start_time + (len(r.strain)-count+r.reduced_pad)/r.sample_rate
            metadata = SpectrumMetadata(len(data),delta_f,epoch,short)
            self.spectra[delta_f] = (metadata,data)
            return metadata,data

    def condition_raw(self, strain):
        import jax
        import jax.numpy as jnp
        from pycbc.filter.resample import cached_firwin
        from pycbc.types import TimeSeries
        r = self.reader
        if len(strain) < 128:
            raise ValueError('JAX FIR requires at least 128 raw samples')
        with jax.default_device(self.device), jax.enable_x64(True):
            key = (len(strain),str(strain.dtype))
            if key not in self.responses:
                def response(coeff,dtype):
                    c = np.zeros(len(strain),dtype=dtype)
                    c[:len(coeff)] = coeff[::-1]
                    c = np.roll(c,len(c)-len(coeff)+1)
                    return jnp.conj(jnp.fft.rfft(self.put(c)))
                high = cached_firwin(2*r.highpass_samples+1,
                    r.highpass_frequency/(int(1/strain.delta_t)/2),
                    window=('kaiser',r.beta),pass_zero=False)
                down = cached_firwin(r.factor*20+1,1/r.factor,
                    window=('kaiser',5)) if r.factor != 1 else [1.]
                self.responses[key] = (response(high,strain.dtype),response(down,np.float32))
            high,down = self.responses[key]
            result = condition_kernel(r.highpass_samples,r.factor*10,r.factor,r.dyn_range_fac)(self.put(np.asarray(strain)),high,down)
            return TimeSeries(np.asarray(result),delta_t=1/r.sample_rate,epoch=strain.start_time)

    def welch(self, strain, seg_len, seg_stride):
        import jax
        from pycbc.types import FrequencySeries
        segments = len(strain)//seg_stride
        if (segments-1)*seg_stride+seg_len > len(strain): segments -= 1
        count = (segments-1)*seg_stride+seg_len
        start = (len(strain)-count+1)//2
        with jax.default_device(self.device):
            r = self.reader
            if (hasattr(r, 'strain') and strain.end_time == r.strain.end_time and
                    np.shares_memory(np.asarray(strain),np.asarray(r.strain))):
                self.sync()
                x = ring_slice(len(r.strain),len(strain))(self.buffer,self.head)[start:start+count]
            else:
                x = self.put(np.asarray(strain[start:start+count]))
            result = welch_kernel(seg_len,seg_stride,segments,strain.sample_rate,str(strain.dtype))(x)
        return FrequencySeries(np.asarray(result),delta_f=strain.sample_rate/seg_len,epoch=strain.start_time+start*strain.delta_t)

    def detect_loud_glitches(self, strain, psd_duration, psd_stride, threshold,
                             cluster_window, low_freq_cutoff, corrupt_time):
        """GPU spectral math; native FindChirp clustering and gate decisions."""
        import jax
        from pycbc.psd import interpolate
        from pycbc.events import findchirp_cluster_over_window
        x = strain.copy()
        corrupt = int(corrupt_time*x.sample_rate)
        window = np.arange(corrupt)/float(corrupt) if corrupt else np.empty(0)
        if corrupt:
            x.numpy()[:corrupt] *= window.astype(x.dtype)
            x.numpy()[len(x)-corrupt:] *= window[::-1].astype(x.dtype)
        length = 1 << (len(x)-1).bit_length()
        p = self.welch(x[corrupt:len(x)-corrupt] if corrupt else x,
                       int(psd_duration*x.sample_rate),int(psd_stride*x.sample_rate))
        p = interpolate(p,x.sample_rate/length)
        _,weight = self.inverse(p,int(psd_duration*x.sample_rate),low_freq_cutoff,hann=True)
        with jax.default_device(self.device):
            mag = np.array(glitch_kernel(len(x),length,x.sample_rate,low_freq_cutoff)(
                self.put(np.asarray(x)),weight),copy=True)

        if corrupt:
            mag[:corrupt] = 0; mag[-corrupt:] = 0
        indices = np.flatnonzero(mag > threshold)
        chosen = findchirp_cluster_over_window(indices,np.array(mag[indices]),int(cluster_window*x.sample_rate))
        return [idx*x.delta_t+x.start_time for idx in indices[chosen]]
