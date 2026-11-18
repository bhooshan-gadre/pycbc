"""Host-ready timing of the waveform and matched-filter stages, without vetoes."""
import time
import numpy as np


def measure(call, repeats):
    began = time.perf_counter()
    call()
    cold = time.perf_counter() - began
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        call()
        samples.append(time.perf_counter()-began)
    return dict(first_s=cold, median_s=float(np.median(samples)),
                p95_s=float(np.percentile(samples, 95)), samples_s=samples)


def benchmark_waveforms(bank, templates, devices, repeats):
    """Same grid/distance, native LAL versus production ripple host-ready path.

    Include serial and compiled batched ripple generation. First_s may include compilation but is not guaranteed
    a clean-process cold compilation measurement.
    """
    import jax
    from pycbc import DYN_RANGE_FAC
    from pycbc.waveform.jax_phenom import _ripple_phenomd, phenomd_implementation, generate_imrphenomd_batch
    from pycbc.pnutils import mass1_mass2_to_mchirp_eta
    if phenomd_implementation() != 'ripple':
        raise RuntimeError('Waveform timing requires real ripplegw IMRPhenomD')
    native = lambda: [bank.get_template(int(t.id), delta_f=t.delta_f) for t in templates]
    result = {'native_lal': measure(native, repeats)}
    groups={}
    for t in templates:groups.setdefault((len(t),t.delta_f),[]).append(t)
    specs=[]
    for (length,df),group in groups.items():
        params=[]
        for t in group:
            m1,m2=float(t.params.mass1),float(t.params.mass2)
            c1,c2=float(t.params.spin1z),float(t.params.spin2z)
            if m1 < m2:c1,c2=c2,c1
            mc,eta=mass1_mass2_to_mchirp_eta(m1,m2)
            params.append([mc,eta,c1,c2,1/DYN_RANGE_FAC,0,0,0])
        lower=np.array([float(t.f_lower) for t in group])
        specs.append((length,df,params,lower))
    for platform, device in devices.items():
        def generate():
            with jax.default_device(device):
                return [_ripple_phenomd(len(t), t.delta_f, float(t.params.f_lower),
                        0, 1/DYN_RANGE_FAC, float(t.params.mass1), float(t.params.mass2),
                        float(t.params.spin1z), float(t.params.spin2z)) for t in templates]
        result['ripple_'+platform] = dict(device=str(device), **measure(generate, repeats))
        def generate_batch():
            return [np.asarray(generate_imrphenomd_batch(length,df,params,low,low,
                    (length-1)*df,device)) for length,df,params,low in specs]
        result['ripple_batch_'+platform] = dict(device=str(device), **measure(generate_batch,repeats))
    for item in result.values():
        item['speedup_vs_native_lal'] = result['native_lal']['median_s']/item['median_s']
    import matplotlib
    matplotlib.use('Agg')
    # Caller saves the JSON; a paired component plot is made by plot_components.
    return dict(scope='full-bank serial and compiled batched waveform generation, host-ready complex64; no filtering',
                templates=len(templates), timings=result)


def benchmark_jax_filter(engine, start, stop, repeats):
    import jax
    def call():
        with jax.default_device(engine.device):
            for state in engine._states:
                # Includes correlation, IFFT, normalization/peak reduction;
                # input/PSD already resident. No power/SG veto computation.
                engine._evaluate_surface(state, start, stop-start).block_until_ready()
    return measure(call, repeats)


def benchmark_native_filter(serial, start, stop, repeats):
    from pycbc.filter import matched_filter_core
    def call():
        for t in serial.templates:
            rho, _, norm = matched_filter_core(t, serial.reader._stilde,
                h_norm=float(t.sigmasq(serial.reader._stilde.psd)),
                out=serial.out, corr_out=serial.corr)
            # Match the JAX peak-reduction work and host readiness.
            int(np.argmax(np.abs(np.asarray(rho)[start:stop])**2))
    return measure(call, repeats)


def plot_components(directory):
    """Read saved JSON reports; do not conflate setup or waveform time with FFTs."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    directory = Path(directory)
    w = json.loads((directory/'waveform_timings.json').read_text())['timings']
    cases = [json.loads(p.read_text()) for p in sorted(directory.glob('*/case.json'))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(list(w), [1000*v['median_s'] for v in w.values()])
    axes[0].set_title('Waveform generation: entire bank, host ready')
    if cases:
        times = cases[0]['timings']
        mf = {k: v['matched_filter_only'] for k,v in times.items() if 'matched_filter_only' in v}
        axes[1].bar(list(mf), [1000*v['median_s'] for v in mf.values()])
    axes[1].set_title('Matched filter/peak reduction; no vetoes\n' + (cases[0]['dataset'] if cases else 'no cases'))
    for ax in axes:
        ax.set_ylabel('Median milliseconds'); ax.tick_params(axis='x', rotation=30)
    fig.tight_layout(); fig.savefig(directory/'component_timings.png', dpi=150); plt.close(fig)


def benchmark_native_batch(engine, reader, start, stop, repeats):
    def call():
        for i, group in enumerate(engine.tgroups):
            engine.corr[i].execute(reader.overwhitened_data(group[0].delta_f))
            engine.ifts[engine.mids[i]].execute()
            for template in group:
                # Same cached PSD normalization and peak work as serial.
                template.sigmasq(reader._stilde.psd)
                np.argmax(np.abs(np.asarray(template.out)[start:stop])**2)
    return measure(call, repeats)
