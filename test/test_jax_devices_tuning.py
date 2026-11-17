"""GPU assignment, measured tuning decisions, and numerical parity."""
import os
import numpy as np
import pytest
from pycbc.filter.jax_devices import plan_gpu_assignment, visible_gpu_tokens
from pycbc.filter.jax_tuning import choose_candidate, memory_estimate


def record(rank, tokens, requested='auto', gpu=True):
    return dict(rank=rank, visible=tokens, requested=requested, gpu=gpu)


def test_assignment_respects_overlapping_masks_and_coordinator():
    rows = [record(0, [], gpu=False), record(1, ['GPU-A','GPU-B']),
            record(2, ['GPU-A'])]
    assert plan_gpu_assignment(rows) == {1:'GPU-B', 2:'GPU-A'}
    assert plan_gpu_assignment([record(1,['2']),record(2,['3'])]) == {1:'2',2:'3'}


def test_assignment_explicit_conflicts_and_sharing():
    with pytest.raises(ValueError, match='Not enough'):
        plan_gpu_assignment([record(1,['0']), record(2,['0'])])
    with pytest.raises(ValueError, match='outside'):
        plan_gpu_assignment([record(1,['0'], '1')])
    with pytest.raises(ValueError, match='explicitly'):
        plan_gpu_assignment([record(1,['0'],'0'),record(2,['0'],'0')])
    assert plan_gpu_assignment([record(1,['0']),record(2,['0'])], True) == {1:'0',2:'0'}
    assert visible_gpu_tokens({'CUDA_VISIBLE_DEVICES':'GPU-B, GPU-A'}) == ['GPU-B','GPU-A']
    assert visible_gpu_tokens({'CUDA_VISIBLE_DEVICES':'-1'}) == []


def test_tuner_uses_worst_latency_and_rejects_all_invalid():
    rows = [dict(batch_size=1, status='measured', objective_s=.002,p95_s=.004),
            dict(batch_size=4, status='measured', objective_s=.001,p95_s=.006),
            dict(batch_size=8, status='memory_skipped')]
    assert choose_candidate(rows) == 4
    assert choose_candidate(rows, 5) == 1
    with pytest.raises(RuntimeError, match='No batch'):
        choose_candidate(rows, 1)
    groups = [(list(range(10)), np.zeros((10,129), dtype=np.complex64))]
    assert memory_estimate(groups,4) > memory_estimate(groups,1)


def test_duration_chunks_fractional_limit_and_fft_grid():
    from pycbc.filter.jax_matchedfilter import _duration_chunks
    class Template:
        def __init__(self, n, df): self.n,self.delta_f=n,df
        def __len__(self): return self.n
    templates = [Template(129, .25) for _ in range(5)] + [Template(257, .125)]
    ordered, sizes = _duration_chunks(templates, 640)
    assert list(sizes) == [2,2,1,1]
    assert sum(sizes) == len(ordered)
    with pytest.raises(ValueError): _duration_chunks([],10)


def test_tuned_filter_parity(tmp_path):
    import subprocess
    import sys
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               JAX_PLATFORMS='cpu', MPLCONFIGDIR=str(tmp_path/'mpl'))
    result = subprocess.run([sys.executable,str(root/'examples/search/jax/validate_cpu_gpu.py'),
        '--gpu','off','--approximants','TaylorF2','--bank-sizes','2','--repeats','1',
        '--veto-samples','3','--autotune','--tuning-repeats','1','--output',str(tmp_path)],
        env=env, cwd=root, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((tmp_path/'summary.json').read_text())
    for case in report['cases']:
        tuned = case['implementations']['jax_cpu_tuned']
        assert tuned['filter_only']['passed']
        assert tuned['tuning']['batch_size'] in [1,2]
        assert len(tuned['tuning']['candidates']) == 2


def test_assignment_rejects_alias_ambiguity():
    with pytest.raises(ValueError, match='Mixed numeric'):
        plan_gpu_assignment([record(1,['0']), record(2,['GPU-A'])])
    with pytest.raises(ValueError, match='abbreviated'):
        plan_gpu_assignment([record(1,['GPU-A']), record(2,['GPU-AB'])])
    assert plan_gpu_assignment([record(1,['1']),record(2,['10'])]) == {1:'1',2:'10'}


def test_strict_device_selection(monkeypatch):
    from types import SimpleNamespace
    import pycbc.filter.jax_matchedfilter as mf
    cpu, gpu = object(), object()
    fake = SimpleNamespace(config=SimpleNamespace(update=lambda *args: None),
                           devices=lambda platform: [cpu] if platform == 'cpu' else [gpu])
    monkeypatch.setattr(mf, '_import_jax', lambda: (fake, None))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-A,GPU-B')
    assert mf.configure_jax('cpu')[0] is cpu
    assert mf.configure_jax('gpu')[0] is gpu
    for index in [-1,1]:
        with pytest.raises(ValueError, match='outside'):
            mf.configure_jax('gpu',index)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-A,GPU-B'


def test_tuning_budget_failure_restores_thresholds():
    from types import SimpleNamespace
    from pycbc.filter.jax_tuning import tune_live_filter
    engine = SimpleNamespace(_bank_groups=[([0],np.zeros((1,129)))],
        tuning_candidates=[1,2], device=SimpleNamespace(memory_stats=lambda:None, platform='cpu'),
        memory_budget_mb=1e-9, max_batch_latency_ms=None,
        snr_threshold=5, snr_abort_threshold=500, newsnr_threshold=4,
        max_triggers_in_batch=30, sg_chisq=SimpleNamespace(do=False,snr_threshold=4),
        _tuning=False)
    with pytest.raises(RuntimeError, match='No batch candidate'):
        tune_live_filter(engine,None)
    assert not engine._tuning
    assert engine.snr_threshold == 5 and engine.max_triggers_in_batch == 30
    assert engine.sg_chisq.snr_threshold == 4
