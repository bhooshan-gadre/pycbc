"""The full-live comparison must retain missing rows and strict veto errors."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'examples/live/benchmark_imrphenomd.py'
spec = importlib.util.spec_from_file_location('live_bench', SCRIPT)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def product():
    return {('block.hdf','H1',0):dict(snr=20.,chisq=1.,sg_chisq=1.,
                                   end_time=100.,coa_phase=.3)}


def test_comparison_wraps_phase_and_rejects_missing_and_veto_error(tmp_path):
    ref = product(); actual = product()
    actual[next(iter(actual))]['coa_phase'] += 2*np.pi
    assert b.compare(ref,actual,2048,tmp_path,'wrapped')['passed']
    actual[next(iter(actual))]['chisq'] += .1
    report = b.compare(ref,actual,2048,tmp_path,'bad')
    assert not report['passed']
    assert report['same_peak_time_rows'] == 1
    assert report['metrics']['chisq']['max_same_peak_time_error'] > .09
    assert (tmp_path/'bad_live_rows.csv').is_file()
    assert not b.compare(ref,{},2048,tmp_path,'missing')['passed']


def test_collect_does_not_silently_accept_no_triggers(tmp_path):
    with pytest.raises(RuntimeError, match='No live trigger'):
        b.collect(tmp_path)
