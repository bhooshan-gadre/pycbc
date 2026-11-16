#!/usr/bin/env python
"""CPU versus JAX live-filter parity for TaylorF2 and IMRPhenomD."""

import os
import subprocess
import sys


def test_jax_live_filter_parity():
    script = os.path.join(
        os.path.dirname(__file__), '..', 'examples', 'live', 'compare_jax_live.py')
    script = os.path.abspath(script)
    source = os.path.abspath(os.path.join(os.path.dirname(script), '..', '..'))
    env = os.environ.copy()
    env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
    completed = subprocess.run(
        [sys.executable, script, '--jax-platform', 'cpu'],
        check=False, env=env)
    assert completed.returncode == 0, completed.returncode
