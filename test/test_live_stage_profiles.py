"""Performance attribution preserves per-rank overlap and startup scope."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage_analysis',ROOT/'examples/live/analyze_backend_performance.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def test_stage_analysis_does_not_add_concurrent_rank_times(tmp_path):
    stages=tmp_path/'stages';stages.mkdir()
    for rank in [0,1]:
        records=[dict(phase='startup',rank=rank,setup_to_loop_s=2)]
        for i in range(3):
            records.append(dict(phase='block',rank=rank,block_s=3,mpi_s=1,
                detectors={'H1':dict(advance_s=.5,psd_check_s=.2,filter_s=.4,
                    jax=dict(surface_s=.1,vetoes_s=.2),device_buffer=dict(uploaded_bytes=1024),
                    **({'batching':dict(batches=4,selected=8)} if i == 1 else {}))}))
        (stages/f'rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    report=analysis.analyze(dict(runs=[dict(mode='jax_gpu',repeat=0,returncode=0,
                                           wall_s=11,directory=str(tmp_path))]))
    row=report['runs'][0]
    assert row['outside_rank0_loop_s']==2
    assert row['ranks']['rank1']['stages']['H1/jax/surface_s']['median_s']==.1
    assert row['ranks']['rank1']['buffer_transfers']['H1']['uploaded_bytes']==1024
    assert row['ranks']['rank0']['blocks']==3
    assert row['ranks']['rank1']['batching']['H1']['selected']==8
