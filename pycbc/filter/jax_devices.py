"""MPI GPU binding without initializing JAX on unassigned devices."""
import os
import subprocess
import sys


def visible_gpu_tokens(environ=None):
    """Return scheduler-visible CUDA identifiers, preserving their order."""
    env = os.environ if environ is None else environ
    mask = env.get('CUDA_VISIBLE_DEVICES')
    if mask is not None:
        return [x.strip() for x in mask.split(',') if x.strip() and x.strip() != '-1']
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        check=True, capture_output=True, text=True, timeout=15)
    return [x.strip() for x in result.stdout.splitlines() if x.strip()]


def plan_gpu_assignment(records, allow_sharing=False):
    """Match ranks to visible GPU tokens, including overlapping masks.

    This pure function runs separately for each shared-memory MPI node.
    Explicit indices are fixed; automatic ranks use augmenting-path matching.
    """
    records = sorted((r for r in records if r['gpu']), key=lambda r: r['rank'])
    tokens = {token for r in records for token in r['visible']}
    numeric = {t for t in tokens if t.isdecimal()}
    if numeric and numeric != tokens:
        raise ValueError('Mixed numeric/UUID visibility masks cannot be safely disambiguated; use GPU UUIDs')
    if not numeric and any(a != b and b.startswith(a) for a in tokens for b in tokens):
        raise ValueError('Ambiguous abbreviated GPU identifiers; use complete UUIDs')
    by_rank = {r['rank']: r for r in records}
    owners, selected, fixed = {}, {}, set()
    for r in records:
        tokens = r['visible']
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError(f"rank {r['rank']}: empty or duplicated visible GPU list")
        requested = r.get('requested', 'auto')
        if requested == 'auto':
            continue
        index = int(requested)
        if not 0 <= index < len(tokens):
            raise ValueError(f"rank {r['rank']}: GPU index {index} outside its visible list")
        token = tokens[index]
        if token in owners and not allow_sharing:
            raise ValueError(f'GPU {token} explicitly assigned to more than one local rank')
        owners[token], selected[r['rank']] = r['rank'], token
        fixed.add(r['rank'])

    def place(rank, seen):
        for token in by_rank[rank]['visible']:
            if token in seen:
                continue
            seen.add(token)
            old = owners.get(token)
            if old is None or (old not in fixed and place(old, seen)):
                owners[token], selected[rank] = rank, token
                return True
        return False

    for r in records:
        rank = r['rank']
        if rank in fixed:
            continue
        if not place(rank, set()):
            if not allow_sharing:
                raise ValueError('Not enough distinct visible GPUs for local filtering ranks; '
                                 'use fewer GPU ranks or explicitly allow sharing')
            selected[rank] = r['visible'][rank % len(r['visible'])]
    return selected


def bind_mpi_gpu(comm, gpu_rank, requested='auto', allow_sharing=False):
    """Collective on *all* ranks, including CPU/coordinator ranks.

    Restrict each GPU rank to one identifier from its existing scheduler mask
    before JAX import. Return visible device index zero and binding metadata.
    All ranks propagate binding failures collectively rather than hanging.
    """
    from mpi4py import MPI
    rank = comm.Get_rank()
    error, visible = None, []
    try:
        if gpu_rank:
            if 'jax' in sys.modules:
                raise RuntimeError('MPI GPU binding must run before importing JAX')
            visible = visible_gpu_tokens()
    except Exception as exc:
        error = f'rank {rank}: {exc}'
    local = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    record = dict(rank=rank, gpu=bool(gpu_rank), visible=visible,
                  requested=requested, error=error)
    records = local.allgather(record)
    selection = {}
    try:
        errors = [r['error'] for r in records if r['error']]
        if errors:
            raise RuntimeError('; '.join(errors))
        selection = plan_gpu_assignment(records, allow_sharing)
    except Exception as exc:
        error = str(exc)
    finally:
        local.Free()
    errors = [x for x in comm.allgather(error) if x]
    if errors:
        raise RuntimeError('MPI GPU binding failed: ' + '; '.join(sorted(set(errors))))
    if not gpu_rank:
        return None
    token = selection[rank]
    original = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = token
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    return dict(world_rank=rank, original_visibility=original,
                selected_gpu=token, device_index=0, sharing_allowed=allow_sharing)
