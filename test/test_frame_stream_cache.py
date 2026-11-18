"""Reuse unchanged incremental frame streams without hiding updated data."""
from types import SimpleNamespace
import pytest


def test_incremental_stream_reuse_and_file_invalidation(tmp_path,monkeypatch):
    from pycbc.frame import frame as module
    buffer=object.__new__(module.DataBuffer)
    buffer.raw_buffer=SimpleNamespace(end_time=104)
    buffer.beg='H1-TEST';buffer.ref=100;buffer.dur=32
    buffer.increment_update_cache=str(tmp_path)
    buffer.channel_name='H1:STRAIN';buffer.stream=object()
    calls=[]
    monkeypatch.setattr(module,'locations_to_cache',lambda keys:list(keys))
    def open_stream(cache):
        calls.append(tuple(cache));return object()
    monkeypatch.setattr(module.lalframe,'FrStreamCacheOpen',open_stream)
    buffer._retrieve_metadata=lambda stream,channel:('float',512)
    frame=tmp_path/'H1-TEST-100-32.gwf';frame.write_bytes(b'first frame')
    buffer.update_cache_by_increment(8);first=buffer.stream
    buffer.raw_buffer.end_time=112
    buffer.update_cache_by_increment(8)
    assert buffer.stream is first and len(calls)==1
    frame.write_bytes(b'updated frame data')
    buffer.update_cache_by_increment(8)
    assert buffer.stream is not first and len(calls)==2
    # An explicit stream/cache refresh must not be overridden by a stale hit.
    buffer.stream=object();buffer.update_cache_by_increment(8)
    assert len(calls)==3
    next_frame=tmp_path/'H1-TEST-132-32.gwf';next_frame.write_bytes(b'next')
    buffer.raw_buffer.end_time=132;buffer.update_cache_by_increment(8)
    assert calls[-1]==(str(next_frame),) and len(calls)==4
    next_frame.unlink()
    with pytest.raises(RuntimeError):buffer.update_cache_by_increment(8)


def test_real_incremental_frames_preserve_samples_and_reload_replacement(tmp_path):
    import numpy as np
    from pycbc.frame import write_frame
    from pycbc.frame.frame import DataBuffer
    from pycbc.types import TimeSeries
    rng=np.random.default_rng(581);rate=512
    samples=rng.normal(size=64*rate)
    first=tmp_path/'H1-TEST-100-32.gwf'
    for start in [100,132]:
        offset=(start-100)*rate
        write_frame(str(tmp_path/f'H1-TEST-{start}-32.gwf'),'H1:STRAIN',
            TimeSeries(samples[offset:offset+32*rate],delta_t=1/rate,epoch=start))
    buffer=DataBuffer([str(tmp_path/'H1-TEST-*.gwf')],'H1:STRAIN',100,
                      max_buffer=8,force_update_cache=False,increment_update_cache=str(tmp_path))
    streams=[]
    for block in range(6):
        if block==2:
            samples[16*rate:32*rate]+=10
            replacement=tmp_path/'replacement.gwf'
            write_frame(str(replacement),'H1:STRAIN',TimeSeries(samples[:32*rate],delta_t=1/rate,epoch=100))
            replacement.replace(first)
        actual=buffer.attempt_advance(8,timeout=0)
        np.testing.assert_array_equal(np.asarray(actual),samples[block*8*rate:(block+1)*8*rate])
        streams.append(buffer.stream)
    assert streams[0] is streams[1]
    assert streams[2] is not streams[1] and streams[2] is streams[3]
    assert streams[4] is not streams[3] and streams[4] is streams[5]


def test_failed_read_invalidates_stream_before_retry(monkeypatch):
    from pycbc.frame import frame as module
    buffer=object.__new__(module.DataBuffer)
    buffer.force_update_cache=False;buffer.increment_update_cache='frames'
    buffer.raw_buffer=SimpleNamespace(end_time=100)
    buffer._increment_cache_signature='old'
    buffer._increment_cache_stream=object()
    buffer.update_cache_by_increment=lambda blocksize:None
    seen=[]
    def advance(self,blocksize):
        seen.append(self._increment_cache_signature)
        if len(seen)==1:raise RuntimeError('transient read error')
        return 'read_ok'
    monkeypatch.setattr(module.DataBuffer,'advance',advance)
    monkeypatch.setattr(module.pycbc,'gps_now',lambda:0)
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    assert buffer.attempt_advance(8)=='read_ok'
    assert seen==['old',None]
