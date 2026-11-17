"""Independent concatenated inverse transforms must not mix template rows."""
import numpy as np
import pytest

from pycbc.fft.npfft import IFFT
from pycbc.scheme import CPUScheme
from pycbc.types import Array, zeros


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128, np.float32, np.float64])
def test_batched_inverse_preserves_independent_dc_amplitudes(dtype):
    real_output = np.issubdtype(dtype, np.floating)
    spectrum_dtype = np.complex64 if dtype in (np.complex64, np.float32) else np.complex128
    size, count = 32, 3
    flen = size//2+1 if real_output else size
    amplitudes = np.array([1, 2, -4], dtype=spectrum_dtype)
    if not real_output:
        amplitudes += np.array([2j, -3j, 1j], dtype=spectrum_dtype)
    spectra = np.zeros((count, flen), dtype=spectrum_dtype)
    spectra[:, 0] = amplitudes
    with CPUScheme(num_threads=1):
        data = Array(spectra.reshape(-1))
        out = zeros(count*size, dtype=dtype)
        transform = IFFT(data, out, nbatch=count, size=size)
        for factor in [1, 2]:
            data.data[:] = (factor*spectra).reshape(-1)
            transform.execute()
            expected = np.broadcast_to(factor*amplitudes[:, None], (count, size))
            if real_output:
                expected = expected.real
            np.testing.assert_allclose(np.asarray(out).reshape(count, size), expected,
                                       rtol=1e-6, atol=1e-6)
