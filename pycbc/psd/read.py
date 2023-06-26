#!/usr/bin/python
# Copyright (C) 2012 Alex Nitz, Tito Dal Canton
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Utilities to read PSDs from files.
"""

import logging
import numpy
import h5py
import scipy.interpolate

from ligo import segments
from pycbc.types import FrequencySeries, load_frequencyseries, zeros, float32
import pycbc.psd


def from_numpy_arrays(freq_data, noise_data, length, delta_f, low_freq_cutoff):
    """Interpolate n PSD (as two 1-dimensional arrays of frequency and data)
    to the desired length, delta_f and low frequency cutoff.

    Parameters
    ----------
    freq_data : array
        Array of frequencies.
    noise_data : array
        PSD values corresponding to frequencies in freq_arr.
    length : int
        Length of the frequency series in samples.
    delta_f : float
        Frequency resolution of the frequency series in Herz.
    low_freq_cutoff : float
        Frequencies below this value are set to zero.

    Returns
    -------
    psd : FrequencySeries
        The generated frequency series.
    """
    # Only include points above the low frequency cutoff
    if freq_data[0] > low_freq_cutoff:
        raise ValueError('Lowest frequency in input data '
                         ' is higher than requested low-frequency cutoff ' + str(low_freq_cutoff))

    kmin = int(low_freq_cutoff / delta_f)
    flow = kmin * delta_f

    data_start = (0 if freq_data[0] == low_freq_cutoff else numpy.searchsorted(
        freq_data, flow) - 1)

    # If the cutoff is exactly in the file, start there
    if freq_data[data_start+1] == low_freq_cutoff:
        data_start += 1

    freq_data = freq_data[data_start:]
    noise_data = noise_data[data_start:]

    if (length - 1) * delta_f > freq_data[-1]:
        logging.warning('Requested number of samples exceeds the highest '
                        'available frequency in the input data, '
                        'will use max available frequency instead. '
                        '(requested %f Hz, available %f Hz)',
                        (length - 1) * delta_f, freq_data[-1])
        length = int(freq_data[-1]/delta_f + 1)

    flog = numpy.log(freq_data)
    slog = numpy.log(noise_data)

    psd_interp = scipy.interpolate.interp1d(
        flog, slog, fill_value=(slog[0], slog[-1]), bounds_error=False)
    psd = numpy.zeros(length, dtype=numpy.float64)

    vals = numpy.log(numpy.arange(kmin, length) * delta_f)
    psd[kmin:] = numpy.exp(psd_interp(vals))

    return FrequencySeries(psd, delta_f=delta_f)


def from_txt(filename, length, delta_f, low_freq_cutoff, is_asd_file=True):
    """Read an ASCII file containing one-sided ASD or PSD  data and generate
    a frequency series with the corresponding PSD. The ASD or PSD data is
    interpolated in order to match the desired resolution of the
    generated frequency series.

    Parameters
    ----------
    filename : string
        Path to a two-column ASCII file. The first column must contain
        the frequency (positive frequencies only) and the second column
        must contain the amplitude density OR power spectral density.
    length : int
        Length of the frequency series in samples.
    delta_f : float
        Frequency resolution of the frequency series in Herz.
    low_freq_cutoff : float
        Frequencies below this value are set to zero.
    is_asd_file : Boolean
        If false assume that the second column holds power spectral density.
        If true assume that the second column holds amplitude spectral density.
        Default: True

    Returns
    -------
    psd : FrequencySeries
        The generated frequency series.

    Raises
    ------
    ValueError
        If the ASCII file contains negative, infinite or NaN frequencies
        or amplitude densities.
    """
    file_data = numpy.loadtxt(filename)
    if (file_data < 0).any() or \
            numpy.logical_not(numpy.isfinite(file_data)).any():
        raise ValueError('Invalid data in ' + filename)

    freq_data = file_data[:, 0]
    noise_data = file_data[:, 1]
    if is_asd_file:
        noise_data = noise_data ** 2

    return from_numpy_arrays(freq_data, noise_data, length, delta_f,
                             low_freq_cutoff)


def from_xml(filename, length, delta_f, low_freq_cutoff, ifo_string=None,
             root_name='psd'):
    """Read an ASCII file containing one-sided ASD or PSD  data and generate
    a frequency series with the corresponding PSD. The ASD or PSD data is
    interpolated in order to match the desired resolution of the
    generated frequency series.

    Parameters
    ----------
    filename : string
        Path to a two-column ASCII file. The first column must contain
        the frequency (positive frequencies only) and the second column
        must contain the amplitude density OR power spectral density.
    length : int
        Length of the frequency series in samples.
    delta_f : float
        Frequency resolution of the frequency series in Herz.
    low_freq_cutoff : float
        Frequencies below this value are set to zero.
    ifo_string : string
        Use the PSD in the file's PSD dictionary with this ifo string.
        If not given and only one PSD present in the file return that, if not
        given and multiple (or zero) PSDs present an exception will be raised.
    root_name : string (default='psd')
        If given use this as the root name for the PSD XML file. If this means
        nothing to you, then it is probably safe to ignore this option.

    Returns
    -------
    psd : FrequencySeries
        The generated frequency series.

    """
    import lal.series
    from ligo.lw import utils as ligolw_utils

    with open(filename, 'rb') as fp:
        ct_handler = lal.series.PSDContentHandler
        xml_doc = ligolw_utils.load_fileobj(fp, compress='auto',
                                            contenthandler=ct_handler)
        psd_dict = lal.series.read_psd_xmldoc(xml_doc, root_name=root_name)

    if ifo_string is not None:
        psd_freq_series = psd_dict[ifo_string]
    elif len(psd_dict.keys()) == 1:
        psd_freq_series = psd_dict[tuple(psd_dict.keys())[0]]
    else:
        err_msg = "No ifo string given and input XML file contains not "
        err_msg += "exactly one PSD. Specify which PSD you want to use."
        raise ValueError(err_msg)

    noise_data = psd_freq_series.data.data[:]
    freq_data = numpy.arange(len(noise_data)) * psd_freq_series.deltaF

    return from_numpy_arrays(freq_data, noise_data, length, delta_f,
                             low_freq_cutoff)


class PrecomputedTimeVaryingPSD(object):
    def __init__(self, opt, length, delta_f, sample_rate):
        self.opt = opt
        self.file_name = opt.precomputed_psd_file
        self.f_low = opt.low_frequency_cutoff
        self.length = length
        self.delta_f = delta_f
        self.sample_rate = sample_rate
        self.psd_inverse_length = opt.psd_inverse_length
        self.invpsd_trunc_method = opt.invpsd_trunc_method

        with h5py.File(self.file_name, 'r') as f:
            detector = tuple(f.keys())[0]
            self.start_times = f[detector + '/start_time'][:]
            self.end_times = f[detector + '/end_time'][:]
            self.file_f_low = f.attrs['low_frequency_cutoff']

        self.begin = self.start_times.min()
        self.end = self.end_times.max()
        self.detector = detector

    def assosiate_psd_to_inspiral_segment(self, inp_seg, delta_f=None):
        '''Find 2 PSDs that are closest to the segment and choose the best
        based on the maximum overlap.
        '''
        best_psd = None
        if inp_seg[0] > self.end or inp_seg[1] < self.begin:
            err_msg = "PSD file doesn't contain require times. \n"
            err_msg += "PSDs are within range ({}, {})".format(
                self.begin, self.end)
            raise ValueError(err_msg)
        sidx = numpy.argpartition(
            numpy.abs(self.start_times - inp_seg[0]), 2)[:2]
        nearest = segments.segment(
            self.start_times[sidx[0]], self.end_times[sidx[0]])
        next_nearest = segments.segment(
            self.start_times[sidx[1]], self.end_times[sidx[1]])

        psd_overlap = 0
        if inp_seg.intersects(nearest):
            psd_overlap = abs(nearest & inp_seg)
            best_psd = self.get_psd(sidx[0], delta_f)

        if inp_seg.intersects(next_nearest):
            if psd_overlap < abs(next_nearest & inp_seg):
                psd_overlap = abs(next_nearest & inp_seg)
                best_psd = self.get_psd(sidx[1], delta_f)

        if best_psd is None:
            err_msg = "No PSDs found intersecting segment!"
            raise ValueError(err_msg)

        if self.psd_inverse_length:
            best_psd = pycbc.psd.inverse_spectrum_truncation(best_psd,
                                              int(self.psd_inverse_length *
                                                  self.sample_rate),
                                              low_frequency_cutoff=self.f_low,
                                              trunc_method=self.invpsd_trunc_method)

        return best_psd

    def get_psd(self, index, delta_f=None):
        group = self.detector + '/psds/' + str(index)
        psd = load_frequencyseries(self.file_name, group=group)
        if delta_f is not None and psd.delta_f != delta_f:
            psd = pycbc.psd.interpolate(psd, delta_f)
        if self.length is not None and self.length != len(psd):
            psd2 = FrequencySeries(zeros(self.length, dtype=psd.dtype),
                                   delta_f=psd.delta_f)
            if self.length > len(psd):
                psd2[:] = numpy.inf
                psd2[0:len(psd)] = psd
            else:
                psd2[:] = psd[0:self.length]
            psd = psd2
        if self.f_low is not None and self.f_low < self.file_f_low:
            # avoid using the PSD below the f_low given in the file
            k = int(self.file_f_low / psd.delta_f)
            psd[0:k] = numpy.inf

        return psd
