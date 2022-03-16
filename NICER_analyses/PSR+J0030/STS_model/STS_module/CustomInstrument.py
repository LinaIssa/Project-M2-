from __future__ import print_function, division

import numpy as np
import math

import xpsi

from xpsi import Parameter, make_verbose

class CustomInstrument(xpsi.Instrument):
    """ Methods and attributes specific to the NICER instrument.

    Currently tailored to the NICER light-curve SWG model specification.

    """
    def construct_matrix(self):
        """ Implement response matrix parameterisation. """
        # Multiplying beta to response matrix
        beta_d = self['beta'] * 0.01**2  # beta_d = beta * (d kpc)^2
        matrix = beta_d*self.matrix
        matrix[matrix < 0.0] = 0.0

        return matrix

    def __call__(self, signal, *args):
        """ Overwrite. """

        matrix = self.construct_matrix()

        self._cached_signal = np.dot(matrix, signal)

        return self._cached_signal

    @classmethod
    @make_verbose('Loading response matrix',
                  'Response matrix loaded')
    def from_SWG(cls,
                 bounds, values,
                 ARF, RMF,
                 max_input, min_input=0,
                 channel_edges = None):
        """ Constructor which converts files into :class:`numpy.ndarray`s.

        :param str ARF: Path to ARF which is compatible with
                                :func:`numpy.loadtxt`.

        :param str RMF: Path to RMF which is compatible with
                                :func:`numpy.loadtxt`.

        """
        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=3)
        RMF = np.loadtxt(RMF, dtype=np.double, skiprows=3, usecols=-1)

        if channel_edges:
            channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)

        matrix = np.zeros((1501,3451))

        for i in range(3451):
            matrix[:,i] = RMF[i*1501:(i+1)*1501]

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        edges = np.zeros(ARF[min_input:max_input,3].shape[0]+1, dtype=np.double)

        edges[0] = ARF[min_input,1]; edges[1:] = ARF[min_input:max_input,2]

        RSP = np.ascontiguousarray(np.zeros(matrix[30:300,min_input:max_input].shape), dtype=np.double)

        for i in range(RSP.shape[0]):
            RSP[i,:] = matrix[i+30,min_input:max_input] * ARF[min_input:max_input,3]*49./52.

        channels = np.arange(30, 300)

        beta = Parameter('beta',
                          strict_bounds = (0.1,30.0),
                          bounds = bounds.get('beta', None),
                          doc='Units of kpc^-2',
                          symbol = r'$\beta$',
                          value = values.get('beta', None))

        return cls(RSP, edges, channels, channel_edges[30:301,1], beta)
