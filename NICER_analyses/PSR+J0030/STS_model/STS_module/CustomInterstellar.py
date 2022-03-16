from __future__ import print_function, division

import numpy as np
import math

import xpsi
from xpsi import Parameter

from scipy.interpolate import Akima1DInterpolator

class CustomInterstellar(xpsi.Interstellar):
    """ Apply interstellar attenuation. """

    def __init__(self, energies, attenuation, bounds, values = {}):

        assert len(energies) == len(attenuation), 'Array length mismatch.'

        self._lkp_energies = energies # for lookup
        self._lkp_attenuation = attenuation # for lookup

        N_H = Parameter('column_density',
                        strict_bounds = (0.0,10.0),
                        bounds = bounds.get('column_density', None),
                        doc = 'Units of 10^20 cm^-2.',
                        symbol = r'$N_{\rm H}$',
                        value = values.get('column_density', None))

        self._interpolator = Akima1DInterpolator(self._lkp_energies,
                                                 self._lkp_attenuation)
        self._interpolator.extrapolate = True

        super(CustomInterstellar, self).__init__(N_H)

    def attenuation(self, energies):
        """ Interpolate the attenuation coefficients.

        Useful for post-processing.

        """
        return self._interpolate(energies)**(self['column_density']/0.4)

    def _interpolate(self, energies):
        """ Helper. """
        _att = self._interpolator(energies)
        _att[_att < 0.0] = 0.0
        return _att

    @classmethod
    def from_SWG(cls, path, **kwargs):
        """ Load attenuation file from the NICER SWG. """

        temp = np.loadtxt(path, dtype=np.double)

        energies = temp[0:351,0]

        attenuation = temp[0:351,2]

        return cls(energies, attenuation, **kwargs)
