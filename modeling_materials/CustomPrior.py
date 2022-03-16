""" CustomPrior.py """

from __future__ import print_function, division

import numpy as np
import math
from scipy.stats import truncnorm

import xpsi
from xpsi.global_imports import _G, _csq, _km, _2pi
from xpsi.global_imports import gravradius, inv_gravradius

from xpsi.cellmesh.mesh_tools import eval_cedeCentreCoords

from scipy.interpolate import Akima1DInterpolator

class CustomPrior(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Source: PSR J0030+0451
    Model variant: ST+PST
        Two single-temperature hot regions with unshared parameters
        and different complexity levels.

    Parameter vector: (print the likelihood object)

    * p[0] = (rotationally deformed) gravitational mass (solar masses)
    * p[1] = coordinate equatorial radius (km)
    * p[2] = distance (kpc)
    * p[3] = cos(inclination of Earth to rotational axis)
    * p[4] = primary cap phase shift (cycles); (alias for initial azimuth, periodic)
    * p[5] = primary centre colatitude (radians)
    * p[6] = primary angular radius (radians)
    * p[7] = primary log10(comoving NSX FIH effective temperature [K])
    * p[8] = secondary cap phase shift (cycles)
    * p[9] = secondary centre colatitude (radians)
    * p[10] = secondary angular radius (radians)
    * p[11] = secondary omit colatitude (radians)
    * p[12] = secondary omit angular radius (radians)
    * p[13] = secondary omit azimuth (radians); periodic
    * p[14] = secondary log10(comoving NSX FIH effective temperature [K])
    * p[15] = hydrogen column density (10^20 cm^-2)
    * p[16] = instrument parameter alpha
    * p[17] = instrument parameter beta
    * p[18] = instrument parameter gamma

    """

    __derived_names__ = ['compactness',
                         's__annulus_width',
                         's__transformed_phase',
                         's__f',
                         's__xi',
                         's__super_offset_fraction',
                         's__super_offset_azi']

    a_f = 0.0
    b_f = 2.0
    a_xi = 0.001
    b_xi = math.pi/2.0 - a_xi

    vals = np.linspace(0.0, b_xi, 1000)

    def __init__(self):
        """ Construct mapping from unit interval. """

        self.interpolator = Akima1DInterpolator(self._vector_super_radius_mass(self.vals), self.vals)
        self.interpolator.extrapolate = True

    def __call__(self, p = None):
        """ Evaluate distribution at ``p``.

        :param list p: Model parameter values.

        :return: Logarithm of the distribution evaluated at ``p``.

        """
        temp = super(CustomPrior, self).__call__(p)
        if not np.isfinite(temp):
            return temp

        # based on contemporary EOS theory
        if not self.parameters['radius'] <= 16.0:
            return -np.inf

        ref = self.parameters.star.spacetime # shortcut

        # polar radius at photon sphere for ~static star (static ambient spacetime)
        R_p = 1.0 + ref.epsilon * (-0.788 + 1.030 * ref.zeta)
        if R_p < 1.5 / ref.R_r_s:
            return -np.inf

        # limit polar radius to try to exclude deflections >= \pi radians
        # due to oblateness this does not quite eliminate all configurations
        # with deflections >= \pi radians
        #if R_p < 1.76 / ref.R_r_s:
        #    return -np.inf

        mu = math.sqrt(-1.0 / (3.0 * ref.epsilon * (-0.788 + 1.030 * ref.zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # for high spin frequenies
        if mu < 1.0:
            return -np.inf

        ref = self.parameters # redefine shortcut

        phi = (0.5 + ref['s__phase_shift']) * _2pi
        phi -= ref['s__omit_azimuth']
        phi = ref['p__phase_shift'] * _2pi - phi

        ang_sep = xpsi.HotRegion.psi(ref['s__super_colatitude'],
                                     phi,
                                     ref['p__super_colatitude'])

        # hot regions cannot overlap
        if ang_sep < ref['p__super_radius'] + ref['s__super_radius']:
            return -np.inf

        return 0.0

    def _I(self, x):
        return x * np.log(self.b_xi/self.a_xi)

    def _II(self, x):
        return 2.0*(x - self.a_xi) - x*np.log(x/self.b_xi)

    def _scalar_super_radius_mass(self, x):
        if x >= self.a_xi:
            mass = self._II(x)
        else:
            mass = self._I(x)

        return mass

    def _vector_super_radius_mass(self, x):
        masses = np.zeros(len(x))

        for i, _ in enumerate(x):
            masses[i] = self._scalar_super_radius_mass(_)

        masses /= (self.b_f - self.a_f)
        masses /= (self.b_xi - self.a_xi)

        return masses

    def _inverse_sample_cede_radius(self, x, psi):
        if psi < self.a_xi:
            return self.a_xi*np.exp(x * np.log(self.b_xi/self.a_xi))
        elif psi >= self.a_xi and x <= 1.0/(1.0 + np.log(self.b_xi/psi)):
            return x*psi*(1.0 + np.log(self.b_xi/psi))
        else:
            return psi*np.exp(x*(1.0 + np.log(self.b_xi/psi)) - 1.0)

    def inverse_sample(self, hypercube = None):
        """ Draw sample uniformly from the distribution via inverse sampling.

        :param hypercube: A pseudorandom point in an n-dimensional hypercube.

        :return: A parameter ``list``.

        """
        to_cache = self.parameters.vector

        if hypercube is None:
            hypercube = np.random.rand(len(self))

        _ = super(CustomPrior, self).inverse_sample(hypercube)

        ref = self.parameters # redefine shortcut

        # draw from flat prior in inclination
        idx = ref.index('cos_inclination')
        a, b = ref.get_param('cos_inclination').bounds
        a = math.acos(a); b = math.acos(b)
        ref['cos_inclination'] = math.cos(b + (a - b) * hypercube[idx])

        idx = ref.index('distance')
        ref['distance'] = truncnorm.ppf(hypercube[idx],
                                        -10.0, 10.0,
                                        loc=0.325, scale=0.009)

        idx = ref.index('p__phase_shift')
        phase = 0.35 + 0.2 * hypercube[idx]
        if phase > 0.5:
            ref['p__phase_shift'] = phase - 1.0
        else:
            ref['p__phase_shift'] = phase

        idx = ref.index('s__phase_shift')
        phase = -0.25 + hypercube[idx]
        if phase > 0.5:
            ref['s__phase_shift'] = phase - 1.0
        else:
            ref['s__phase_shift'] = phase

        idx = ref.index('s__omit_radius')
        ref['s__omit_radius'] = float(self.interpolator(hypercube[idx]))

        idx = ref.index('s__super_radius')
        ref['s__super_radius'] = self._inverse_sample_cede_radius(hypercube[idx],
                                                                  ref['s__omit_radius'])

        idx = ref.index('s__super_colatitude')
        if ref['s__omit_radius'] <= ref['s__super_radius']:
            # temp var
            t = hypercube[idx] * (ref['s__super_radius'] + ref['s__omit_radius'])
        else:
            # temp var
            t = ref['s__omit_radius'] - ref['s__super_radius']
            t += 2.0 * hypercube[idx] * ref['s__super_radius']

        idx = ref.index('s__omit_azimuth')
        # temp var
        u = hypercube[idx] * _2pi

        # function from mesh tools module
        # in this case the ceding region is the "super" region, which
        # cedes to the omission region
        ref['s__super_colatitude'], ref['s__omit_azimuth'] = \
                eval_cedeCentreCoords(ref['s__omit_colatitude'], t, u)

        ref['s__omit_azimuth'] *= -1.0

        idx = ref.index('alpha')
        ref['alpha'] = truncnorm.ppf(hypercube[idx],
                                     -5.0, 5.0,
                                     loc=1.0, scale=0.1)

        idx = ref.index('gamma')
        ref['gamma'] = truncnorm.ppf(hypercube[idx],
                                     -5.0, 5.0,
                                     loc=1.0, scale=0.1)

        # restore proper cache
        for parameter, cache in zip(self.parameters, to_cache):
            parameter.cached = cache

        return self.parameters.vector # only free parameter values returned

    def transform(self, p, old_API = False):
        """ A transformation for post-processing.

        Note that if you want to use dictionary-like access to values,
        you could make a dictionary, e.g.:

        .. code-block:: python

            ref = dict(zip(self.parameters.names, p))

        and use the ``__getitem__`` functionality of ``ref`` instead of
        numeric indexing.

        """

        p = list(p) # copy

        if old_API:
            idx = self.parameters.index('cos_inclination')
            p[idx] = math.cos(p[idx])

        # used ordered names and values
        ref = dict(zip(self.parameters.names, p))

        # compactness ratio M/R_eq
        p += [gravradius(ref['mass']) / ref['radius']]

        p += [ref['s__super_radius'] - ref['s__omit_radius']]

        if ref['s__phase_shift'] > 0.0:
            p += [ref['s__phase_shift'] - 1.0]
        else:
            p += [ref['s__phase_shift']]

        temp = eval_cedeCentreCoords(-1.0*ref['s__omit_colatitude'],
                                     ref['s__super_colatitude'],
                                     -1.0*ref['s__omit_azimuth'])

        azi = temp[1]

        if azi < 0.0:
            azi += 2.0*math.pi

        p += [ref['s__omit_radius']/ref['s__super_radius'] \
              if ref['s__omit_radius'] <= ref['s__super_radius'] \
              else 2.0 - ref['s__super_radius']/ref['s__omit_radius']] # f

        p += [ref['s__super_radius'] if ref['s__omit_radius'] \
              <= ref['s__super_radius'] else ref['s__omit_radius']] # xi

        p += [temp[0]/(ref['s__super_radius'] + ref['s__omit_radius']) \
              if ref['s__omit_radius'] <= ref['s__super_radius'] \
              else (temp[0] - ref['s__omit_radius'] + ref['s__super_radius'])/(2.0*ref['s__super_radius'])] # kappa

        p += [azi/math.pi]

        return p
