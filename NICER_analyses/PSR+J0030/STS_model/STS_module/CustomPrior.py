from __future__ import print_function, division

import numpy as np
import math
from scipy.stats import truncnorm

import xpsi
from xpsi.global_imports import _G, _csq, _km, _2pi, gravradius, _dpr
from xpsi import Parameter

from xpsi.PostProcessing import fix_random_seed
xpsi.PostProcessing.set_random_seed(0) # prevent noise during prior generation

from scipy.interpolate import Akima1DInterpolator

class CustomPrior(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Source: PSR J0437
    Model variant: ST-S
        Two single-temperature, simply-connected circular hot regions with
        shared parameters.

    Parameter vector: (print the likelihood object)

    * p[0] = (rotationally deformed) gravitational mass (solar masses)
    * p[1] = coordinate equatorial radius (km)
    * p[2] = cos(inclination of Earth to rotational axis)
    * p[3] = primary cap phase shift (cycles); (alias for initial azimuth, periodic)
    * p[4] = primary centre colatitude (radians)
    * p[5] = primary angular radius (radians)
    * p[6] = primary log10(comoving NSX FIH effective temperature [K])
    * p[7] = signal scaling (effective area uncertainty dominates distance)
    * p[8] = hydrogen column density (10^20 cm^-2)

    """

    __derived_names__ = ['compactness', ]

    @fix_random_seed
    def __init__(self):
        # Generating alpha and D prior distributions
        uniform_sampler1 = np.random.uniform(0, 1, 100000)
        uniform_sampler2 = np.random.uniform(0, 1, 100000)
        self._alpha = truncnorm.ppf(uniform_sampler1, -5.0, 5.0, loc=1.0, scale=0.1)
        self._distance = truncnorm.ppf(uniform_sampler2, -5.0, 5.0, loc=0.325, scale=0.009)

        # Generating beta and its cdf from alpha and D prior distibutions
        self._beta_created = self._alpha/np.power(self._distance, 2)
        self._ordered_beta = np.unique(self._beta_created)
        # https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
        self._beta_cdf = np.array(range(len(self._ordered_beta)))/float(len(self._ordered_beta))

        # Setting up interpolator for beta inverse sampling
        self.interpolator = Akima1DInterpolator(self._beta_cdf, self._ordered_beta)
        self.interpolator.extrapolate = True

	super(CustomPrior, self).__init__()

    def __call__(self, p = None):
        """ Evaluate distribution at ``p``.

        :param list p: Model parameter values.

        :returns: Logarithm of the distribution evaluated at ``p``.

        """
        temp = super(CustomPrior, self).__call__(p)
        if not np.isfinite(temp):
            return temp

        # based on contemporary EOS theory
        if not self.parameters['radius'] <= 16.0:
            return -np.inf

        ref = self.parameters.star.spacetime # shortcut

        # limit polar radius tobe outside the Schwarzschild photon sphere
        R_p = 1.0 + ref.epsilon * (-0.788 + 1.030 * ref.zeta)
        if R_p < 1.5 / ref.R_r_s:
            return -np.inf

        mu = math.sqrt(-1.0 / (3.0 * ref.epsilon * (-0.788 + 1.030 * ref.zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # only for high spin frequencies
        if mu < 1.0:
            return -np.inf

        # check effective gravity at pole (where it is maximum) and
        # at equator (where it is minimum) are in NSX limits
        grav = xpsi.surface_radiation_field.effective_gravity(np.array([1.0, 0.0]),
                                                               np.array([ref.R] * 2 ),
                                                               np.array([ref.zeta] * 2),
                                                               np.array([ref.epsilon] * 2))
        for g in grav:
            if not 13.7 <= g <= 15.0:
                return -np.inf

        return 0.0

    def inverse_sample(self, hypercube=None):
        """ Draw sample uniformly from the distribution via inverse sampling. """

        to_cache = self.parameters.vector

        if hypercube is None:
            hypercube = np.random.rand(len(self))

        # the base method is useful, so to avoid writing that code again:
        _ = super(CustomPrior, self).inverse_sample(hypercube)

        ref = self.parameters # redefine shortcut

        # Inverse sampling for beta by interpolating over self._beta_cdf
        idx = ref.index('beta')
        ref['beta'] = float(self.interpolator(hypercube[idx]))


        # flat priors in cosine of hot region centre colatitudes (isotropy)
        # support modified by no-overlap rejection condition
        idx = ref.index('p__super_colatitude')
        a, b = ref.get_param('p__super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

        # restore proper cache
        for parameter, cache in zip(self.parameters, to_cache):
            parameter.cached = cache

        # it is important that we return the desired vector because it is
        # automatically written to disk by MultiNest and only by MultiNest
        return self.parameters.vector

    def transform(self, p,**kwargs):
        """ A transformation for post-processing. """

        p = list(p) # copy

        # used ordered names and values
        ref = dict(zip(self.parameters.names, p))

        # compactness ratio M/R_eq
        p += [gravradius(ref['mass']) / ref['radius']]

        return p
