""" Main module for NICER J0030 <- X-PSI v0.7.5 ST-S. """
from __future__ import print_function, division

import argparse

parser = argparse.ArgumentParser(
    description='''
    Main module for X-PSI ST-S modelling of NICER J0030 event data.

    You can run this module as a script and launch a sampler, optionally
    with a world of MPI processes.

    Alternate usage: mpiexec -n 4 python -m mpi4py %(prog)s [-h] @<config.ini> [--multinest] [--emcee]

    ''',
    fromfile_prefix_chars='@')

_prfx = 'Absolute or relative path to '
parser.add_argument('--matrix-path', type=str, help=_prfx + 'response matrix file.')
parser.add_argument('--event-path', type=str, help=_prfx + 'event list file.')
parser.add_argument('--arf-path', type=str, help=_prfx + 'ARF file.')
parser.add_argument('--rmf-path', type=str, help=_prfx + 'RMF file.')
parser.add_argument('--channels-path', type=str, help=_prfx + 'channel bounds file.')
parser.add_argument('--attenuation-path', type=str, help=_prfx + 'attenuation file.')
parser.add_argument('--atmosphere-path', type=str, help=_prfx + 'atmosphere file.')

parser.add_argument('--multinest', action='store_true',
                    help='Launch MultiNest sampler. Takes precedence.')
parser.add_argument('--emcee', action='store_true',
                    help='Launch emcee sampler.')


if __name__ == '__main__':
    args = parser.parse_args()
else:    args = parser.parse_args(['@/Users/linaissa/Documents/MASTER/Stage/research_stuff/NICER_analyses/PSR+J0030/STS_model/STS_module/config.ini'])  
#print(args)
#check if interactive input needed

if not args.matrix_path:
    args.matrix_path = raw_input('Specify the response matrix path: ')

if not args.event_path:
    event_path = raw_input('Specify the event file path: ')

if not args.arf_path:
    args.arf_path = raw_input('Specify the ARF file path: ')

if not args.rmf_path:
    args.rmf_path = raw_input('Specify the RMF file path: ')

if not args.channels_path:
    args.channels_path = raw_input('Specify the channel energy file path: ')

if not args.attenuation_path:
    args.attenuation_path = raw_input('Specify the attenuation file path: ')

if not args.atmosphere_path:
    args.atmosphere_path = raw_input('Specify the atmosphere file path: ')




import numpy as np
import math

import xpsi

print('Rank reporting: %d' % xpsi._rank)

# reqd lib for time.time()
import time
import sys
import os

from xpsi.global_imports import gravradius

from CustomInstrument import CustomInstrument
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal
from CustomPrior import CustomPrior
from CustomPhotosphere import CustomPhotosphere


try:
    counts = np.loadtxt(args.matrix_path, dtype=np.double)
    print('check 1')
except IOError:
    print('check 2')
    data = xpsi.Data.phase_bin__event_list(args.event_path,
                                           channels=np.arange(30, 300),
                                           phases=np.linspace(0.0, 1.0, 33),
                                           phase_column=0,
                                           channel_column=1,
                                           skiprows=3,
                                           dtype=np.double,
                                           first=0,
                                           last=269,
                                           exposure_time=1936864.0)

    np.savetxt(args.matrix_path, data.counts)
else:
    data = xpsi.Data(counts,
                     channels=np.arange(30, 300), #[25,300)
                     phases=np.linspace(0.0, 1.0, 33),
                     first=0,
                     last=269,
                     exposure_time=1936864.0) #1, 936, 864 s

NICER = CustomInstrument.from_SWG(bounds = dict(beta = (None, None)),
                                  values = {},
                                  ARF = args.arf_path,
                                  RMF = args.rmf_path,
                                  max_input = 700,
                                  min_input = 0,
                                  channel_edges = args.channels_path)

interstellar = CustomInterstellar.from_SWG(args.attenuation_path,
                                           bounds = dict(column_density = (0.0,5.0)))
signal = CustomSignal(data = data,
                      instrument = NICER,
                      interstellar = interstellar,
                      cache = True, #True for postprocessing
                      workspace_intervals = 1000,
                      epsrel = 1.0e-8,
                      epsilon = 1.0e-3,
                      sigmas = 10.0)

bounds = dict(mass = (1.0, 3.0), # Gaussian prior
              radius = (3.0*gravradius(1.0), 16.0),
              cos_inclination = (0.0,1.0)) # Gaussian prior

spacetime = xpsi.Spacetime(bounds, dict(frequency = 1.0/(4.87e-3),
                                        distance = 0.01)) # fixed dummy distance

bounds = dict(super_colatitude = (0.001, math.pi/2.0),
              super_radius = (0.001, math.pi/2.0 - 0.001),
              phase_shift = (-0.25, 0.75),
              super_temperature = (5.1, 6.8))

primary = xpsi.HotRegion(bounds=bounds,
                            values={},
                            symmetry=True,
                            omit=False,
                            cede=False,
                            concentric=False,
                            sqrt_num_cells=32,
                            min_sqrt_num_cells=16,
                            max_sqrt_num_cells=64,
                            num_leaves=32,
                            num_rays=512,
                            is_antiphased=False,
                            image_order_limit=1, # up to primary
                            prefix='p')

class derive_colatitude(xpsi.Derive):
    def __init__(self):
        pass

    def __call__(self, boundto, caller=None):
        global primary
        return math.pi - primary['super_colatitude']


class derive_radius(xpsi.Derive):
    def __init__(self):
        pass

    def __call__(self, boundto, caller=None):
        global primary
        return primary['super_radius']


class derive_phase(xpsi.Derive):
    def __init__(self):
        pass

    def __call__(self, boundto, caller=None):
        global primary
        return primary['phase_shift']


class derive_temperature(xpsi.Derive):
    def __init__(self):
        pass

    def __call__(self, boundto, caller=None):
        global primary
        return primary['super_temperature']

values = {'super_temperature': derive_temperature(),
          'super_colatitude': derive_colatitude(),
          'super_radius': derive_radius(),
          'phase_shift': derive_phase()}

bounds = dict(super_colatitude=None,  # declare fixed/derived variable
              super_radius=None,  # declare fixed/derived variable
              phase_shift=None,  # declare fixed/derived variable
              super_temperature=None)  # declare fixed/derived variable

secondary = xpsi.HotRegion(bounds=bounds,
                            values=values,
                            symmetry=True,
                            omit=False,
                            cede=False,
                            concentric=False,
                            sqrt_num_cells=32,
                            min_sqrt_num_cells=16,
                            max_sqrt_num_cells=64,
                            num_leaves=32,
                            num_rays=512,
                            is_antiphased=True,
                            image_order_limit=1,
                            prefix='s')

from xpsi import HotRegions

hot = HotRegions((primary, secondary))

photosphere = CustomPhotosphere(hot = hot, elsewhere = None,
                                values=dict(mode_frequency = spacetime['frequency']))
photosphere.hot_atmosphere = args.atmosphere_path

star = xpsi.Star(spacetime = spacetime, photospheres = photosphere)

prior = CustomPrior()

likelihood = xpsi.Likelihood(star = star, signals = signal,
                             num_energies = 128,
                             threads = 1,
                             externally_updated = True,
                             prior = prior)

#likelihood.prior = prior
"""
p = [1.4830284250798584,
    7.228831356180482,
    0.09528360613515485,
    0.30013977370083467,
    1.3245969196623892,
    0.2128495121295967,
    5.952986732404897,
    9.477506589767344,
    0.7170381731553449]
"""
p = [0.293045584170070850E+01,
    0.159949966990249184E+02,
    0.329436304996646778E+00,
    -0.878369880224431632E-01,
    0.125615907715870057E+01,
    0.892071499716681343E-01,
    0.607308071126313376E+01,
    0.101909578045078479E+02,
    0.891377814452705853E-03]





likelihood.externally_updated = False
likelihood.clear_cache()
likelihood(p, reinitialise=True)
#Lv = -1e90
#while Lv<-100000:
#    p = prior.inverse_sample()
#    likelihood.clear_cache()
#    Lv = likelihood(p, reinitialise=True)
#print (Lv, p)
#print (p)
#likelihood.check(None, [-2.99547340e+89], 1.0e-5, physical_points=[p])

if __name__ == '__main__': # sample from the posterior
    # transform relevant input information below to conmmand line arguments
    # and config file arguments

    wrapped_params = [0] * len(likelihood)
    wrapped_params[likelihood.index('p__phase_shift')] = 1

    runtime_params = {'resume': False,
                      'importance_nested_sampling': False,
                      'multimodal': False,
                      'n_clustering_params': None,
                      'outputfiles_basename': './run1_nlive1000_eff0.3_noCONST_noMM_noIS_tol-1',
                      'n_iter_before_update': 100, #100
                      'n_live_points': 1000,
                      'sampling_efficiency': 0.3,
                      'const_efficiency_mode': False,
                      'wrapped_params': wrapped_params,
                      'evidence_tolerance': 0.1,
                      'max_iter': -1,
                      'verbose': True}

    xpsi.Sample.nested(likelihood, prior, **runtime_params)


