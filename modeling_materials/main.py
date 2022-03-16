#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Sebastien Guillot - IRAP
# Modified by: Lina Issa & Wilfried Mercier - IRAP

# Import section
from    __future__            import print_function, division

import xpsi
import math
import numpy                  as     np
from   xpsi                   import HotRegions
from   CustomData             import CustomData
from   CustomInstrument       import CustomInstrument
from   CustomSignal           import CustomSignal
from   CustomInterstellar     import CustomInterstellar
from   CustomPhotosphere      import CustomPhotosphere
from   CustomPrior            import CustomPrior
from   CustomBackground       import CustomBackground
from   xpsi.global_imports    import _c, _G, _dpr, gravradius, _csq, _km, _2pi
from   xpsi.ParameterSubspace import ParameterSubspace

print('Rank reporting: %d' % xpsi._rank)

##########################################################
#                     Importing data                     #
##########################################################

path         = '../data/NICER_J0030_PaulRay_fixed_evt_25to299__preprocessed.txt'

# Settings for when importing data
obs_settings = dict(counts        = np.loadtxt(path, dtype=np.double),
                    channels      = np.arange(25, 300),
                    phases        = np.linspace(0.0, 1.0, 33),
                    first         = 0, 
		    last          = 274,
                    exposure_time = 1936864.0)

#data = CustomData.from_SWG('../data/NICER_J0030_PaulRay_fixed_evt_25to299__preprocessed.txt', 1936864.0)
data         = xpsi.Data(**obs_settings)

#######################################################################
#                NICER INSTRUMENT MODEL WITH PARAMETER                #
#######################################################################

# Bounds for NICER instrument
bounds_NICER = dict(alpha = (0.5, 1.5), beta = (0.0, 1.0), gamma = (0.5, 1.5))

# Generate Instrument instance
NICER        = CustomInstrument.from_SWG(bounds        = bounds_NICER,
                                         values        = {},
                                         ARF           = '../model_data/ni_xrcall_onaxis_v1.02_arf.txt',
                                         RMF           = '../model_data/nicer_upd_d49_matrix.txt',
                                         ratio         = '../model_data/crab_ratio_SA80_d49.txt',
                                         max_input     = 700,
                                         min_input     = 0,
                                         channel_edges = '../model_data/nicer_upd_energy_bounds.txt')


####################################################
#                INTERSTELLAR MODEL                #
####################################################

# Generate Interstellar instance
interstellar = CustomInterstellar.from_SWG('../model_data/interstellar_phot_frac.txt',
                                           bounds = dict(column_density = (0.0, 5.0)))
     

##############################################
#                SIGNAL MODEL                #
##############################################

# Generate Signal instance
signal       = CustomSignal(data                = data,         # Loaded data
                            instrument          = NICER,        # Custom NICER instrument
                            interstellar        = interstellar, # Custom interstellar instrument
                            cache               = True,
                            workspace_intervals = 1000,
                            epsrel              = 1.0e-8,
                            epsilon             = 1.0e-3,
                            sigmas              = 10.0)


#################################################
#                SPACETIME MODEL                #
#################################################

# Bounds for parameters of spacetime
bounds_SPACE = dict(mass = (1.0, 3.0),                             # Mass in solmass
                    radius          = (3.0*gravradius(1.0), 16.0), # Equatorial radius in km
                    distance        = (0.05, 2.0),                 # Earth distance in kpc
                    cos_inclination = (0.0, math.cos(0.001)))      # Cos of earth inclination to rotation axis

# Generate spacetime instance
spacetime    = xpsi.Spacetime(bounds=bounds_SPACE, 
                              values=dict(frequency = 1.0/(4.87e-3)))


##################################################
#                HOT REGION MODEL                #
##################################################

# Bounds for parameters of the primary hot region
bounds_HOT1  = dict(super_colatitude  = (0.001, math.pi - 0.001),     # Colatitude in ?
                    super_radius      = (0.001, math.pi/2.0 - 0.001), # Radius in ?
                    phase_shift       = (0.0, 1.0),                   # defined relative to 0.35 cycles
                    super_temperature = (None, None))                 # Temperature in ?

# Generate primary hot region instance
primary      = xpsi.HotRegion(bounds             = bounds_HOT1,
                              values             = {},         # no initial values and no derived/fixed
                              symmetry           = True,
                              omit               = False,
                              cede               = False,
                              concentric         = False,
                              sqrt_num_cells     = 24,
                              min_sqrt_num_cells = 10,
                              max_sqrt_num_cells = 64,
                              do_fast            = False,
                              num_leaves         = 100,
                              num_rays           = 200,
                              is_secondary       = False,
                              prefix             = 'p'         # unique prefix needed because >1 instance
                             )
                           
# Bounds for parameters of the secondary hot region (None for no bounds)
bounds_HOT2  = dict(super_colatitude  = (None, None), 
                    super_radius      = (None, None),
                    phase_shift       = (0.0, 1.0),
                    super_temperature = (None, None),
                    omit_colatitude   = (0.0, math.pi), 
                    omit_radius       = (None, None),
                    omit_azimuth      = (None, None)) 

# Generate secondary hot region instance
# overlap of an omission region and a radiating super region
secondary    = xpsi.HotRegion(bounds             = bounds_HOT2,
                              values             = {},
                              symmetry           = True,
                              omit               = False,
                              cede               = False,
                              concentric         = False,
                              sqrt_num_cells     = 24,
                              min_sqrt_num_cells = 10,
                              max_sqrt_num_cells = 64,
                              num_leaves         = 100,
                              num_rays           = 200,
                              do_fast            = False,
                              is_secondary       = True,
                              prefix             = 's')

# Generate total hot region
hot          = xpsi.HotRegions((primary, secondary))


###################################################
#                PHOTOSPHERE MODEL                #
###################################################

photosphere  = CustomPhotosphere(hot       = hot, # Hot region model
                                 elsewhere = None,
                                 values=dict(mode_frequency = spacetime['frequency']))

# Link hot amosphere file to photosphere instance	
photosphere.hot_atmosphere = '../model_data/nsx_H_v171019.out'


############################################
#                STAR MODEL                #
############################################

# Generate star instance
star         = xpsi.Star(spacetime    = spacetime,  # Spacetime model
                         photospheres = photosphere # Photosphere model
                        )


##################################################
#                LIKELIHOOD MODEL                #
##################################################

# Generate likelihood instance
likelihood   = xpsi.Likelihood(star               = star,   # Star model
                               signals            = signal, # Signal model
                               num_energies       = 128,
                               threads            = 1,      # Not recomended to use more than one
                               externally_updated = True)

#############################################
#                PRIOR MODEL                #
#############################################


prior        = CustomPrior()

# Parameter vector to update subspace (not sure this is really relevant)

p = [1.4033703360094012,
     13.378462458584202,
     0.32897884439908337,
     math.cos(1.004349731136371),
     0.4542555093514883,
     2.1937752730930784,
     0.07916088420116879,
     6.106556223820221,
     0.4768294130316574,
     2.7162985247930496,
     0.32234225478780626,
     2.7463301464251777,
     -0.048326090505605386,
     0.2844169651751102,   
     6.1173049179880445,   
     1.0335682718716097,
     0.02227107198360202,
     0.8748566319738948,
     0.4604998629950954]



# Once prior is correctly initialised we pass it as prior for the likelihood
likelihood.prior = prior

# Check that likelihood and prior are ok
#likelihood.check(None, [-36316.354394388654], 1.0e-4,
#                 physical_points=[p])


####################################
#           CALLING XPSI           #
####################################

if __name__ == '__main__': 

     # No clue what this does... looks like initialising parameters
     wrapped_params                                     = [0]*len(likelihood)
#     wrapped_params[likelihood.index('phase_shift')] = 1
     wrapped_params[likelihood.index('s__phase_shift')] = 1

     runtime_params = {'resume'                     : False,
                       'importance_nested_sampling' : False,
                       'multimodal'                 : False,
                       'n_clustering_params'        : None,
                       'outputfiles_basename'       : './Run1',
                       'n_iter_before_update'       : 100,
                       'n_live_points'              : 1000,
                       'sampling_efficiency'        : 0.3,
                       'const_efficiency_mode'      : False,
                       'wrapped_params'             : wrapped_params, # That's the params defined just above
                       'evidence_tolerance'         : 0.1,
                       'max_iter'                   : -1,             # No number of max iteration ?
                       'verbose'                    : True            # Will print stuff on stdout
                      }
 
     xpsi.Sample.nested(likelihood, prior, **runtime_params)