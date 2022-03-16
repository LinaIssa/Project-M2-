from __future__ import print_function, division

import numpy as np
import math

import xpsi

class CustomPhotosphere(xpsi.Photosphere):
    """ A photosphere extension to preload the numerical atmosphere NSX.

    Fully-ionized hydrogen, v200802 (W.C.G. Ho).

    """

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        NSX = np.loadtxt(path, dtype=np.double)
        logT = np.zeros(35)
        logg = np.zeros(11) # 14
        mu = np.zeros(67)
        logE = np.zeros(166)

        reorder_buf = np.zeros((35,11,67,166)) # np.zeros((35,14,67,166))

        index = 0
        for i in range(reorder_buf.shape[0]):
            for j in range(reorder_buf.shape[1]):
                for k in range(reorder_buf.shape[3]):
                   for l in range(reorder_buf.shape[2]):
                        logT[i] = NSX[index,3]
                        logg[j] = NSX[index,4]
                        logE[k] = NSX[index,0]
                        mu[reorder_buf.shape[2] - l - 1] = NSX[index,1]
                        reorder_buf[i,j,reorder_buf.shape[2] - l - 1,k] = 10.0**(NSX[index,2])
                        index += 1

        buf = np.zeros(np.prod(reorder_buf.shape))

        bufdex = 0
        for i in range(reorder_buf.shape[0]):
            for j in range(reorder_buf.shape[1]):
                for k in range(reorder_buf.shape[2]):
                   for l in range(reorder_buf.shape[3]):
                        buf[bufdex] = reorder_buf[i,j,k,l]; bufdex += 1

        self._hot_atmosphere = (logT, logg, mu, logE, buf)

    @property
    def global_variables(self):
        """ This method is needed if we also want to invoke the image-plane signal simulator.

        The extension module compiled is surface_radiation_field/archive/local_variables/two_spots.pyx,
        which replaces the contents of surface_radiation_field/local_variables.pyx.

        """
        return np.array([self['p__super_colatitude'],
                          self['p__phase_shift'] * 2.0 * math.pi,
                          self['p__super_radius'],
                          self['p__super_temperature'],
                          self['s__super_colatitude'],
                          (self['s__phase_shift'] + 0.5) * 2.0 * math.pi,
                          self['s__super_radius'],
                          self['s__super_temperature']])
