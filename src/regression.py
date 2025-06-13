#!/usr/bin/env python
"""
.. module:: regression.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v3.0 license Copyright (C)
       2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

from math import sqrt
import numpy as np

class BaseReg:
    """ Base class for regressions
        :param pnts: array of coordinates (n,2) or (n,3)
    """
    def __init__(self, pnts:np.ndarray):
        """ """
        self._pnts = pnts
        self._params = None

    def get_dim(self) ->int:
        """ Return dimension of points 2/3 """
        return self._pnts.shape[1]

    def get_n(self) ->int:
        """ Return number of points """
        return self._pnts.shape[0]

    def get_pnts_by_index(self, ind):
        """ Return subset of points """
        return self._pnts[ind]

    def set_pnts(self, pnts:np.ndarray):
        """ update coordinates """
        self._pnts = pnts

class LinearReg(BaseReg):
    """ class for linear regressions (line, plane)
    """
    def __init__(self, pnts: np.ndarray):
        """ """
        super().__init__(pnts)

    def lkn_reg(self, inds=None) ->np.ndarray:
        """ Calculate best fitting linear function
            2D line or 3D plane

            inds: point idices to use
        """
        if inds is None:
            pnts_act = self._pnts
        else:
            pnts_act = self._pnts[inds]
        # move origin to weight point (pure term = 0)
        pnts_mean = np.mean(pnts_act, axis=0)
        A = pnts_act - pnts_mean
        AxA = A.T @ A
        eig, eig_vec = np.linalg.eig(AxA)
        # find smallest eigenvalue
        norm = eig_vec[:,np.argmin(np.abs(eig))]
        # move back from weight point
        pure_term = - np.dot(norm, pnts_mean)
        self._params = np.r_[norm, pure_term]
        return self._params

    def dist(self) ->np.ndarray:
        """ Calculate distance from the line or plane """
        n = self._pnts.shape[1]   # dimension
        return self._pnts.dot(self._params[:n]) + self._params[n]

    def RMS(self) ->float:
        """ Calculate mean square error
        """
        return sqrt(np.sum(self.dist()**2) / self._pnts.shape[0])

    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return self.get_dim()

class CircleReg(BaseReg):
    """ class for circle regression
        :param pnts: array of coordinates (n,2) or (n,3)
    """
    def __init__(self, pnts:np.ndarray):
        """ """
        super().__init__(pnts)

    def lkn_reg(self, inds=None) ->np.ndarray:
        """ Calculate best fitting circle parameters
        """
        if inds is None:
            pnts_act = self._pnts
        else:
            pnts_act = self._pnts[inds]
        pnts_mean = np.mean(pnts_act, axis=0)   # weight point
        east = pnts_act[:,0] - pnts_mean[0]     # origin to weight point
        north = pnts_act[:,1] - pnts_mean[1]
        A = np.c_[east, north, np.ones_like(east)]
        # pure term
        b = -(east * east + north * north)
        # solution for a1, a2, a3
        Q = np.linalg.inv(A.T @ A)
        par = Q @ (A.T @ b)
        # calculating the original unknowns
        east_0 = -0.5 * par[0]
        north_0 = -0.5 * par[1]
        r = sqrt(east_0**2 + north_0**2 - par[2])
        east_0 += pnts_mean[0]   # shift back weight point
        north_0 += pnts_mean[1]
        self._params = np.array([east_0, north_0, r])
        return self._params

    def dist(self):
        """Calculate distance from circle """
        return np.sqrt((self._pnts[:,0] - self._params[0])**2 +
                       (self._pnts[:,1] - self._params[1])**2) - self._params[2]

    def RMS(self) ->float:
        """ Calculate mean square error
        """
        return sqrt(np.sum(self.dist()**2) / self._pnts.shape[0])

    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 3

class SphereReg(BaseReg):
    """ Calculate best fitting sphere parameters """
    def __init__(self, pnts:np.ndarray):
        """ """
        super().__init__(pnts)

    def lkn_reg(self, inds=None) ->np.ndarray:
        """
            calculate best fitting sphere (LSM) on points
            :returns: x0, y0, z0, R as a numpy array
        """
        if inds is None:
            pnts_act = self._pnts
        else:
            pnts_act = self._pnts[inds]
        pnts_mean = np.mean(pnts_act, axis=0)   # weight point
        pnts_act -= pnts_mean     # origin to weight point
        n = pnts_act.shape[0]
        A = np.c_[pnts_act, np.full(n, 1, 'float64')]
        b = -np.square(pnts_act[:,0]) - np.square(pnts_act[:,1]) - np.square(pnts_act[:,2])
        res = np.linalg.lstsq(A, b, rcond=None)[0]
        self._params = np.array([-0.5 * res[0] + pnts_mean[0],
                                 -0.5 * res[1] + pnts_mean[1],
                                 -0.5 * res[2] + pnts_mean[2],
              sqrt((res[0]**2 + res[1]**2 + res[2]**2) / 4 - res[3])])
        return self._params

    def dist(self):
        """Calculate distance from sphere """
        return np.sqrt((self._pnts[:,0] - self._params[0])**2 +
                       (self._pnts[:,1] - self._params[1])**2 +
                       (self._pnts[:,2] - self._params[2])**2) - self._params[3]

    def RMS(self) ->float:
        """ Calculate mean square error
        """
        return sqrt(np.sum(self.dist()**2) / self._pnts.shape[0])

    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 4

if __name__ == "__main__":
    east = np.array([1, 5, 3])
    north = np.array([-3, 7, 9])
    elev = np.array([9, -2, 2])
    pp = LinearReg(np.c_[east, north, elev])
    print(pp.lkn_reg())
    print(f"RMS: {pp.RMS()}")
    cc = CircleReg(np.c_[east, north])
    print(cc.lkn_reg())
    print(f"RMS: {cc.RMS()}")
