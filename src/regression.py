#!/usr/bin/env python
"""
.. module:: regression.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v3.0 license Copyright (C)
       2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

# TODO move RMS to the BaseReg class

from math import sqrt, sin, cos, acos, hypot, atan2
import numpy as np

def sign(x):
    """ signum function """
    return  1 if x > 0 else -1 if x < 0 else 0

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
            pnts_act = self._pnts.copy()
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
            pnts_act = self._pnts.copy()
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

    def dist(self) ->np.ndarray:
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
            pnts_act = self._pnts.copy()
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

    def dist(self) ->np.ndarray:
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

class EllipseReg(BaseReg):
    """ class for ellipse regression
        :param pnts: array of coordinates (n,2)
    """
    def __init__(self, pnts:np.ndarray):
        """ """
        super().__init__(pnts)

    @staticmethod
    def par2geom(A, B, C, D, E, F):
        """ calculate geometrical parameters from A B C D E F"""
        ap = abs(-sqrt(abs(2 * (A * E**2 + C * D**2 - B * D * E  + (B**2 - 4 * A * C) * F) * ((A + C) + sqrt((A - C)**2 + B**2)))) / (B**2 - 4 * A * C))
        bp = abs(-sqrt(abs(2 * (A * E**2 + C * D**2 - B * D * E  + (B**2 - 4 * A * C) * F) * ((A + C) - sqrt((A - C)**2 + B**2)))) / (B**2 - 4 * A * C))
        x0 = (2 * C * D - B * E) / (B**2 - 4 * A * C)
        y0 = (2 * A * E - B * D) / (B**2 - 4 * A * C)
        phi = atan2(-B, C-A) / 2
        if ap < bp:
            ap, bp = bp, ap
            phi -= np.pi / 2
        while phi < 0:
            phi += 2 * np.pi
        while phi > np.pi:
            phi -= np.pi
        return np.array([x0, y0, ap, bp, phi])

    def lkn_reg(self, inds=None) ->np.ndarray:
        """ Calculate best fitting ellipse parameters
            Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
            :params inds: index array to filter points
        """
        if inds is None:
            pnts_act = self._pnts.copy()
        else:
            pnts_act = self._pnts[inds]
        x = pnts_act[:,0]
        y = pnts_act[:,1]
        mat = np.c_[x**2, x*y, y**2, x, y, np.ones_like(x)]
        S = mat.T @ mat
        eigvals, eigvecs = np.linalg.eig(S)
        eigvals_sorted_idx = np.argsort(eigvals)
        eigvals = eigvals[eigvals_sorted_idx]
        eigvecs = eigvecs[:, eigvals_sorted_idx]
        A, B, C, D, E, F = eigvecs[:, 0]
        # calculating ellipse geometric parameters
        self._params = self.par2geom(A, B, C, D, E, F)
        return self._params

    def pnt_ell_dist(self, pnt: np.ndarray) ->float:
        """ Calculate point ellipse distance from standard ellipse
            (center at origin main axis horizontal)
        """
        xp = abs(pnt[0])
        yp = abs(pnt[1])
        a = self._params[2]
        b = self._params[3]
        if xp > yp:
            a, b = b, a
            xp, yp = yp, xp
        l = b * b - a * a
        m = a * xp / l
        m2 = m * m
        n = b * yp / l
        n2 = n * n
        c = (m2 + n2 - 1.0) / 3.0
        c3 = c**3
        q = c3 + m2 * n2 * 2.0
        d = c3 + m2 * n2
        g = m + m * n2
        if d < 0.0:
            p = acos(q / c3) / 3.0
            s = cos(p)
            t = sin(p) * sqrt(3.0)
            rx = sqrt(-c * (s + t + 2.0) + m2)
            ry = sqrt(-c * (s - t + 2.0) + m2)
            co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) / 2.0
        else:
            h = 2.0 * m * n * sqrt(d)
            s = sign(q + h) * abs(q + h)**(1.0 / 3.0)
            u = sign(q - h) * abs(q - h)**(1.0 / 3.0)
            rx = -s - u - c * 4.0 + 2.0 * m2
            ry = (s - u) * sqrt(3.0)
            rm = sqrt(rx * rx + ry * ry)
            p = ry / sqrt(rm -rx)
            co = (p + 2.0 * g / rm - m) / 2.0
        si = sqrt(abs(1.0 - co * co))
        xe, ye = a * co, b * si
        return hypot(xe - xp, ye - yp)

    def dist(self) ->np.ndarray:
        """ distance from points to ellipse """
        phi = self._params[4]           # rotational angle
        rot = np.array([[cos(phi), -sin(phi)],
                        [sin(phi), cos(phi)]])
        # move to origin and rotate to horizontal
        loc_pnts = (self._pnts - self._params[:2]) @ rot
        distances = [self.pnt_ell_dist(loc_pnt) for loc_pnt in loc_pnts]
        return np.array(distances)

    def RMS(self) ->float:
        """ Calculate mean square error
        """
        return sqrt(np.sum(self.dist()**2) / self._pnts.shape[0])

    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 5

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
    # ellipse
    pnts = np.array(
            [[8.950, 1.450],
             [7.761, 1.885],
             [6.000, 1.500],
             [3.934, 0.354],
             [1.879, -1.379]])
    ee = EllipseReg(pnts)
    print(ee.lkn_reg())
    print(f"RMS: {ee.RMS()}")
