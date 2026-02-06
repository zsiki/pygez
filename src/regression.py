#!/usr/bin/env python
"""
.. module:: regression.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v3.0 license Copyright (C)
       2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

from math import sqrt, sin, cos, acos, hypot
import numpy as np
import shapely
from scipy.optimize import least_squares

def sign(x):
    """ signum function """
    return  1 if x > 0 else -1 if x < 0 else 0

class BaseReg:
    """ Base class for regressions
        :param pnts: array of coordinates (n,2) or (n,3)
    """
    def __init__(self, pnts:np.ndarray):
        """ """
        self._pnts = pnts.copy()
        self._params = None     # actual parameters of geometry

    @property
    def params(self) -> np.ndarray:
        """ Return parameters """
        return self._params

    @property
    def dim(self) ->int:
        """ Return dimension of points 2/3 """
        if self._pnts is None:
            return 0
        return self._pnts.shape[1]

    @property
    def nump(self) ->int:
        """ Return number of points """
        if self._pnts is None:
            return 0
        return self._pnts.shape[0]

    def get_pnts_by_index(self, ind) ->np.ndarray:
        """ Return subset of points """
        return self._pnts[ind]

    def dist(self):
        """ dummy method it has to be implemented in inherited classes
        """
        return None

    def RMS(self) ->float:
        """ Calculate mean square error
        """
        return sqrt(np.mean(self.dist()**2))

class LinearReg(BaseReg):
    """ class for linear regressions (line, plane)
    """
    def __init__(self, pnts: np.ndarray):
        """ """
        super().__init__(pnts)

    def lkn_reg(self, inds=None, limits=False) ->np.ndarray:
        """ Calculate best fitting linear function
            2D line or 3D plane

            :param inds: point indices to use, all used if None
            :param limits: add limits to parameters
            :returns: a, b, c, [d], e0, n0, e1, n1 or convex hull

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
        if limits:
            if self.dim == 2:
                # find start & end point if 2D
                d = pnts_act.dot(norm) + pure_term  # distance from line
                en_proj = pnts_act - np.c_[norm[0]*d, norm[1]*d]
                t = (en_proj - pnts_mean) / np.array([norm[1], -norm[0]])
                min_ind = np.argmin(t[:,0])
                max_ind = np.argmax(t[:,0])
                self._params = np.r_[norm, pure_term, pnts_act[min_ind], pnts_act[max_ind]]
            else:
                # set up 3D coordinate system aligned to the plane
                if abs(norm[0]) < 0.9:  # non-parallel vector to norm
                    t = np.array([1, 0, 0], float)
                else:
                    t = np.array([0, 1, 0], float)
                u = np.cross(norm, t)   # axis directions in plane
                u /= np.linalg.norm(u)
                v = np.cross(norm, u)
                v /= np.linalg.norm(v)
                origin = np.zeros(3)
                idx = np.argmax(np.abs(norm))
                origin[idx] = -pure_term / norm[idx]
                # transform points to plane coordinate system
                rel = pnts_act - origin
                U = rel @ u
                V = rel @ v
                # get convex poly in plane using U and V
                convex_hull = shapely.convex_hull(shapely.geometry.MultiPoint(np.c_[U, V]))
                # convex_hull to numpy array
                hull_coords = np.array(convex_hull.exterior.coords)
                hull_coords = np.c_[hull_coords, np.zeros(hull_coords.shape[0])]

                # transform back to original CS
                convex_hull_3d = origin + hull_coords[:,0,None] * u + \
                                          hull_coords[:,1,None] * v + \
                                          hull_coords[:,2,None] * norm
                # add convex-hull_3d to _params
                self._params = np.r_[norm, pure_term,
                                     np.reshape(convex_hull_3d, (-1))]
        return self._params.copy()

    def dist(self) ->np.ndarray:
        """ Calculate distance from the line or plane """
        n = self.dim   # dimension
        return np.abs(self._pnts.dot(self._params[:n]) + self._params[n])

    @property
    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return self.dim

class CircleReg(BaseReg):
    """ class for circle regression
        :param pnts: array of coordinates (n,2) or (n,3)
    """
    def __init__(self, pnts:np.ndarray):
        """ """
        super().__init__(pnts)

    def lkn_reg(self, inds=None, limits=False) ->np.ndarray:
        """ Calculate best fitting circle parameters

            :param inds: index filter to points
            :param limits: dummy
            :returns: x0, y0, r
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
        if limits:
            pmin = self._params[0:2] - self._params[2]
            pmax = self._params[0:2] + self._params[2]
            self._params = np.r_[self._params, pmin, pmax]
        return self._params.copy()

    def dist(self) ->np.ndarray:
        """Calculate distance from circle """
        return np.sqrt((self._pnts[:,0] - self._params[0])**2 +
                       (self._pnts[:,1] - self._params[1])**2) - self._params[2]

    @property
    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 3

class SphereReg(BaseReg):
    """ Calculate best fitting sphere parameters """
    def __init__(self, pnts:np.ndarray):
        """ """
        super().__init__(pnts)

    def lkn_reg(self, inds=None, limits=False) ->np.ndarray:
        """
            calculate best fitting sphere (LSM) on points
            :param inds: index filter to points
            :param limits: dummy
            :returns: x0, y0, z0, R, xmin, ymin, zmin, xmax, ymax, zmax as a numpy array
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
        if limits:
            p0 = self._params[0:3] - self._params[3]
            p1 = self._params[0:3] + self._params[3]
            self._params = np.r_[self._params, p0, p1]
        return self._params.copy()

    def dist(self) ->np.ndarray:
        """Calculate distance from sphere """
        return np.sqrt((self._pnts[:,0] - self._params[0])**2 +
                       (self._pnts[:,1] - self._params[1])**2 +
                       (self._pnts[:,2] - self._params[2])**2) - self._params[3]

    @property
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
    def par2geom(a, b, c, d, e, f, eps=1e-12):
        """ Elliptical polynom parameters to geometrical params
        """
        denom = b**2 - 4*a*c
        if abs(denom) < eps:
            # points are collinear or fit failed
            raise ValueError("Degenerate conic (denominator ~ 0). Points may be collinear or fit failed.")

        x0 = (2 * c * d - b * e) / denom
        y0 = (2 * a * e - b * d) / denom
        # Translate to center
        F = f + a*x0**2 + b*x0*y0 + c*y0**2 + d*x0 + e*y0
        if -F < 0:
            # flipping sings
            a, b, c, d, e, f = (-a, -b, -c, -d, -e, -f)
            F = -F

        A = np.array([[a, b/2.], [b/2., c]])
        evals, evecs = np.linalg.eigh(A)

        # Clamp tiny negative eigenvalues (numerical noise) to zero
        evals_clamped = np.where(evals < 0, np.where(np.abs(evals) < 1e-12, 0.0, evals), evals)
        # If any eigenvalue is <= 0 (not positive definite), warn / fail
        if np.any(evals_clamped <= 0):
        #    print("This may not be an ellipse (or fit numerics are poor).")
            raise ValueError("This may not be an ellipse (or fit numerics are poor).")

        axes_sq = -F / (evals_clamped + eps)
        axes_sq = np.where(axes_sq < 0, np.where(np.abs(axes_sq) < 1e-8, 0.0, axes_sq), axes_sq)
        if np.any(axes_sq < 0):
            # Computed negative squared axis length — fit failed or points are not elliptical
            raise ValueError("Computed negative squared axis length — fit failed or points are not elliptical.")
        axis_lengths = np.sqrt(axes_sq)
        order = np.argsort(-axis_lengths)  # indices that sort descending
        axis_a, axis_b = axis_lengths[order[0]], axis_lengths[order[1]]
        # rotation: angle of the eigenvector corresponding to axis_a
        v_rot = evecs[:, order[0]]
        theta = np.arctan2(v_rot[1], v_rot[0])
        # move theta to 0-pi
        while theta < 2 * np.pi:
            theta += 2 * np.pi
        while theta > np.pi:
            theta -= np.pi

        return np.array([x0, y0, axis_a, axis_b, theta])

    def lkn_reg(self, inds=None, limits=False) ->np.ndarray:
        """ Calculate best fitting ellipse parameters
            Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
            :param inds: index array to filter points
            :param limits: dummy
            :returns: x0, y0, a, b, phi if limits true + MBR (four points)
        """
        if inds is None:
            pnts_act = self._pnts.copy()
        else:
            pnts_act = self._pnts[inds]
        pnts_mean = np.mean(pnts_act, axis=0)   # weight point
        pnts_act -= pnts_mean     # origin to weight point
        x = pnts_act[:,0]
        y = pnts_act[:,1]
        mat = np.c_[x**2, x*y, y**2, x, y, np.ones_like(x)]
        S = mat.T @ mat
        # Constraint matrix (enforces conic to be ellipse)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        # Solve generalized eigenvalue problem
        E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
        # Find eigenvector V corresponding to positive eigenvalue
        v = V[:, np.argmax(E.real)]
        a, b, c, d, e, f = v.real

        # calculating ellipse geometric parameters
        self._params = self.par2geom(a, b, c, d, e, f)
        # move origin back
        self._params[0] += pnts_mean[0]
        self._params[1] += pnts_mean[1]
        if limits:
            a, b = self._params[3:5]
            w = np.array([sin(self._params[4]), -cos(self._params[4])])
            p1 = self._params[0:2] + np.dot(np.array([a, b]), w)
            p2 = self._params[0:2] + np.dot(np.array([a, -b]), w)
            p3 = self._params[0:2] + np.dot(np.array([-a, -b]), w)
            p4 = self._params[0:2] + np.dot(np.array([-a, -b]), w)
            self._params = np.r_[self._params, p1, p2, p3, p4]
        return self._params.copy()

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

    def old_dist(self) ->np.ndarray:
        """ distance from points to ellipse """
        phi = self._params[4]           # rotational angle
        rot = np.array([[cos(phi), -sin(phi)],
                        [sin(phi), cos(phi)]])
        # move to origin and rotate to horizontal
        loc_pnts = (self._pnts - self._params[:2]) @ rot
        distances = [self.pnt_ell_dist(loc_pnt) for loc_pnt in loc_pnts]
        return np.array(distances)

    def dist(self, signed=False):
        """
            Approximate distance from points to an ellipse (geometric form).
            :param signed : return signed distance (negative = inside)
            :returns: approximate distance to ellipse
        """
        x0, y0 = self._params[:2]
        a, b = self._params[2:4]
        theta = self._params[4]
        # Translate points
        X = self._pnts[:, 0] - x0
        Y = self._pnts[:, 1] - y0
        c = np.cos(theta)
        s = np.sin(theta)
        # Rotate into ellipse frame
        xp =  c * X + s * Y
        yp = -s * X + c * Y
        # Implicit function
        f = (xp**2) / a**2 + (yp**2) / b**2 - 1.0
        # Gradient in ellipse frame
        fxp = 2 * xp / a**2
        fyp = 2 * yp / b**2
        # Rotate gradient back to world frame
        fx =  c * fxp - s * fyp
        fy =  s * fxp + c * fyp
        grad_norm = np.sqrt(fx**2 + fy**2)
        grad_norm = np.maximum(grad_norm, 1e-12)
        dist = f / grad_norm if signed else np.abs(f) / grad_norm
        return dist

    @property
    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 5

class Line3dReg(BaseReg):
    """ class for 3D line regression
        :param pnts: array of coordinates (n,3)
    """
    def __init__(self, pnts:np.ndarray):
        """ """
        super().__init__(pnts)

    def lkn_reg(self, inds=None, limits=False) ->np.ndarray:
        """ Calculate best fitting line parameters
            x = x0 + a * t; y = y0 + b * t; z = z0 + c * t
            :params inds: index array to filter points
            :returns: array of x0, y0, z0, a, b, c (a,b,c) vector normalized
        """
        if inds is None:
            pnts_act = self._pnts.copy()
        else:
            pnts_act = self._pnts[inds]
        centroid = pnts_act.mean(axis=0)  # weight point
        centered = pnts_act - centroid    # move weight point to origin
        if pnts_act.shape[0] == 2:
            direction = pnts_act[1] - pnts_act[0]
        else:
            cov = np.cov(centered.T)        # covariance matrix
            eig_vals, eig_vecs = np.linalg.eig(cov)
            direction = eig_vecs[:, np.argmax(eig_vals)]  # line direction
        direction /= np.linalg.norm(direction)  # normalize direction
        self._params = np.r_[centroid, direction]
        if limits:
            # project points to line to get limits
            diff = pnts_act - centroid  # vector from the point on line
            # projection scalar for points
            t = np.dot(diff, direction)
            p0 = centroid + np.min(t) * direction
            p1 = centroid + np.max(t) * direction
            self._params = np.r_[centroid, direction, p0, p1]
        return self._params.copy()

    def dist(self) ->np.ndarray:
        """Calculate distance from line """
        return np.linalg.norm(np.cross((self._pnts - self._params[:3]), self._params[3:6]), axis=1)

    @property
    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 2

def cyl_dist(params, act) -> np.ndarray:
    """ Calculate distance from the cylinder
        :param params: ellipse parameters
        :param act: points
    """
    p0 = params[:3]
    v = params[3:6]
    r = params[6]

    diff = act - p0  # vector from p0 to each point
    # project diff onto v
    proj_len = diff @ v
    proj = np.outer(proj_len, v)

    # perpendicular distances from axis
    perp = diff - proj
    d = np.linalg.norm(perp, axis=1)
    # residuals, distances from surface
    return d - r

class CylinderReg(BaseReg):
    """ class for cylinder regression, parameters are 
        x0, 0, z0 a point on the axis
        vx, vy, vz direction of the axis (normalised)
        r radius of the cílinder

        :param pnts: array of coordinates (n,3)
    """
    def __init__(self, pnts:np.ndarray, params0=None,
                 ftol=2.3e-16, gtol=2.3e-16, xtol=2.3e-16, loss='linear'):
        """ """
        super().__init__(pnts)
        self._ftol = ftol
        self._gtol = gtol
        self._xtol = xtol
        self._loss = loss
        self._params0 = params0
        self._params = None

    def dist(self, act=None) -> np.ndarray:
        """ Calculate distance from the cylinder """
        if act is None:
            dd = cyl_dist(self._params, self._pnts)
        else:
            dd = cyl_dist(self._params, act)
        return dd

    @property
    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 6

    def lkn_reg(self, inds=None, limits=False) ->np.ndarray:
        """ Calculate best fitting cílinder parameters
            :params inds: index array to filter points
            :returns: array of x0, y0, z0, a, b, c, r, x1, y1, z1 (a,b,c) vector normalized and limit points on the axis if limits is True
        """
        if inds is None:
            pnts_act = self._pnts.copy()
        else:
            pnts_act = self._pnts[inds]
        if self._params0 is None:
            centroid = pnts_act.mean(axis=0)  # weight point
            # PCA for approximate axis direction
            _, _, vh = np.linalg.svd(pnts_act - centroid)
            axis_dir = vh[0]  # first principal component
            # Approximate radius
            v = axis_dir / np.linalg.norm(axis_dir)
            proj_len = (pnts_act - centroid) @ v
            proj = np.outer(proj_len, v)
            perp = pnts_act - centroid - proj
            radius_guess = np.mean(np.linalg.norm(perp, axis=1))

            params0 = np.hstack([centroid, axis_dir, radius_guess])
        else:
            params0 = self._params0
        # Least-squares optimization
        res = least_squares(cyl_dist, params0, args=(pnts_act,),
                            ftol=self._ftol, gtol=self._gtol, xtol=self._xtol,
                            loss=self._loss)
        # normalize direction
        res.x[3:6] = res.x[3:6] / np.linalg.norm(res.x[3:6])
        # save params even in unsuccessful case
        self._params = res.x
        if not res.success:
            raise ValueError(res.message)
        if limits:
            # find min/max points on axis
            t = np.dot(pnts_act - res.x[0:3], res.x[3:6])
            p_min = res.x[0:3] + np.min(t) * res.x[3:6]
            p_max = res.x[0:3] + np.max(t) * res.x[3:6]
            res.x[0:3] = p_min
            self._params = np.r_[res.x, p_max]
        return self._params.copy()

def cone_dist(params, act) -> np.ndarray:
    """ distances from cone """
    p0 = params[0:3]
    v = params[3:6]
    alpha = params[6]
    # Vector from apex to each point
    w = act - p0
    # Projection onto axis
    h = w @ v  # signed height along axis
    u = w - np.outer(h, v)
    r = np.linalg.norm(u, axis=1)
    # Expected cone radius at each h
    return r - np.abs(h) * np.tan(alpha)

class ConeReg(BaseReg):
    """ class for cone regression, parameters are 
        x0, 0, z0 a apex of cone
        vx, vy, vz direction of the axis (normalised)
        alpha half angle of cone

        :param pnts: array of coordinates (n,3)
    """
    def __init__(self, pnts:np.ndarray, params0=None, ftol=2.3e-16,
                 gtol=2.3e-16, xtol=2.3e-16, loss='linear'):
        """ """
        super().__init__(pnts)
        self._ftol = ftol
        self._gtol = gtol
        self._xtol = xtol
        self._loss = loss
        self._params0 = params0
        self._params = None

    def dist(self) -> np.ndarray:
        """ Calculate distance from the cone """
        return cone_dist(self._params, self._pnts)

    def lkn_reg(self, inds=None, limits=False) ->np.ndarray:
        """ Calculate best fitting cone parameters
            :params inds: index array to filter points
            :returns: array of x0, y0, z0, a, b, c, r (a,b,c) vector normalized
        """
        if inds is None:
            pnts_act = self._pnts.copy()
        else:
            pnts_act = self._pnts[inds]
        if self._params0 is None:
            centroid = np.mean(pnts_act, axis=0)
            cov = np.cov((pnts_act - centroid).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by descending eigenvalue (principal direction first)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            # TODO
            # 1st set up a plane using first two eingenvectors
            #normal = np.cross(eigenvectors[0], eigenvectors[1])
            #normal = normal / np.linalg.norm(normal)
            #d = -np.dot(centroid, normal)
            #plane = np.c_[centroid, d]
            # 2nd find points close to the plane
            #distance = np.dot(plane, np.c_[pnts_act, np.ones(pnts_act.shape[0])])
            #plane_pnts = pnts_act[distance < 0.1]
            # 3rd project points and the 1st eigenvector to the plane

            # 4th RANSAC fit 2D line
            # find the intersection of the eigenvector and the 2D line for apex
            # find the angle of eigenvector and 2D line for alpha
            # Rough initial guess using PCA
            _, _, vh = np.linalg.svd(pnts_act - centroid)
            v0 = vh[0]  # first principal component

            # Initial apex and half-angle estimate
            p0_guess = centroid - 0.5 * v0
            diff = pnts_act - p0_guess
            h = diff @ v0
            proj = np.outer(h, v0)
            r = np.linalg.norm(diff - proj, axis=1)

            alpha_guess = np.arctan(np.mean(r) / np.mean(np.abs(h)))

            params0 = np.hstack([p0_guess, v0, alpha_guess])
        else:
            params0 = self._params0
        # Optimize
        res = least_squares(cone_dist, params0, args=(pnts_act,),
                            ftol=self._ftol, gtol=self._gtol, xtol=self._xtol,
                            loss=self._loss)
        if not res.success:
            raise ValueError("Cone fitting failed")
        self._params = res.x
        # normalize direction
        self._params[3:6] = np.linalg.norm(self._params[3:6])
        if limits:
            # find min/max points on axis
            t = np.dot(pnts_act - res.x[0:3], res.x[3:6])
            p_min = res.x[0:3] - np.min(t) * res.x[3:6]
            p_max = res.x[0:3] - np.max(t) * res.x[3:6]
            self._params = np.r_[res.x, p_min, p_max]

        return self._params.copy()

    @property
    def min_n(self) ->int:
        """ return minimal number of points to define geometry """
        return 7
