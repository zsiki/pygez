#!/usr/bin/env python
"""
.. module:: ransac.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v3.0 license Copyright (C)
       2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

from random import shuffle
from math import log
import numpy as np

class Ransac:
    """ Class to solve generic ransac filtering
        :param reg_obj: object to handle geometry
        :param tolerance: tolerance distance from geometry
    """
    def __init__(self, reg_obj):
        """ initialize """
        self._reg_obj = reg_obj

    @property
    def reg_obj(self):
        """ return gemometry object """
        return self._reg_obj

    def ransac_filter(self, tolerance:float=0.03, iterations:int=None,
                      p:float=0.99, w:float=0.51) ->np.ndarray:
        """ Apply RANSAC filter 
            :param iteration: iteration number for RANSAC
            :params p: probability for result
            :params w: percentage of inlier points
            :returns: array of filtered points

            p and w are used if iteration is not given (None)
        """
        n = self.reg_obj.nump
        n_geom = self.reg_obj.min_n
        if iterations is None:
            if p is None or w is None:
                iterations = 10 * n
            else:
                iterations = int(log(1 - p) / log(1 - w**n_geom) + 1) 
        indices = list(range(n))
        best = 0
        best_enz = None
        for _ in range(iterations):
            shuffle(indices)
            ind_n = indices[:n_geom]
            try:
                self.reg_obj.lkn_reg(ind_n, limits=False)
            except ValueError:
                continue
            distances = self.reg_obj.dist()
            fit = distances < tolerance
            n_fit = distances[fit].size
            if n_fit > best:
                best = n_fit
                best_enz = self.reg_obj.get_pnts_by_index(fit)
            if n_fit == n: # all points are on the geometry
                break
        return best_enz, iterations
