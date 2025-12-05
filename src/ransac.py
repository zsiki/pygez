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
import numpy as np

class Ransac:
    """ Class to solve generic ransac filtering
        :param reg_obj: object to handle geometry
        :param tolerance: tolerance distance from geometry
    """
    def __init__(self, reg_obj, tolerance:float=0.1, iterations:int=None):
        """ initialize """
        self._reg_obj = reg_obj
        self._tolerance = tolerance
        self._iterations = iterations

    @property
    def reg_obj(self):
        """ return gemometry object """
        return self._reg_obj

    @property
    def tolerance(self) ->float:
        """ return tolerance """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tol:float):
        """ set tolerance """
        self._tolerance = tol

    @property
    def iterations(self) ->int:
        """ return iterations """
        return self._iterations

    @iterations.setter
    def iterations(self, it:int):
        """ set iterations"""
        self._iterations = it

    def ransac_filter(self, iterations=None) ->np.ndarray:
        """ Apply RANSAC filter 
            returns array of filtered points
        """
        n = self.reg_obj.nump
        n_geom = self.reg_obj.min_n
        if iterations is None:
            iterations = 10 * n
        indices = list(range(n))
        best = 0
        for _ in range(iterations):
            shuffle(indices)
            ind_n = indices[:n_geom]
            self.reg_obj.lkn_reg(ind_n)
            distances = self.reg_obj.dist()
            fit = distances < self.tolerance
            n_fit = len(fit)
            if n_fit > best:
                best = n_fit
                best_enz = self.reg_obj.get_pnts_by_index(fit)
            if n_fit == n: # all points are on the geometry
                break
        return best_enz
