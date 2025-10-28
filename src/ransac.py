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
#import numpy as np

class Ransac:
    """ Class to solve generic ransac filtering
        :param reg_obj: object to handle geometry
        tolerance: tolerance distance from geometry
    """
    def __init__(self, reg_obj, tolerance=0.1):
        """ initialize """
        self.reg_obj = reg_obj
        self.tolerance = tolerance

    def ransac_filter(self, iterations=None):
        """ Apply RANSAC flter 
            returns array of filtered points
        """
        n = self.reg_obj.get_n()
        n_geom = self.reg_obj.min_n()
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
