#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. module:: writers.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
           publish observation results.
           GPL v3.0 license
           Copyright (C) 2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

import sys
import datetime
from angle import Angle
from readers import codes

# PyGEZ keys to GeoEasy 3 codes
inv_codes = dict(zip(codes.values(), codes.keys()))

class Writer():
    """ Base class for different writers (virtual)

            :param angle: angle unit to use (str), default GON
            :param dist: distance and coordinate format (str), default .3f
            :param dt: date/time format (str), default ansi
            :param filt: list of keys to output (list), default None=print all
    """
    WR_OK = 0
    WR_OPEN = -1
    WR_WRITE = -2

    def __init__(self, angle:str='GON', dist:str='.3f',
                 dt:str='%Y-%m-%d %H:%M:%S', filt=None):
        """ Constructor
        """
        self.angle_format = angle
        self.dist_format = dist
        self.filt = filt
        self.dt_format = dt
        self.state = self.WR_OK

    def str_val(self, val):
        """ Get string representation of value

            :param val: value to convert to string
            :returns: value in string format
        """
        if isinstance(val, Angle):
            sval = str(val.GetAngle(self.angle_format))
        elif isinstance(val, float):
            sval = f"{val:{self.dist_format}}"
        elif isinstance(val, datetime.datetime):
            sval = val.strftime(self.dt_format)
        else:
            sval = str(val)
        return sval

class FileWriter(Writer):
    """ Class to write observations to file, in the form key=value;key=value,...

            :param fname: name of text file to write to (str), default None (write to stdout)
            :param angle: angle unit to use (str), default GON
            :param dist: distance and coordinate format (str), default 3 decimals
            :param dt: date/time format (str), default ansi
            :param filt: list of allowed keys (list), default None
            :param mode: mode of file open (a or w) (str)
            :param sep: field separator
    """

    def __init__(self, fname:str='stdout', angle:str='GON', dist:str='.3f',
                 dt:str='%Y-%m-%d %H:%M:%S', filt=None,
                 mode:str='a', sep:str=';', encoding:str='UTF-8'):
        """ Constructor
        """
        super().__init__(angle, dist, dt, filt)
        self.fname = fname
        self.mode = mode
        self.sep = sep
        self.encoding = encoding
        self.fp = None
        if fname == 'stdout':
            # write to stdout
            self.fp = sys.stdout
        else:
            try:
                self.fp = open(fname, mode, encoding=self.encoding)
            except Exception:
                self.state = self.WR_OPEN

    def __del__(self):
        """ Destructor
        """
        if self.fname != 'stdout':
            try:
                self.fp.close()
            except Exception:
                pass

    def write_data(self, data:dict):
        """ Write observation data to file

            :param data: dictionary with observation data
            :returns: True/False
        """
        line = ""
        for key, val in data.items():
            if self.filt is None or key in self.filt:
                line += key + "=" + self.str_val(val) + ";"
        try:
            self.fp.write(line + "\n")
            self.fp.flush()
        except Exception:
            self.state = self.WR_WRITE
            return False
        return True

    def write_all(self, all_data:dict):
        """ write all data from a dictionary 
            :param all_data: is a dictionary of dictionaries
        """
        for idd, vals in all_data.items():
            vals['id'] = idd
            if not self.write_data(vals):    # error writing
                break

class CsvWriter(FileWriter):
    """ Class to write observations to csv file
            :param fname: name of text file to write to (str)
            :param angle: angle unit to use (str), default GON
            :param dist: distance and coordinate format (str), default .3f
            :param dt: date/time format (str), default ansi
            :param filt: list of keys to output (list)
            :param mode: mode of file open (a or w) (str)
            :param sep: separator character in file (str)
            :param header: add header to file if true and mode is 'w'
    """

    def __init__(self, fname:str='stdout', angle:str='GON', dist:str='.3f',
                 dt:str='%Y-%m-%d %H:%M:%S',
                 filt:tuple=('id', 'east', 'north', 'elev'),
                 mode:str='a', sep:str=';', encoding:str='UTF-8', header=None):
        """ Constructor
        """
        super().__init__(fname, angle, dist, dt, filt, mode, sep, encoding)
        if self.state == self.WR_OK:
            if header and self.mode == 'w':
                try:
                    self.fp.write(self.sep.join(self.filt) + "\n")
                except Exception:
                    self.state = self.WR_WRITE

    def write_data(self, data:dict):
        """ Write observation data to csv file

            :param data: dictionary with observation data
            :returns: 0/-1/-2 OK/write error/empty not written
        """
        if self.state != self.WR_OK:
            return False
        linelist = ['' for i in range(len(self.filt))]
        for key, val in data.items():
            if key in self.filt:
                index = self.filt.index(key)
                linelist[index] = self.str_val(val)
        try:
            self.fp.write(self.sep.join(linelist) + "\n")
            self.fp.flush()
        except Exception:
            self.state = self.WR_WRITE
            return False
        return True

class CooWriter(FileWriter):
    """ Class to write observations to csv file
            :param fname: name of text file to write to (str)
            :param angle: angle unit to use (str), default GON
            :param dist: distance and coordinate format (str), default .3f
            :param dt: date/time format (str), default ansi
            :param filt: list of keys to output (list)
            :param mode: mode of file open (a or w) (str)
            :param sep: separator character in file (str)
    """

    def __init__(self, fname:str='stdout', angle:str='GON', dist:str='.3f',
                 dt:str='%Y-%m-%d %H:%M:%S',
                 filt:tuple=('id', 'east', 'north', 'elev'),
                 mode:str='a', sep:str=';', encoding:str='UTF-8'):
        """ Constructor
        """
        super().__init__(fname, angle, dist, dt, filt, mode, sep, encoding)

    def write_data(self, data):
        """ Write observation data to coo file

            :param data: dictionary with observation data
            :returns: 0/-1/-2 OK/write error/empty not written
        """
        if self.state != self.WR_OK:
            return False
        line = ""
        op = "{"
        cl = "}"
        for key, val in data.items():
            if self.filt is None or key in self.filt:
                line += f"{op}{inv_codes[key]} {self.str_val(val)}{cl} "
        try:
            self.fp.write(line + "\n")
            self.fp.flush()
        except Exception:
            self.state = self.WR_WRITE
            return False
        return True
