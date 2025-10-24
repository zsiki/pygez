#!/usr/bin/env python
"""
.. module:: fieldbook.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v3.0 license Copyright (C)
       2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

from typing import Union
from os import path
from readers import (GeoReader, DmpReader)

class FieldBook:
    """ Base field book object
        :param: fb_type fieldbook type (directions, levelling/distances)
        :param: source description of source (filename, URL, etc.
    """
    DIRECTIONS_FB = 1
    LEVELLING_FB = 2
    DISTANCES_FB = 3
    FB_TYPES = [DIRECTIONS_FB, LEVELLING_FB, DISTANCES_FB]

    FB_LOADED = 1   # bitwise states
    FB_CHANGED = 2
    FB_ERROR = 4

    # jump table to reader based on extension
    reader_table = {'.geo': GeoReader, '.dmp': DmpReader, '.gsi': GsiReader}

    def __init__(self, fb_type: int, source: Union[str, None]):
        """ initialize field book """
        self.source = source
        self.fb_type = fb_type
        self.__state = 0
        self._stations = []

    @property
    def state(self) -> int:
        """ get reader state """
        return self.__state

    @state.setter
    def state(self, act_state: int) -> None:
        """ set reader state """
        self.__state = act_state

    @property
    def data(self):
        """ get reader state """
        return self._stations

    @data.setter
    def data(self, my_data):
        """ return data """
        self._stations = my_data

    def load(self):
        """ load a field book from source
        """
        _, ext = path.splitext(self.source)
        reader_obj = self.reader_table.get(ext, None) # reader object to create
        if reader_obj:
            my_reader = reader_obj(self.source)
            self._stations = my_reader.load_data()

    def to_str(self, station_fields:tuple=('station', 'ih'),
               target_fields:tuple=('id', 'th', 'hz', 'v', 'distance')) -> str:
        """ create printable string from fieldbook
            :param: station_fields list of fields from station data to show
            :param: target_fields list of fields from target data to show
            :returns: string of field-book data
        """
        res = ""
        for station in self._stations:
            # print station record
            for station_field in station_fields:
                if station_field in station[0]:
                    res += f"{station_field}: {station[0][station_field]} "
            res += "\n"
            for record in station[1]:
                for target_field in target_fields:
                    if target_field in record:
                        res += f"{target_field}: {record[target_field]} "
                res += "\n"
        return res

    def to_md(self, station_fields:tuple=(('station','s'), ('ih', ".2f")),
              target_fields:tuple=(('id', 's'), ('th', '.2f'), ('hz', 'DMS'),
                                   ('v', 'DMS'), ('distance', '.3f'))) -> str:
        """ createe markdown string representation of field-book
            :param: station_fields list of tuples (fieldname format)
            :param: target_fields list of tuples (fieldname format)
            :returns: markdown representation of filed-book

            format is any f-string format specifier
        """
        md_str = ""
        for station in self._stations:
            # print station record
            for station_field in station_fields:
                if station_field[0] in station[0]:
                    md_str += f"{station[0][station_field[0]]:{station_field[1]}} "
            md_str += "\n"
            for record in station[1]:
                for target_field in target_fields:
                    if target_field[0] in record:
                        md_str += f"{record[target_field[0]]:{target_field[1]}} "
                md_str += "\n"
        return md_str


if __name__ == "__main__":
    fb = FieldBook(FieldBook.DIRECTIONS_FB, '../testdata/test.geo')
    fb.load()
    if fb.state == 0:
        print(fb.to_md())
