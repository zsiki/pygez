#!/usr/bin/env python
"""
.. module:: point.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v3.0 license Copyright (C)
       2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

from math import atan2, hypot
from typing import Union
import numpy as np

class Pnt:
    """ Base point object without ID
        use east, north, elev names for coordinates
        other name/tags can be used for extra values
    """

    def __init__(self, **kwargs):
        """ initialize new point object """
        if 'east' in kwargs and 'north' in kwargs or 'elev' in kwargs:
            self.__point_attrs = kwargs
        else:
            raise ValueError('Pnt: horizontal position or elevation have to be given')

    def get_attr(self, attr_name: str):
        """ Get the value of a property
            :param: attr_name name of attribute
            :returns: value of the attribute
        """
        return self.__point_attrs.get(attr_name, None)

    def set_attr(self, attr_name:str, attr_val):
        """ Set attribute value use only for those attributes which have no
            setter method
            :param: attr_name attribute name to modify
            :param: attr_val value of attribute to set
        """
        self.__point_attrs[attr_name] = attr_val

    @property
    def east(self) -> float:
        """ :returns: east coordinate or None
        """
        return self.__point_attrs.get('east', None)

    @east.setter
    def east(self, new_value: float):
        """ set east coordinate of point
            :param: new_value the new east coordinate
        """
        self.__point_attrs['east'] = new_value

    @property
    def north(self) -> float:
        """ :returns: north coordinate or None
        """
        return self.__point_attrs.get('north', None)

    @north.setter
    def north(self, new_value: float):
        """ set north coordinate of point
            :param: new_value the new north coordinate
        """
        self.__point_attrs['north'] = new_value

    @property
    def elev(self) -> float:
        """ :returns: elevation or None
        """
        return self.__point_attrs.get('elev', None)

    @elev.setter
    def elev(self, new_value: float):
        """ set elevation of point
            :param: new_value the new elevation
        """
        self.__point_attrs['elev'] = new_value

    def __str__(self) -> str:
        """ called by print method """
        res = ""
        for key, val in self.__point_attrs.items():
            res += f" {key}: {val}"
        return res

    def __iadd__(self, p_1):
        """ move point (sum of coordinates)
            :param: p_1 offset to move with
        """
        try:
            self.east += p_1.east
        except TypeError:
            pass
        try:
            self.north += p_1.north
        except TypeError:
            pass
        try:
            self.elev += p_1.elev
        except TypeError:
            pass
        return self

    def __add__(self, p_1):
        """ create a new point from sum of coordinates
            :param: p_1 point to add
        """
        res = {}
        try:
            res['east'] = self.east + p_1.east
        except TypeError:
            pass
        try:
            res['north'] = self.north + p_1.north
        except TypeError:
            pass
        try:
            res['elev'] = self.elev + p_1.elev
        except TypeError:
            pass
        return Pnt(**res)

    def __isub__(self, p_1):
        """ coordinate differences
            :param: p_1 offset to move with
        """
        try:
            self.east -= p_1.east
        except TypeError:
            pass
        try:
            self.north -= p_1.north
        except TypeError:
            pass
        try:
            self.elev -= p_1.elev
        except TypeError:
            pass
        return self

    def __sub__(self, p_1):
        """ create a new point from differences of coordinates
            :param: p_1 point to substract
        """
        res = {}
        try:
            res['east'] = self.east - p_1.east
        except TypeError:
            pass
        try:
            res['north'] = self.north - p_1.north
        except TypeError:
            pass
        try:
            res['elev'] = self.elev - p_1.elev
        except TypeError:
            pass
        return Pnt(**res)

    def bearing(self) -> Union[float, None]:
        """ calculate bearing from origin """
        try:
            angle = atan2(self.east, self.north)
        except TypeError:
            return None
        return angle

    def distance(self) -> Union[float, None]:
        """ calculate distance from origin """
        try:
            dist = hypot(self.east, self.north)
        except TypeError:
            return None
        return dist

class Point(Pnt):
    """ Base point object with ID """

    def __init__(self, point_id, **kwargs):
        """ initialize new point object """
        self.__point_id = point_id
        super().__init__(**kwargs)

    @property
    def point_id(self) -> str:
        """ :returns: point identifier
        """
        return self.__point_id

    @point_id.setter
    def point_id(self, new_id: str):
        """ set identifier of point
            :param: new_id the new point identifier
        """
        self.__point_id = new_id

    def __str__(self) -> str:
        """ called by print method """
        res = f"point_id: {self.__point_id}"
        res += super().__str__()
        return res

class PntList:
    """ class for point list (dictionary) """

    def __init__(self, p_dic:Union[dict, None]=None):
        """ initialize new point list
            :param p_dic: dictionary to initialize PntList
        """
        if p_dic is None:
            self.__pnt_list = {}
        else:
            self.__pnt_list = p_dic.copy()
        self.__list_source = None
        self.__changed = False

    def get_pnt(self, pnt_id: str) -> Union[dict, None]:
        """ get a point by id
            :param: pnt_id point to get
            :returns: dictionary of pnt data or None if id not found
        """
        return self.__pnt_list.get(pnt_id, None)

    def add_pnt(self, new_id: str, new_pnt:Pnt, overwrt:bool=True) -> bool:
        """ add a new point or overwrite an existing one """
        if overwrt or not new_id in self.__pnt_list:
            self.__pnt_list[new_id] = new_pnt
            self.__changed = True
            return True
        return False

    def remove_pnt(self, pnt_id: str) -> bool:
        """ remove a point from point list
            :param: pnt_id point ID to remove
            :returms: True/False succes/no such key
        """
        try:
            del self.__pnt_list[pnt_id]
        except KeyError:
            return False
        self.__changed = True
        return True

    def add_pnts(self, new_dict: dict):
        """ add an other point dictionary common points are overwitten by new
            :param: new_dict dictionary of point data
        """
        self.__pnt_list = self.__pnt_list.update(new_dict)
        self.__changed = True

    def add_pnts_safe(self, new_dict: dict):
        """ add an other point dictionary common point are not overwritten
            :param: new_dict dictionary of point data
        """
        for key, val in new_dict.items():
            if not key in self.__pnt_list:
                self.__pnt_list[key] = val
                self.__changed = True

    def load_coo(self, coo_path: str, append_coo: bool=True):
        """ load point data from GeoEasy 3 coo file
            :param: coo_path path to file to load
            :param: append_coo keep old points in list
        """
        # TODO it is implemented in readers
        self.__list_source = coo_path
        pass

    def save_coo(self):
        """ Save coordinate list to source """
        if self.__changed:
            # TODO it is implemented in writers
            pass

    def save_as(self, target: str):
        """ Save coordinates to other format """
        # TODO use writer classes
        pass

    def to_array(self, dimension: int) -> np.ndarray:
        """ Convert point list to a numpy array, easting, northing and elev only
            :param dimension: 2/3
            :returns array of coordinates (2D or 3D)
        """
        # TODO filter on points
        res = []
        for point_id, data in self.__list.items():
            if 'east' in data and 'north' in data:
                if dimension == 2:
                    res.append([data["east"], data["north"]])
                elif dimension == 3 and "elev" in data:
                    res.append([data["east"], data["north"], data["elev"]])
        return np.array(res)

if __name__ == "__main__":

    my_point = Pnt(east=1, north=1)
    print(my_point)
    print(my_point.east, my_point.north, my_point.elev)
    print(my_point.bearing(), my_point.distance())
    my_point += Pnt(east=5, north=7)
    print(my_point)
    print(my_point + Pnt(east=-5, north=-7))
