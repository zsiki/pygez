"""
.. module:: readers.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v2.0 license Copyright (C)
       2024- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

import re
from datetime import datetime
from angle import Angle

# GeoEasy codes to PyGE keys
codes = {2: 'station', 3: 'ih', 4: 'code', 5: 'id', 6: 'th',
         7: 'hz', 8: 'v', 9: 'distance', 11: 'hd', 20: 'pc', 21: 'hz',
         37: 'north', 38: 'east', 39: 'elev', 51: 'datetime',
         112: 'faces'}

gsi_codes = {11: 'id', 21: 'hz', 22: 'v', 31: 'distance', 32: 'hd',
             41: 'stn', 42: 'id', 43: 'ih', 71: 'code',
             84: 'stn_east', 85: 'stn_north', 86: 'stn_elev',
             87: 'th', 88: 'ih'}
gsi_keys = gsi_codes.keys()

class Reader:
    """ base class for all readers """

    RD_OK = 0
    RD_OPEN = -1
    RD_READ = -2
    RD_EOF = -3

    def __init__(self):
        self.__state = self.RD_OK

    @property
    def state(self) -> int:
        """ get reader state """
        return self.__state

    @state.setter
    def state(self, act_state: int) -> None:
        """ set reader state """
        self.__state = act_state

    def clear_state(self):
        """ reset reader state """
        self.state = self.RD_OK

class FileReader(Reader):
    """ class to read text file """

    def __init__(self, path: str, encoding:str="UTF-8"):
        """ initialize class

            :param: path path to file
            :param: text encoding of file
        """
        self._encoding = encoding
        self._path = path
        super().__init__()
        try:
            self._fp = open(self._path, "r", encoding=self._encoding)
        except (FileNotFoundError, IsADirectoryError):
            self.state = self.RD_OPEN
            self._fp = None

    def __del__(self):
        """ Close file before delete
        """
        try:
            self._fp.close()
        except Exception as _:
            pass

    def _get_line(self) -> str:
        """ read next line from file
            :returns: line as string or empty string on end of file
        """
        act_line = self._fp.readline().strip('\n\r')
        if not act_line:
            self.state = self.RD_EOF
        return act_line

    def get_next(self):
        """ dummy function
            have to be implemented in child class

            :returns: tuple of row key/empty, values
        """
        return None, None

    def load_data(self):
        """ Load all records into a dict

            :returns: dict of data
        """
        res = {}
        act_buf = None
        while self.state == self.RD_OK:
            key, act_buf = self.get_next()
            if self.state != self.RD_OK:
                break
            if key:
                res[key] = act_buf
        return res

class CsvReader(FileReader):
    """reader for csv coordinate list

        :param: path path to file
        :param: text encoding of file
        :param: header read header from first row of line with field names
        :param: field_names name of columns, header from file overrides it
        :param: separator field separator in file
    """

    def __init__(self, path:str, encoding:str='UTF-8', header:bool=True,
                 field_names:tuple=('id', 'east', 'north', 'elev', 'code'),
                 separator:str=";"):
        """ initialize class

        """
        super().__init__(path, encoding)
        self._header = header
        self._field_names = field_names
        self._separator = separator
        if self._header:
            buffer = [x.strip() for x in self._get_line().split(self._separator)]
            self._field_names = buffer

    def get_next(self):
        """ get next line as a dictionary """
        fields = [x.strip() for x in self._get_line().split(self._separator)]
        res = {}
        act_id = None
        for key, value in zip(self._field_names, fields):
            if len(value):
                if key == 'id':
                    act_id = value
                if key in ('east', 'north', 'elev'):
                    res[key] = float(value) # numeric
                else:
                    res[key] = value
        return act_id, res

class CooReader(FileReader):
    """ class to read GeoEasy 3 coo files """

    #def __init__(self, path: str, encoding:str="UTF-8"):
    #    """ initialize class
    #
    #        :param: path path to coo file
    #    """
    #    super().__init__(path, encoding)

    def get_next(self):
        """ get next line from file
            :returns: dictionary of data
        """
        res = {}
        act_id = None
        working_buf = self._get_line()
        if self.state != self.RD_OK:
            return None, res
        working_buf = re.split(r'[{}]', working_buf)
        for tag in working_buf:
            if len(tag) > 2:
                words = tag.strip().split(' ')
                key = int(words[0])
                if key in codes:
                    if key in (2, 5):     # ID
                        act_id = ' '.join(words[1:])
                    elif key in (37, 38, 39):
                        res[codes[key]] = float(words[1]) # numeric
                    elif key == 51:
                        try:
                            res[codes[key]] = datetime.strptime(
                                    ' '.join(words[1:]), '%Y-%m-%d %H:%M:%S')
                        except ValueError as _:
                            pass    # skip if datetime format is not valid
                    else:
                        res[codes[key]] = ' '.join(words[1:])
        return act_id, res

class GeoReader(FileReader):
    """ read geo file """

    #def __init__(self, path: str, encoding:str="UTF-8"):
    #    """ initialize class
    #
    #        :param path: path to geo file
    #        :param encoding: file encoding
    #    """
    #    super().__init__(path, encoding)

    def get_next(self) -> tuple:
        """ get next line from file
            :returns: dictionary of data
        """
        res = {}
        rec_id = None
        working_buf = self._get_line()
        if self.state != self.RD_OK:
            return None, res
        working_buf = re.split(r'[{}]', working_buf)
        for tag in working_buf:
            if len(tag) > 2:
                words = tag.strip().split(' ')
                key = int(words[0])
                if key in codes:
                    if key == 2:     # ID
                        rec_id = 'STATION'
                        res[codes[key]] = ' '.join(words[1:])
                    elif key in (5, 62):     # ID
                        rec_id = 'TARGET'
                        res[codes[key]] = ' '.join(words[1:])
                    elif key in (7, 8, 21):   # angles
                        # angles in DMS?
                        if re.search('-', words[1]):
                            res[codes[key]] = Angle(words[1], 'DMS')
                        else:
                            res[codes[key]] = Angle(float(words[1]))
                    elif key in (3, 6, 9, 11, 20):
                        res[codes[key]] = float(words[1]) # numeric
                    elif key == 112:
                        res[codes[key]] = int(words[1]) # numeric
                    elif key == 51:
                        try:
                            res[codes[key]] = datetime.strptime(
                                    ' '.join(words[1:]), '%Y-%m-%d %H:%M:%S')
                        except ValueError as _:
                            pass    # skip if datetime format is not valid
                    else:
                        res[codes[key]] = ' '.join(words[1:])
        return rec_id, res

    def load_data(self):
        """ Load all records into a dict

            :returns: list of data
        """
        res = []
        act_buf = None
        act_station = None
        act_targets = []
        while self.state == self.RD_OK:
            key, act_buf = self.get_next()
            if self.state != self.RD_OK:
                break
            if key == "STATION":
                if act_station:
                    res.append([act_station, act_targets])
                act_targets = []
                act_station = act_buf
            elif key == "TARGET" and act_station:
                act_targets.append(act_buf)
        if act_station:
            res.append([act_station, act_targets])
        return res

class DmpReader(FileReader):
    """ read dmp field-book file

        :param path: path to geo file
        :param encoding: file encoding
        :param header: file header given in input file
    """
    def __init__(self, path:str, encoding:str='UTF-8', header:bool=True,
                 field_names:tuple=('station', 'id', 'hz', 'v', 'distance', 'th','ih'),
                 separator:str=";"):
        """ initialize class

        """
        super().__init__(path, encoding)
        self._header = header
        self._field_names = field_names
        self._separator = separator
        if self._header:
            buffer = [x.strip() for x in self._get_line().split(self._separator)]
            self._field_names = buffer

    def get_next(self):
        """ get next line as a dictionary """
        fields = [x.strip() for x in self._get_line().split(self._separator)]
        res = {}
        act_stn = None
        act_ih = None
        for key, value in zip(self._field_names, fields):
            if len(value):
                if key == 'station':
                    act_stn = value
                elif key in ('hz', 'v'):
                    if '-' in value:
                        res[key] = Angle(value, 'DMS')
                    else:
                        res[key] = Angle(value)
                elif key in ('distance', 'th'):
                    res[key] = float(value) # numeric
                elif key == 'ih':
                    if act_ih is None:
                        act_ih = float(value)
                else:
                    res[key] = value
        return act_stn, act_ih, res

    def load_data(self):
        """ Load all records into a list of dicts

            :returns: list of data
        """
        res = []
        act_buf = None
        last_station = None
        last_ih = None
        act_targets = []
        while self.state == self.RD_OK:
            act_station, act_ih, act_buf = self.get_next()
            if self.state != self.RD_OK:
                break
            if last_station is not None and act_station != last_station:
                st = {'station': last_station}
                if last_ih is not None:
                    st['ih'] = last_ih
                res.append([st, act_targets])
                act_targets = []
            else:
                act_targets.append(act_buf)
            if act_station is not None:
                last_station = act_station
            if act_ih is not None:
                last_ih = act_ih
        # save last station
        if len(act_targets) > 0:
            res.append([act_station, act_targets])
        return res

class GsiReader(FileReader):
    """ read observations from GSI file

    """
    #def __init__(self, path:str, encoding:str='UTF-8'):
    #    """ initialize
    #    """
    #    super().__init__(path, encoding)

    @staticmethod
    def gsi_val(val, code, unit_code):
        """ convert value to different units
            :param value: value to convert
            :param code: GSI code e.g. 21, 22
            :param unit: unit code 
                0: Meter (last digit: 1mm)
                1: Feet (last digit: 1/1000ft)
                2: 400 gon
                3: 360° decimal
                4: 360° sexagesimal
                5: 6400 mil
                6: Meter (last digit: 1/10mm)
                7: Feet (last digit: 1/10‘000ft)
                8: Meter (last digit: 1/100mm)
            :returns: decimal value in meter or radian (Angle)
        """
        # remove leading zeros
        val = val.lstrip('0')
        if len(val) == 0:
            val = '0'
        if code in gsi_keys:
            if code in (11, 41, 42, 71):  # point id or point code no conversion
                pass
            elif '.' in val:
                val = float(val)
            else:
                val = int(val)
                if unit_code == 0:
                    val = int(val) / 1000   # meter
                elif unit_code == 1: # foot
                    val = int(val) / 1000 * 0.3048
                elif unit_code == 2:  # GON
                    val = Angle(int(val) / 100000, "GON")
                elif unit_code == 3:  # degree
                    val = Angle(int(val) / 10000, "DEG")
                elif unit_code == 4:    # DMS
                    val = Angle(int(val) / 10000, "PDEG")
                elif unit_code == 5:    # mil
                    val = Angle(int(val), "MIL")
                elif unit_code == 6:
                    val = int(val) / 10000  # meter
                elif unit_code == 7:
                    val = int(val) / 10000 * 0.3048 # foot
                elif unit_code == 8:
                    val = int(val) / 100000 # meter
        return val

    def get_next(self):
        """ get next line as a dictionary """
        act_line = self._get_line().strip()
        res = {}
        if len(act_line) == 0 or act_line[0] in ('!', chr(3)):
            return res
        field_size = 16
        if act_line[0] == '*':
            # 16 byte long words
            act_line = act_line[1:]
            field_size = 24
        while len(act_line) > 0:
            act_field = act_line[:field_size-1]
            act_line = act_line[field_size:]
            act_code = int(act_field[:2])
            act_unit = int(act_field[5]) if act_field[5] in range(9) else 0
            #act_sign = 1 if act_field[7] == '+' else -1 # TODO
            act_val = act_field[7:]
            if act_code in gsi_keys:
                res[gsi_codes[act_code]] = self.gsi_val(act_val, act_code, act_unit)
        return res

    def load_data(self):
        """ Load all records into a list of dicts

            :returns: list of data
             41: 'stn', 42: 'stn_id', 43: 'ih', 71: 'code',
             84: 'stn_east', 85: 'stn_north', 86: 'stn_elev',
        """
        res = []
        act_buf = None
        act_targets = []
        while self.state == self.RD_OK:
            act_buf = self.get_next()
            if self.state != self.RD_OK:
                break
            if not act_buf: # empty line
                continue
            if 'stn' in act_buf and act_buf['stn'] in ('2', '21') or \
                'stn_east' in act_buf or 'ih' in act_buf:
                # new station start
                if len(act_targets) > 0:
                    res.append([st, act_targets])
                    act_targets = []
                st = {'station': act_buf['id']}
                if 'ih' in act_buf:
                    st['ih'] = act_buf['ih']
            else:
                act_targets.append(act_buf)
        # save last station
        if len(act_targets) > 0:
            res.append([st, act_targets])
        return res

if __name__ == "__main__":
    cr = GsiReader('../src/tsdata/leica/network.GSI')
    data = cr.load_data()
    for station in data:
        print(station[0])
        for record in station[1]:
            print(record)
    print('-'*80)
    cr = CooReader('demodata/test1.coo')
    data = cr.load_data()
    print(data)
    print('-'*80)
    cr = CsvReader('demodata/test1.csv', header=False)
    data = cr.load_data()
    print(data)
    print('-'*80)
    cr = GeoReader('demodata/test1.geo')
    data = cr.load_data()
    for station in data:
        print(station[0])
        for record in station[1]:
            print(record)
    print('-'*80)
    cr = DmpReader('demodata/test1.dmp')
    data = cr.load_data()
    print(data)
    print('-'*80)
