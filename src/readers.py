"""
.. module:: readers.py
   :platform: Unix, Windows
   :synopsis: PyGeoEZ - an open source project for surveying calculations
       GPL v3.0 license Copyright (C)
       2025- Zoltan Siki <siki1958@gmail.com>

.. moduleauthor:: Zoltan Siki <siki1958@gmail.com>
"""

# TODO finish observation reader for M5/RW5/...
import re
from math import pi
from datetime import datetime
import xml.etree.ElementTree as ET
from angle import Angle

# GeoEasy codes to PyGEZ keys
COO_CODES = {4: 'code', 5: 'id', 37: 'north', 38: 'east', 39: 'elev',
             51: 'datetime'}
OBS_CODES = {2: 'station', 3: 'ih', 4: 'code', 5: 'id', 6: 'th',
             7: 'hz', 8: 'v', 9: 'sd', 10: 'dh', 11: 'hd', 20: 'pc', 21: 'hz',
             51: 'datetime', 112: 'faces'}
CODES = COO_CODES | OBS_CODES

# Leica GSI codes for total stations
GSI_COO_CODES = {11: 'id', 71: 'code', 81: 'east', 82: 'north', 83: 'elev',
                 84: 'stn_east', 85: 'stn_north', 86: 'stn_elev'}
GSI_OBS_CODES = {11: 'id', 21: 'hz', 22: 'v', 31: 'sd', 32: 'hd',
                 41: 'stn', 42: 'id', 43: 'ih', 71: 'code', 87: 'th', 88: 'ih'}
GSI_CODES = GSI_COO_CODES | GSI_OBS_CODES
GSI_KEYS = GSI_CODES.keys()

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
            USE IT FOR COORDINATES

            :returns: dict of data
        """
        res = {}
        act_buf = None
        while self.state == self.RD_OK:
            act_buf = self.get_next()
            if self.state != self.RD_OK:
                break
            if not act_buf: # empty line
                continue
            if 'id' in act_buf:
                act_id = act_buf.pop('id')
                if 'east' in act_buf or 'north' in act_buf or 'elev' in act_buf:
                    res[act_id] = act_buf
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
        """ get next coordinate line as a dictionary """
        fields = [x.strip() for x in self._get_line().split(self._separator)]
        res = {}
        for key, value in zip(self._field_names, fields):
            if len(value):
                if key in ('east', 'north', 'elev'):
                    res[key] = float(value) # numeric
                else:
                    res[key] = value
        return res

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
        working_buf = self._get_line()
        if self.state != self.RD_OK:
            return None, res
        working_buf = re.split(r'[{}]', working_buf)
        for tag in working_buf:
            if len(tag) > 2:
                words = tag.strip().split(' ')
                key = int(words[0])
                if key in COO_CODES:
                    if key in (2, 5):     # ID
                        res['id'] = ' '.join(words[1:])
                    elif key in (37, 38, 39):
                        res[COO_CODES[key]] = float(words[1]) # numeric
                    elif key == 51:
                        try:
                            res[COO_CODES[key]] = datetime.strptime(
                                    ' '.join(words[1:]), '%Y-%m-%d %H:%M:%S')
                        except ValueError as _:
                            pass    # skip if datetime format is not valid
                    else:
                        res[COO_CODES[key]] = ' '.join(words[1:])
        return res

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
                if key in OBS_CODES:
                    if key == 2:     # ID
                        rec_id = 'STATION'
                        res[OBS_CODES[key]] = ' '.join(words[1:])
                    elif key in (5, 62):     # ID
                        rec_id = 'TARGET'
                        res[OBS_CODES[key]] = ' '.join(words[1:])
                    elif key in (7, 8, 21):   # angles
                        # angles in DMS?
                        if re.search('-', words[1]):
                            res[OBS_CODES[key]] = Angle(words[1], 'DMS')
                        else:
                            res[OBS_CODES[key]] = Angle(float(words[1]))
                    elif key in (3, 6, 9, 11, 20):
                        res[OBS_CODES[key]] = float(words[1]) # numeric
                    elif key == 112:
                        res[OBS_CODES[key]] = int(words[1]) # numeric
                    elif key == 51:
                        try:
                            res[OBS_CODES[key]] = datetime.strptime(
                                    ' '.join(words[1:]), '%Y-%m-%d %H:%M:%S')
                        except ValueError as _:
                            pass    # skip if datetime format is not valid
                    else:
                        res[OBS_CODES[key]] = ' '.join(words[1:])
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
                 field_names:tuple=('station', 'id', 'hz', 'v', 'sd', 'th','ih'),
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
                elif key in ('sd', 'th'):
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

class GsiCooReader(FileReader):
    """ coordinate reader class for Leica GSI format
        gsi_keys is a dictionary of valid GSI codes
    """
    def __init__(self, path:str, gsi_keys: dict, encoding:str='UTF-8'):
        """ initialize
        """
        super().__init__(path, encoding)
        self.gsi_keys = gsi_keys

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
        if code in GSI_KEYS:
            if code in (11, 41, 42, 71):  # point id or point code no conversion
                pass
            elif '.' in val:
                val = float(val)
            elif val.isnumeric():
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
        field_size = 16 # 8 byte long words
        if act_line[0] == '*':
            # 16 byte long words
            act_line = act_line[1:]
            field_size = 24
        while len(act_line) > 0:
            act_field = act_line[:field_size-1]
            act_line = act_line[field_size:]
            act_code = int(act_field[:2])
            act_unit = int(act_field[5]) if act_field[5] in "0123456789" else 0
            #act_sign = 1 if act_field[7] == '+' else -1 # TODO
            act_val = act_field[7:]
            if act_code in self.gsi_keys:
                res[GSI_CODES[act_code]] = self.gsi_val(act_val, act_code, act_unit)
        return res

class GsiObsReader(GsiCooReader):
    """ read observations from GSI file
    """
    def load_data(self):
        """ Load all observation records into a list of dicts

            :returns: list of data
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
            elif 'hz' in act_buf or 'v' in act_buf or 'sd' in act_buf:
                act_targets.append(act_buf)
        # save last station
        if len(act_targets) > 0:
            res.append([st, act_targets])
        return res

class M5Reader(FileReader):
    """ virtual base for Trimble M5 """
    P_INDEX = 15   # point number position in 2nd value block
    C_INDEX = 10  # code position in 2nd value block
    I_INDEX = 0   # info position in 2nd value block
    UNIT_MUL = {"ft": 0.3048, "m": 1.0,
                "DMS": "PDEG", "gon": "GON", "deg": "DEG", "mil": "MIL"}

    def __init__(self, path: str, filt: list, encoding:str="UTF-8"):
        """ initialize class
    
            :param path: path to M5 file
            :param filt: filter for fields
            :param encoding: file encoding
        """
        super().__init__(path, encoding)
        self._filt = filt

    def get_next(self) -> dict:
        """ get a line and return a dict """
        act_line = self._get_line()
        res = {}
        if not re.match(r'For[ _]M5', act_line):
            return res  # invalid line
        block_type = act_line[17:19]
        ih = 0  # default
        th = 0
        if block_type == "PI":  # point identification
            res["id"] = act_line[21+self.P_INDEX:33+self.P_INDEX].strip()
            code = act_line[21+self.C_INDEX:26+self.C_INDEX].strip()
            if len(code):
                res["code"] = code

        elif block_type == "TI":    # text information
            info = act_line[21+self.I_INDEX:28+self.I_INDEX].strip()
            if re.match(r'[KU]N STAT', info) or \
                    re.match(r'POLAR', info):
                pass
            #elif
        for i in range(49, 96, 23): # process 3rd-5th block
            block_type = act_line[i:i+2].strip()
            if block_type in self._filt:
                value = act_line[i+3:i+17].strip()
                unit = act_line[i+18:i+21].strip()
                if block_type == "ih":
                    ih = float(value) * self.UNIT_MUL[unit]
                elif block_type == "th":
                    th = float(value) * self.UNIT_MUL[unit]
                elif block_type == "SD":
                    res["sd"] = float(value) * self.UNIT_MUL[unit]
                elif block_type == "HD":
                    res["hd"] = float(value) * self.UNIT_MUL[unit]
                elif block_type == "h":
                    res["dh"] = float(value) * self.UNIT_MUL[unit]
                elif block_type == "Hz":
                    res["hz"] = Angle(float(value), self.UNIT_MUL[unit])
                elif block_type == "V1":
                    res["v"] = Angle(float(value), self.UNIT_MUL[unit])
                elif block_type == "V3":    # elevation angle
                    res["v"] = Angle(pi / 2) - \
                               Angle(float(value), self.UNIT_MUL[unit])
                elif block_type == "Y":
                    res["east"] = float(value) * self.UNIT_MUL[unit]
                elif block_type == "X":
                    res["north"] = float(value) * self.UNIT_MUL[unit]
                elif block_type == "Z":
                    res["elev"] = float(value) * self.UNIT_MUL[unit]
        return res

class M5CooReader(M5Reader):
    """ Load coordinates from Trimble M5 file """

    def __init__(self, path: str, encoding:str="UTF-8"):
        super().__init__(path, ["Y", "X", "Z"], encoding)


class M5ObsReader(M5Reader):
    """ Load observations from Trimble M5 file """

    def __init__(self, path: str, encoding:str="UTF-8"):
        super().__init__(path, ["ih", "th", "SD", "Hz", "V1"], encoding)

    def load_data(self):
        # TODO inherited loader is only for coords!
        pass

class SdrCooReader(FileReader):
    """ load coordinates from SRD file """

    def __init__(self, path:str, encoding:str='UTF-8'):
        """ initialize class
        """
        super().__init__(path, encoding)
        self._pn_length = 4 # SDR 2
        self._coo_order = 2  # EN
        self._ang_unit = 1 # DEG
        self._dist_unit = 1  # meter
        self._ang_dir = 1   # clockwise

    def get_next(self) -> dict:
        """ get next line """
        # TODO feet units
        act_line = self._get_line().strip()
        res = {}
        rec_type = act_line[0:2]
        match rec_type:
            case '00':
                # header line
                if act_line[4:9] == 'SDR33':
                    self._pn_length = 16
                units = act_line[-6:]
                self._dist_unit = units[1]  # 1 - meter, 2 feet
                self._coo_order = units[4]
            case '08':
                # co-ordinate
                pn = act_line[4:4+self._pn_length].strip()
                if self._pn_length == 16:
                    code = act_line[68:84].strip()
                    coo_start = 20
                    coo_len = 16
                else:
                    code = act_line[38:54].strip()
                    coo_start = 8
                    coo_len = 10
                east = float(act_line[coo_start:coo_start+coo_len].strip())
                coo_start += coo_len
                north = float(act_line[coo_start:coo_start+coo_len].strip())
                if self._coo_order == 1:    # NE order
                    east, north = north, east
                coo_start += coo_len
                elev = float(act_line[coo_start:coo_start+coo_len].strip())
                res = {'id': pn, 'code': code, 'east': east, 'north': north, 'elev': elev}
        return res

class Rw5CooReader(FileReader):
    """ load coordinates from RW5 file """

    def __init__(self, path:str, encoding:str='UTF-8', separator:str=","):
        """ initialize class
        """
        super().__init__(path, encoding)
        self._separator = separator
        self._dist_unit = 1  # meter

    def get_next(self) -> dict:
        """ read next line and parse """
        act_list = self._get_line().strip().split(self._separator)
        res = {}
        match act_list[0]:
            case 'MO':
                for field in act_list[1:]:
                    if field.startswith("UN"):
                        self._dist_unit = int(field.strip()[-1])    # m or feet
                        break
            case 'OC' | 'SP':
                for field in act_list[1:]:
                    match field[:2]:
                        case 'PN' | 'OP':
                            res['id'] = field[2:]
                        case 'N ':
                            res['north'] = float(field[2:])
                        case 'E ':
                            res['east'] = float(field[2:])
                        case 'EL':
                            res['elev'] = float(field[2:])
                        case '--':
                            res['code'] = field[2:]
        return res

class LandXmlCooReader(Reader):
    """ Class to read CgPoints from LAndXML """
    def __init__(self, path:str):
        super().__init__()
        self._path = path

    def load_data(self) ->dict:
        """ load all CgPoints from LandXML that have a point name """
        # Dictionary to store CgPoints
        points = {}

        try:
            tree = ET.parse(self._path)
        except:
            return points   # TODO error message
        root = tree.getroot()
        # Handle XML namespaces (LandXML uses one)
        ns = {'lx': root.tag.split('}')[0].strip('{')}

        # Iterate over all CgPoint elements
        for point_elem in root.findall('.//lx:CgPoint', ns):
            name = point_elem.get('name')
            if name:
                # Get coordinates (text content)
                text = point_elem.text.strip()
                coords = [float(x) for x in text.split()]
                if len(coords) == 2:
                    points[name] = {'east': coords[0], 'north': coords[1]}
                if len(coords) == 3:
                    points[name] = {'east': coords[0], 'north': coords[1],
                                    'elev': coords[2]}
        return points
