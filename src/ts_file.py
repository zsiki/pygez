"""
    Copyright (C) 2025 Zoltan Siki <siki1958@gmail.com>
    Published under GPL 3 or later license

    Try to find out total station file type from the first few lines of
    a text file
"""
import os
import re
from readers import help_table

def guess_file_type(fname: str, n_lines:int=10, ratio:float=0.6) -> str:
    """ Guess total station file type """
    lines = []
    min_lines = int(ratio * n_lines + 0.5)
    if not os.path.isfile(fname):
        return ""   # file does not exists
    with open(fname, encoding="ASCII") as fp:
        # read the first n_lines lines
        i = 0
        for line in fp:
            lines.append(line.strip())
            i += 1
            if i > n_lines:
                break
    # SDR
    sdr_tag = 0
    sdr_line = 0
    for line in lines:
        if line[4:7] == "SDR":
            sdr_tag += 1
        if re.match(r"[01][0-9][A-Z]{2}", line):
            sdr_line += 1
    if sdr_tag and sdr_line > min_lines:
        return ".sdr"
    # M5
    m5_line = 0
    for line in lines:
        if re.match(r"For[ _]M5\|Adr", line):
            m5_line += 1
    if m5_line > min_lines:
        return ".m5"
    # LandXML
    xml_line = 0
    land_line = 0
    for line in lines:
        if re.match(r"<\?xml ", line):
            xml_line += 1
        if re.match(r"<LandXML ", line):
            land_line += 1
    if xml_line and land_line:
        return ".xml"
    # GSI
    gsi_line = 0
    for line in lines:
        if re.match(r"\*?[41]1[0-9]{4}\+", line):
            gsi_line += 1
    if gsi_line > min_lines:
        return ".gsi"
    # GeoEasy
    gez_line = 0
    for line in lines:
        if re.search(r"\{ *5 ", line):
            gez_line += 1
    if gez_line > min_lines:
        return ".coo"
    # RW5
    rw5_line = 0
    for line in lines:
        if re.match(r"[A-Z]{2},[A-Z]{2}", line):
            rw5_line += 1
    if rw5_line > min_lines:
        return ".rw5"
    # Topcon GTS-7
    GTS_TOKENS = ['JOB', 'DATE', 'NAME', 'INST', 'UNITS', 'SCALE', 'ATMOS',
                  'STN', 'XYZ', 'BKB', 'BS', 'FS', 'SS', 'CTL', 'HV', 'SD',
                  'HD', 'OFFSET', 'PLT_OFF', 'NOTE', 'MLM', 'RES_OBS']
    gts_line = 0
    for line in lines:
        if re.split(r'\W+', line)[0] in GTS_TOKENS:
            gts_line += 1
    if gts_line > min_lines:
        return '.gts7'
    return "unknown"   # unknown file

if __name__ == "__main__":
    import glob

    files = glob.glob("../testdata/*")
    for file in files:
        try:
            print(f"{file} : {help_table.get(guess_file_type(file), 'unknown')}")
        except UnicodeDecodeError:
            print(f"{file} : decode error")
