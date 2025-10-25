#!/usr/bin/env python3

"""
    Copyright (C) 2025 Zoltan Siki <siki1958@gmail.com>
    Published under GPL 3 or later license

    A utility to convert coordinates from total station format to CSV file
    with command line interface (CLI).
    To get a list of supported input file types use -f/--formats switch.

    Usage:
    1st get help on CLI parameters and supported formats
        python3 pnt2csv.py -h
    or
        python3 pnt2csv.py --help
    2nd convert a GSI file and list it to the consol
        python3 pnt2csv.py test.gsi
    3rd convert a LandXML file to a CSV, overwrite output if exists
        python3 pnt2csv.py --ovr -o test.csv test.xml
    4th convert a SurvCE Raw file to CSV and append output to an existing file
        python3 pnt2csv.py --append -o test.csv test.rw5
"""
import os
import sys
import argparse
import readers
import writers

if __name__ == '__main__':
    # jump table for readers by extension
    reader_table = {'.coo': readers.CooReader,
                    '.gsi': readers.GsiCooReader,
                    '.m5' : readers.M5CooReader,
                    '.sdr': readers.SdrCooReader,
                    '.rw5': readers.Rw5CooReader,
                    '.xml': readers.LandXmlCooReader}

    help_table   = {'.coo': "GeoEasy COO files",
                    '.gsi': "Leica GSI files",
                    '.m5' : "Trimble M5 files",
                    '.sdr': "Sokkia SDR files",
                    '.rw5': "Carlson SurvCE Raw format files (RW5)",
                    '.xml': "LandXml  files (XML)"}

    # build epilog for help
    EPILOG = "Supported file formats:\n" + \
             "\n".join([f"{key} : {value}" for key, value in help_table.items()])
    parser = argparse.ArgumentParser(prog='coo2csv', description=
                'Convert different total station formats to a coordinate list',
                epilog=EPILOG,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', metavar='file_name', type=str, nargs=1,
                        help='input totalstation file with coords')
    parser.add_argument('-o', '--out', type=str, default="stdout",
                        help='Name of output file, default=stdout')
    parser.add_argument('-s', '--sep', type=str, default=";",
                        help='field separator in output file, default=;')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--ovr', action='store_true',
                        help="Overwrite output file")
    group.add_argument('-a', '--append', action='store_true',
                        help="Append content to output file")
    args = parser.parse_args()

    # check parameters
    if not os.path.isfile(args.name[0]):
        print(f"Input file does not exist: {args.name[0]}")
        sys.exit()
    if args.out != "stdout":
        if os.path.isfile(args.out) and (not args.ovr or not args.append):
            print(f"Output file exists, use --ovr/--append to overwrite/append: {args.out}")
            sys.exit()
    MODE = "a" if args.append else "w"
    # select reader by extension
    root, extension = os.path.splitext(args.name[0])
    extension = extension.lower()
    cr = reader_table[extension](args.name[0])
    data = cr.load_data()
    cw = writers.CsvWriter(fname=args.out, sep=args.sep, mode=MODE)
    cw.write_all(data)
