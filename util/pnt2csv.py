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
#import readers
from readers import reader_table, help_table
from writers import CsvWriter
from ts_file import guess_file_type

if __name__ == '__main__':
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
    # select reader by content
    extension = guess_file_type(args.name[0])
    if extension == "unknown":
        # select reader by extension
        root, extension = os.path.splitext(args.name[0])
        extension = extension.lower()
    reader = reader_table.get(extension, None)
    if not reader:
        print(f"Unknown file type: {args.name[0]}")
        sys.exit()
    cr = reader_table[extension](args.name[0])
    data = cr.load_data()
    cw = CsvWriter(fname=args.out, sep=args.sep, mode=MODE)
    cw.write_all(data)
