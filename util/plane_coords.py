"""
    Find plane of input points using RANSAC filter.
    Set up a 3D coordinate system aligned to the plane.
    Move origin to the upper left corner of points and
    change y axis direction to 180 degree.
"""

import sys
from os import path
import argparse
import numpy as np

from regression import LinearReg
from ransac import Ransac

parser = argparse.ArgumentParser(prog='plane_coords', description='fit plane to points using RANSAC filtering and calculate plane coordinates')
parser.add_argument('name', metavar='file_name', type=str, nargs=1,
                    help='input ascii point cloud')
parser.add_argument('-s', '--sep', type=str, default=";",
                    help='field separator in input file, default=;')
parser.add_argument('-t', '--tol', type=float, default=0.025,
                    help='Tolerance for RANSAC, default 0.025 m')
parser.add_argument('-r', '--iterations', type=int, default=None,
                    help='RANSAC iterations, default None (estimated)')
parser.add_argument('-c', '--scale', type=float, default=1,
                    help='Scale to pixels, default 1')
parser.add_argument('-o', '--offset', type=float, default=0,
                    help='Offset in pixels, default 0')
parser.add_argument('-i', '--withid', action='store_true',
                    help='there is an id in first column of input file')
parser.add_argument('-n', '--no_align', action='store_true',
                    help='Do not aligned CRS to plane')
args = parser.parse_args()

if args.withid:
    cols = (1,2,3)
else:
    cols = (0,1,2)

try:
    enz = np.loadtxt(args.name[0], delimiter=args.sep, usecols=cols)
except:
    print(f"File error: {args.name[0]}")
    sys.exit(1)

reg = LinearReg(enz)
# filter point with RANSAC
r = Ransac(reg)
enz_plane, iterations = r.ransac_filter(tolerance=args.tol, iterations=args.iterations)
final_reg = LinearReg(enz_plane)
params = final_reg.lkn_reg()
m_dist = np.max(final_reg.dist())
print(f"RMS: {final_reg.RMS():.4f}, max. distance: {m_dist:.4f}, iterations: {iterations}")
print(f"Calculated params: {params[0]:.7f} {params[1]:.7f} {params[2]:.7f} {params[3]:.4f}")
print(f"Filtered points {enz_plane.shape[0]} / {enz.shape[0]}")

if args.no_align:
    exit()
# align coordsys to plane
norm = params[:3]
if abs(norm[1]) < 0.9:  # non-parallel vector to norm
    t = np.array([0, 1, 0], float)
else:
    t = np.array([1, 0, 0], float)
u = np.cross(norm, t)   # axis directions in plane
u /= np.linalg.norm(u)
v = np.cross(norm, u)
v /= np.linalg.norm(v)
origin = np.zeros(3)
idx = np.argmax(np.abs(norm))
origin[idx] = -params[3] / norm[idx]
# transform points to plane coordinate system
rel = enz_plane - origin
U = rel @ u
V = rel @ v
# shitf origin
U = (U - np.min(U)) * args.scale + args.offset
V = np.abs(V - np.max(V)) * args.scale + args.offset
# print 2D coords
base, ext = path.splitext(args.name[0])
out_name = base + "_plane" + ext
np.savetxt(out_name, np.c_[U, V], fmt='%.4f', delimiter=args.sep)
