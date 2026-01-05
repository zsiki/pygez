#! /usr/bin/env python3
"""
    Multi-circle/ellipse fiting using RANSAC filtering before LSM fitting
"""

from math import sqrt, atan, atan2, pi
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ransac import Ransac
from regression import CircleReg, EllipseReg, Line3dReg

def generate_points(x0, y0, ap, bp, phi, npts=100, tmin=0, tmax=2*np.pi):
    """ generate test points on ellipse/circle
        x0, y0 center of ellipse/circle
        ap, bp semi axices (ap = bp circle)
        phi rotation
        npts number of points
        noise to move x and y
        tmin, tmax agle parameter range
    """
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def ell_draw(x, y, xr, yr, x0, y0, ap, bp, phi, title=''):
    """ draw the ellipse
        x, y all points
        xr, yr RANSAC filtered points
        x0, y0, ap, bp, phi ellipse parameters
    """
    plt.rcParams['font.size'] = 14
    plt.plot(x, y, 'x', label='outliers')
    plt.plot(xr, yr, 'o', label='inliers')
    x_lst, y_lst = generate_points(x0, y0, ap, bp, phi, 100)
    plt.plot(x_lst, y_lst)
    plt.axis('scaled')
    plt.title(title)
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.legend()
    plt.show()

parser = argparse.ArgumentParser(prog='tower', description='fit circle or ellipse to points in horizontal sections using RANSAC filtering')
parser.add_argument('name', metavar='file_name', type=str, nargs=1,
                    help='input ascii point cloud')
parser.add_argument('-s', '--sep', type=str, default=";",
                    help='field separator in input file, default=;')
parser.add_argument('-t', '--tol', type=float, default=0.025,
                    help='Tolerance for RANSAC, default 0.025 m')
parser.add_argument('-v', '--vtol', type=float, default=0.025,
                    help='Vertical tolerance for sections, default 0.025 m')
parser.add_argument('-i', '--withid', action='store_true',
                    help='there is an id in first column of input file')
parser.add_argument('-p', '--print_coo', action='store_true',
                    help='print coordinates of points on ellipse/circle')
parser.add_argument('-e', '--elev', nargs='+', required=False, default=['0'],
                    help='elevations to make sections, default=0')
parser.add_argument('-l', '--ellipse', action='store_true',
                    help='fit ellipse not circle')
parser.add_argument('-d', '--draw', action='store_true',
                    help='draw result')
group = parser.add_mutually_exclusive_group()
group.add_argument('-r', '--iterations', type=int, default=None,
                    help='RANSAC iterations, default None (estimated)')
group.add_argument('-w', '--percent', type=float, default=0.51,
                    help='Percent of inliers for RANSAC , default 0.51')
args = parser.parse_args()

if args.withid:
    cols = (1,2,3)
else:
    cols = (0,1,2)

try:
    pc = np.loadtxt(args.name[0], delimiter=args.sep, usecols=cols)
except:
    print(f"File error: {args.name[0]}")
    sys.exit(1)

MIN_N = 5 if args.ellipse else 3
centers = []
pp = []
for i, h in enumerate(args.elev):
    height = float(h)
    # select points at given height
    section = pc[np.abs(pc[:,2] - height) < args.vtol]
    if section.shape[0] > MIN_N:
        if args.ellipse:
            reg = EllipseReg(section[:,:2])
        else:
            reg = CircleReg(section[:,:2])
        r = Ransac(reg)
        en, iterations = r.ransac_filter(tolerance=args.tol, iterations=args.iterations, w=args.percent)
        if args.ellipse:
            reg = EllipseReg(en)
        else:
            reg = CircleReg(en)
        params = reg.lkn_reg()
        e0 = params[0]  # center
        n0 = params[1]
        centers.append([e0, n0, height])
        pp.append(params[2:])
        if args.ellipse:
            a = params[2]
            b = params[3]
            phi = params[4]
            rms = reg.RMS()
            print(f"Ellipse: {e0:.3f},{n0:.3f},{height:.3f},{a:.3f},{b:.3f},{phi * 180 / np.pi:.4f},{rms:.3f},{en.shape[0]}/{section.shape[0]}/{iterations}")
        else:
            a = params[2]
            b = params[2]
            phi = 0
            rms = reg.RMS()
            print(f"Circle: {e0:.3f},{n0:.3f},{height:.3f},{a:.3f},{rms:.3f},{en.shape[0]}/{section.shape[0]}")
        if args.draw:
            if args.ellipse:
                geom = "Ellipse"
            else:
                geom = "Circle"
            text = f"{geom} section {h} m RMS={rms:.3f} {en.shape[0]}/{section.shape[0]}"
            ell_draw(section[:,0], section[:,1], en[:,0], en[:,1],
                     e0, n0, a, b, phi,
                     f"Section {h} m RMS={rms:.3f} {en.shape[0]}/{section.shape[0]}")

if len(centers) > 2:
    # fit 3D line to centers without RANSAC
    enz = np.array(centers)
    pars = np.array(pp)
    lr =Line3dReg(enz)
    l3d = lr.lkn_reg()
    print(f"Axis line:\n x = {l3d[0]:12.3f} + {l3d[3]:12.6f} * t\n y = {l3d[1]:12.3f} + {l3d[4]:12.6f} * t\n z = {l3d[2]:12.3f} + {l3d[5]:12.6f} * t\n")
    tilt = atan(sqrt(l3d[3]**2 + l3d[4]**2) / l3d[5]) / pi * 180
    azi = atan2(l3d[3], l3d[4]) / pi * 180
    if azi < 0:
        azi += 400
    print(f"Tilt angle: {tilt:.4f} deg, Tilt direction: {azi:.4f} deg")
    print(f"RMS: {lr.RMS():.3f}")
    if args.draw:
        # create 3D plot
        # axis between lowest and highest section
        z0 = enz[0,2]
        t0 = (z0 - l3d[2]) / l3d[5]
        e0 = l3d[0] + t0 * l3d[3]
        n0 = l3d[1] + t0 * l3d[4]
        z1 = enz[-1,2]
        t1 = (z1 - l3d[2]) / l3d[5]
        e1 = l3d[0] + t1 * l3d[3]
        n1 = l3d[1] + t1 * l3d[4]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([e0, e1], [n0, n1], [z0, z1], c='blue')
        for i in range(len(centers)):
            e_lst, n_lst= generate_points(enz[i,0], enz[i,1], pars[i,0], pars[i,1], pars[i,2])
            z_lst = np.full_like(e_lst, enz[i, 2])
            ax.plot(e_lst, n_lst, z_lst, c='red')
            ax.set_aspect('equal')
            ax.view_init(32, 60)
        plt.show()
