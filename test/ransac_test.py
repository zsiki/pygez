#from math import sin, cos
import argparse
from matplotlib import pyplot as plt
from  matplotlib import patches
import numpy as np
from ransac import Ransac
from regression import (LinearReg, CircleReg, SphereReg, EllipseReg, Line3dReg,
                        CylinderReg, ConeReg)

DEFAULT_NUMP = 100
DEFAULT_MAX = 1000  # coordinate range 0 - DEFAULT_MAX
DEFAULT_NOISE = 0.1
DEFAULT_LIMIT = 0.03
DEFAULT_SEED = None
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nump', type=int, default=DEFAULT_NUMP,
                    help=f'Number of random points used, default: {DEFAULT_NUMP}')
parser.add_argument('-x', '--max', type=float, default=DEFAULT_MAX,
                    help=f'Maximal coordinate range, default: {DEFAULT_MAX}')
parser.add_argument('-r', '--random_noise', type=float, default=DEFAULT_NOISE,
                    help=f'Random noise range, default: {DEFAULT_NOISE}')
parser.add_argument('-s', '--random_seed', type=int, default=DEFAULT_SEED,
                    help=f'Random seed, default: {DEFAULT_SEED}')
parser.add_argument('-l', '--limit', type=float, default=DEFAULT_LIMIT,
                    help=f'Distance limit for RANSAC filter, default: {DEFAULT_LIMIT}')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Plot points')
args = parser.parse_args()

max_coo = args.max  # coordinate range [0..max_coo]
n_p =  args.nump       # number of points
noise = args.random_noise
ransac_limit = args.limit
if args.random_seed is not None:
    np.random.seed(args.random_seed)
# test for line
param_0 = np.array([-0.15432, 0.98802, 46.234])
east = np.random.rand(n_p) * max_coo    # generate random coords
north = -(param_0[0] * east + param_0[2]) / param_0[1]
en = np.c_[east, north]
en += np.random.rand(*en.shape) * noise / 2   # add noise
lr = LinearReg(en)
r = Ransac(lr)
en_line, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("2D line")
print(f"{en_line.shape[0]}/{en.shape[0]} points fit")
# calculate final params
lr1 = LinearReg(en_line)
params = lr1.lkn_reg(limits=True)
print(f"Calculated params: {params[0]:.5f} {params[1]:.5f} {params[2]:.3f}")
print(f"Limits:            {params[3]:.3f} {params[4]:.3f} - {params[5]:.3f} {params[6]:.3f}")
print(f"Original   params: {param_0[0]:.5f} {param_0[1]:.5f} {param_0[2]:.3f}")
print(f"RMS: {lr1.RMS():.3f}, iterations: {iterations}")
if args.plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(en[:,0], en[:,1], c='red', label='original points')
    ax.scatter(en_line[:,0], en_line[:,1], c='green', label='filtered points')
    ax.plot(params[3:-1:2], params[4::2], c='blue', label='LSM line')
    ax.set_title(f"{en_line.shape[0]}/{en.shape[0]} points fit")
    ax.set_aspect('equal')
    plt.legend()
    plt.show()
# test for plane
param_0 = np.array([0.57735, 0.80829, 0.11547, 12.451])
east = np.random.rand(n_p) * max_coo    # generate random coords
north = np.random.rand(n_p) * max_coo
elev = -(param_0[0] * east + param_0[1] * north + param_0[3]) / param_0[2]
enz = np.c_[east, north, elev]
enz += np.random.rand(*enz.shape) * noise / 3   # add noise
lr = LinearReg(enz)
r = Ransac(lr)
enz_plane, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("Plane")
print(f"{enz_plane.shape[0]}/{enz.shape[0]} points fit")
# calculate final params
lr1 = LinearReg(enz_plane)
params = lr1.lkn_reg(limits=True)
print(f"Calculated params: {params[0]:.5f} {params[1]:.5f} {params[2]:.5f} {params[3]:.3f}")
print("Convex hull: ", np.reshape(params[4:],(-1,3)))
print(f"Original   params: {param_0[0]:.5f} {param_0[1]:.5f} {param_0[2]:.5f} {param_0[3]:.3f}")
print(f"RMS: {lr1.RMS():.3f}, iterations: {iterations}")
if args.plot:
    points = np.reshape(params[4:],(-1,3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(enz[:, 0], enz[:, 1], enz[:, 2], c='red', label='original points')
    ax.scatter(enz_plane[:, 0], enz_plane[:, 1], enz_plane[:, 2], c='green', label='filtered points')
    ax.plot(points[:,0], points[:,1], points[:,2], c='blue', label='plane')
    ax.set_title(f"{enz_plane.shape[0]}/{enz_plane.shape[0]} points fit")
    ax.view_init(32, 60)
    ax.set_aspect('equal')
    plt.show()
# test for circle
param_0 = np.array([652.16, 369.25, 35.01])
alpha = np.random.rand(n_p) * np.pi * 2
east = param_0[2] * np.sin(alpha) + param_0[0]  # generate random coords
north = param_0[2] * np.cos(alpha) + param_0[1]
en = np.c_[east, north]
en += np.random.rand(*en.shape) * noise / 2   # add noise
cr = CircleReg(en)
r = Ransac(cr)
en_circle, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("Circle")
print(f"{en_circle.shape[0]}/{en.shape[0]} points fit")
# calculate final params
cr1 = CircleReg(en_circle)
params = cr1.lkn_reg(limits=True)
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f}")
print(f"RMS: {cr1.RMS():.3f}, iterations: {iterations}")
if args.plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(en[:,0], en[:,1], c='red', label='original points')
    ax.scatter(en_circle[:,0], en_circle[:,1], c='green', label='filtered points')
    circle = patches.Circle(params[:2], params[2], color='blue', linewidth=2,
                                    fill=False)
    ax.add_patch(circle)
    ax.set_title(f"{en_circle.shape[0]}/{en.shape[0]} points fit")
    ax.set_aspect('equal')
    plt.legend()
    plt.show()
# test for sphere
param_0 = np.array([652.161, 369.255, 475.231, 135.012])
alpha = np.random.rand(n_p) * np.pi * 2
beta =  np.random.rand(n_p) * np.pi * 2
east = param_0[3] * np.cos(beta) * np.sin(alpha) + param_0[0]  # generate random coords
north = param_0[3] * np.cos(beta) * np.cos(alpha) + param_0[1]
elev = param_0[3] * np.sin(beta) + param_0[2]
enz = np.c_[east, north, elev]
enz += np.random.rand(*enz.shape) * noise / 3   # add noise
sr = SphereReg(enz)
r = Ransac(sr)
enz_sphere, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("Sphere")
print(f"{enz_sphere.shape[0]}/{enz.shape[0]} points fit")
# calculate final params
sr1 = SphereReg(enz_sphere)
params = sr1.lkn_reg(limits=True)
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f}")
print(f"RMS: {sr1.RMS():.3f}, iterations: {iterations}")
if args.plot:
    # generate sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = params[0] + params[3] * np.outer(np.cos(u), np.sin(v))
    y = params[1] + params[3] * np.outer(np.sin(u), np.sin(v))
    z = params[2] + params[3] * np.outer(np.ones(np.size(u)), np.cos(v))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    ax.plot_wireframe(x, y, z, color='blue')
    ax.scatter(enz[:, 0], enz[:, 1], enz[:, 2], c='red', label='original points')
    ax.scatter(enz_sphere[:, 0], enz_sphere[:, 1], enz_sphere[:, 2], c='green', label='filtered points')
    ax.set_title(f"{enz_plane.shape[0]}/{enz_plane.shape[0]} points fit")
    ax.view_init(32, 60)
    ax.set_aspect('equal')
    plt.show()

# test for ellipse
param_0 = np.array([4.0, -3.5, 7.0, 3.0, np.pi / 6])
t = np.random.rand(n_p) * 2 * np.pi
east = param_0[0] + param_0[2] * np.cos(t) * np.cos(param_0[4]) - \
                    param_0[3] * np.sin(t) * np.sin(param_0[4])
north = param_0[1] + param_0[2] * np.cos(t) * np.sin(param_0[4]) + \
                     param_0[3] * np.sin(t) * np.cos(param_0[4])
en = np.c_[east, north]
en += np.random.rand(*en.shape) * noise / 2   # add noise
er = EllipseReg(en)
r = Ransac(er)
en_ellipse, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("Ellipse")
print(f"{en_ellipse.shape[0]}/{en.shape[0]} points fit")
# calculate final params
el1 = EllipseReg(en_ellipse)
try:
    params = el1.lkn_reg(limits=True)
except ValueError:
    print("*** ELLIPSE FITTING FAILED ***")
    params = None
if params is not None:
    print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f}")
    print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f}")
    print(f"RMS: {el1.RMS():.3f}, iterations: {iterations}")
    if args.plot:
        t_fit = np.linspace(0, 2*np.pi, 400)
        x0, y0, a_fit, b_fit, theta = params[:5]
        xf = x0 + a_fit*np.cos(t_fit)*np.cos(theta) - b_fit*np.sin(t_fit)*np.sin(theta)
        yf = y0 + a_fit*np.cos(t_fit)*np.sin(theta) + b_fit*np.sin(t_fit)*np.cos(theta)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(en[:,0], en[:,1], c='red', label='original points')
        ax.scatter(en_ellipse[:,0], en_ellipse[:,1], c='green', label='filtered points')
        ax.plot(xf, yf, c='blue', label='ellipse')
        ax.set_title(f"{en_ellipse.shape[0]}/{en.shape[0]} points fit")
        ax.set_aspect('equal')
        plt.legend()
        plt.show()

# 3D line
param_0 = np.array([100, 200, 300, 0.4356, 0.3245, 0.8396])
t = np.random.rand(n_p) * 100
east = param_0[0] + t * param_0[3]
north = param_0[1] + t * param_0[4]
elev = param_0[2] + t * param_0[5]
enz = np.c_[east, north, elev]
enz += np.random.rand(*enz.shape) * noise / 3   # add noise
lr =Line3dReg(enz)
r = Ransac(lr)
enz_line, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("3D line")
print(f"{enz_line.shape[0]}/{enz.shape[0]} points fit")
# calculate final params
enz1 = Line3dReg(enz_line)
params = enz1.lkn_reg(limits=True)
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f} {params[5]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f} {param_0[5]:.3f}")
print(f"RMS: {enz1.RMS():.3f}, iterations: {iterations}")
if args.plot:
    x = params[6::3]
    y = params[7::3]
    z = params[8::3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    ax.plot(x, y, z, color='blue')
    ax.scatter(enz[:, 0], enz[:, 1], enz[:, 2], c='red', label='original points')
    ax.scatter(enz_line[:, 0], enz_line[:, 1], enz_line[:, 2], c='green', label='filtered points')
    ax.set_title(f"{enz_line.shape[0]}/{enz.shape[0]} points fit")
    ax.view_init(32, 60)
    ax.set_aspect('equal')
    plt.show()

# Cylinder
u0 = np.array([0.5, -0.2, 0.1])
u0 = u0 / np.linalg.norm(u0)
param_0 = np.array([0.5, -0.2, 0.1, u0[0], u0[1], u0[2], 2.0])
base_center = param_0[:3]
radius = param_0[6]
# Heights and angles
height_max = 5
heights = np.random.uniform(-height_max, height_max, n_p)
angles = np.random.uniform(0, 2 * np.pi, n_p)

# build orthonormal basis (u,v,w) with u=u0
# pick an arbitrary vector not parallel to u:
arbitrary = np.array([1.0, 0.0, 0.0])
if np.abs(np.dot(arbitrary, u0)) > 0.9:
    arbitrary = np.array([0.0, 1.0, 0.0])
v = np.cross(u0, arbitrary)
v /= np.linalg.norm(v)
w = np.cross(u0, v)
w /= np.linalg.norm(w)

# Cylinder surface
# points on axis for each height
axis_points = base_center + np.outer(heights, u0)  # (N,3)
enz = axis_points + (np.cos(angles)[:,None] * radius * v) + \
        (np.sin(angles)[:,None] * radius * w)
# add noise
#enz += np.random.normal(scale=noise, size=enz.shape)
enz += np.random.rand(*enz.shape) * noise / 3   # add noise
#enz = np.array([[-0.337, -0.516, -2.085],
#                [2.438, 4.302, 0.641],
#                [1.915, 1.146, 2.460],
#                [2.521, 0.307, 0.895],
#                [-0.104, -2.498, -3.092],
#                [0.526, -2.564, -3.100]])
cr =CylinderReg(enz)
r = Ransac(cr)
enz_cyl, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("Cylinder")
print(f"{enz_cyl.shape[0]}/{enz.shape[0]} points fit")
# calculate final params
cr1 = CylinderReg(enz_cyl)
try:
    params = cr1.lkn_reg(limits=True)
except ValueError:
    print("*** CYLINDER FITTING FAILED ***")
    params = None
if params is not None:
    print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f} {params[5]:.3f} {params[6]:.3f}")
    print(f"Limits           :{params[7]:.3f} {params[8]:.3f} {params[9]:.3f} - {params[0]:.3f} {params[1]:.3f} {params[2]:.3f}")
    print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f} {param_0[5]:.3f} {param_0[6]:.3f}")
    print(f"RMS: {cr1.RMS():.3f}, iterations: {iterations}")
    if args.plot:
        maxh = np.linalg.norm(params[:3] - params[7:])
        heights = np.linspace(0, maxh, num=10)
        angles = np.linspace(0, 2 * np.pi, num=16)
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(arbitrary, params[3:6])) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        v = np.cross(params[3:6], arbitrary)
        v /= np.linalg.norm(v)
        w = np.cross(params[3:6], v)
        w /= np.linalg.norm(w)
        axis_points = params[:3] + np.outer(heights, params[3:6])  # (N,3)
        section = (np.cos(angles)[:,None] * radius * v) + \
                  (np.sin(angles)[:,None] * radius * w)
        xyz = axis_points[:,None,:] + section[None,:,:]
               
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface
        ax.plot_wireframe(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], color='blue')
        xx = [params[0], params[7]]
        yy = [params[1], params[8]]
        zz = [params[2], params[9]]
        ax.plot(xx, yy, zz, c='blue')
        ax.scatter(enz[:, 0], enz[:, 1], enz[:, 2], c='red', label='original points')
        ax.scatter(enz_cyl[:, 0], enz_cyl[:, 1], enz_cyl[:, 2], c='green', label='filtered points')
        ax.set_title(f"{enz_plane.shape[0]}/{enz_plane.shape[0]} points fit")
        ax.view_init(32, 60)
        ax.set_aspect('equal')
        plt.show()
#res = cr1.dist()
#print("Points X Y Z v")
#for i in range(enz.shape[0]):
#    print(f"{i:5} {enz[i,0]:10.3f} {enz[i,1]:10.3f} {enz[i,2]:10.3f} {res[i]:10.6f}")

# Cone
param_0 = np.array([0, 0, height_max, 0, 0, 1, np.pi/6])
apex = param_0[:3]
u = param_0[3:6]
u = v / np.linalg.norm(v)
alpha = param_0[6]
heights = np.random.rand(n_p) * height_max
axis_points = apex - np.outer(heights, u)  # (N,3)
angles = np.random.rand(n_p) * 2 * np.pi
r = np.abs(height_max - heights) * np.tan(alpha)
# build orthonormal basis (u,v,w) with u=u0
# pick an arbitrary vector not parallel to u:
arbitrary = np.array([1.0, 0.0, 0.0])
if np.abs(np.dot(arbitrary, u)) > 0.9:
    arbitrary = np.array([0.0, 1.0, 0.0])
v = np.cross(u, arbitrary)
v /= np.linalg.norm(v)
w = np.cross(u, v)
w /= np.linalg.norm(w)

enz = axis_points + (np.cos(angles)[:,None] * r[:,None] * v) + \
        (np.sin(angles)[:,None] * r[:,None] * w)
#enz = np.array([r * np.cos(angles), r * np.sin(angles), heights]).T
cr =ConeReg(enz, param_0)
r = Ransac(lr)
enz_cone, iterations = r.ransac_filter(tolerance=ransac_limit)
print("-" * 80)
print("Cone")
print(f"{enz_cone.shape[0]}/{enz.shape[0]} points fit")
# calculate final params
enz1 = ConeReg(enz_cone, param_0)
try:
    params = enz1.lkn_reg()
except ValueError:
    print("*** CONE FITTING FAILED ***")
    params = None
if params:
    print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f} {params[5]:.3f} {params[6]:.3f}")
    print(f"Limits           :{params[7]:.3f} {params[8]:.3f} {params[9]:.3f} - {params[10]:.3f} {params[11]:.3f} {params[12]:.3f}")
    print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f} {param_0[5]:.3f} {params[6]:.3f}")
    print(f"RMS: {enz1.RMS():.3f}, iterations: {iterations}")
    if args.plot:
        maxh = np.linalg.norm(params[:3] - params[7:10])    # height from apex
        maxh1 = np.linalg.norm(params[10:13] - params[7:10])    # real height range
        offset = maxh - maxh1
        heights = np.linspace(0, maxh1, num=10)
        angles = np.linspace(0, 2 * np.pi, num=16)
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(arbitrary, params[3:6])) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        v = np.cross(params[3:6], arbitrary)
        v /= np.linalg.norm(v)
        w = np.cross(params[3:6], v)
        w /= np.linalg.norm(w)
        axis_points = params[:3] - np.outer(heights+offset, params[3:6])  # (N,3)
        radius = (heights+offset) * np.tan(params[6])
        section = (np.cos(angles)[:,None] * radius[:,None] * v) + \
                (np.sin(angles)[:,None] * radius[:,None] * w)
        xyz = axis_points[:,None,:] + section[None,:,:]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface
        ax.plot_wireframe(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], color='blue')
        xx = [params[0], params[7]]
        yy = [params[1], params[8]]
        zz = [params[2], params[9]]
        ax.plot(xx, yy, zz, c='blue')
        ax.scatter(enz[:, 0], enz[:, 1], enz[:, 2], c='red', label='original points')
        ax.scatter(enz_cyl[:, 0], enz_cyl[:, 1], enz_cyl[:, 2], c='green', label='filtered points')
        ax.set_title(f"{enz_plane.shape[0]}/{enz_plane.shape[0]} points fit")
        ax.view_init(32, 60)
        ax.set_aspect('equal')
        plt.show()
