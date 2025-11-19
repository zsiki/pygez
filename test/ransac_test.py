from math import sin, cos
import numpy as np
from ransac import Ransac
from regression import (LinearReg, CircleReg, SphereReg, EllipseReg, Line3dReg,
                        CylinderReg, ConeReg)

max_coo = 1000  # coordinate range [0..max_coo]
n_p =  100       # number of points
# test for line
param_0 = np.array([-0.15432, 0.98802, 46.234])
east = np.random.rand(n_p) * max_coo    # generate random coords
north = -(param_0[0] * east + param_0[2]) / param_0[1]
north += np.random.rand(n_p)    # add noise
en = np.c_[east, north]
lr = LinearReg(en)
r = Ransac(lr, 0.5)
en_line = r.ransac_filter()
print("-" * 80)
print("2D line")
print(f"{en_line.shape[0]} points fit")
# calculate final params
lr1 = LinearReg(en_line)
params = lr1.lkn_reg()
print(f"Calculated params: {params[0]:.5f} {params[1]:.5f} {params[2]:.3f}")
print(f"Original   params: {param_0[0]:.5f} {param_0[1]:.5f} {param_0[2]:.3f}")
print(f"RMS: {lr1.RMS():.3f}")
# test for plane
param_0 = np.array([0.57735, 0.80829, 0.11547, 12.451])
east = np.random.rand(n_p) * max_coo    # generate random coords
north = np.random.rand(n_p) * max_coo
elev = -(param_0[0] * east + param_0[1] * north + param_0[3]) / param_0[2]
elev += np.random.rand(n_p) # add random noise
enz = np.c_[east, north, elev]
lr = LinearReg(enz)
r = Ransac(lr, 0.1)
enz_plane = r.ransac_filter()
print("-" * 80)
print("Plane")
print(f"{enz_plane.shape[0]} points fit")
# calculate final params
lr1 = LinearReg(enz_plane)
params = lr1.lkn_reg()
print(f"Calculated params: {params[0]:.5f} {params[1]:.5f} {params[2]:.5f} {params[3]:.3f}")
print(f"Original   params: {param_0[0]:.5f} {param_0[1]:.5f} {param_0[2]:.5f} {param_0[3]:.3f}")
print(f"RMS: {lr1.RMS():.3f}")
# test for circle
param_0 = np.array([652.16, 369.25, 35.01])
alpha = np.random.rand(n_p) * np.pi * 2
east = param_0[2] * np.sin(alpha) + param_0[0]  # generate random coords
north = param_0[2] * np.cos(alpha) + param_0[1]
north += np.random.rand(n_p)    # add random noise
en = np.c_[east, north]
cr = CircleReg(en)
r = Ransac(cr, 0.3)
en_circle = r.ransac_filter()
print("-" * 80)
print("Circle")
print(f"{en_circle.shape[0]} points fit")
# calculate final params
cr1 = CircleReg(en_circle)
params = cr1.lkn_reg()
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f}")
print(f"RMS: {cr1.RMS():.3f}")
# test for sphere
param_0 = np.array([652.161, 369.255, 475.231, 135.012])
alpha = np.random.rand(n_p) * np.pi * 2
beta =  np.random.rand(n_p) * np.pi * 2
east = param_0[3] * np.cos(beta) * np.sin(alpha) + param_0[0]  # generate random coords
north = param_0[3] * np.cos(beta) * np.cos(alpha) + param_0[1]
elev = param_0[3] * np.sin(beta) + param_0[2]
elev += np.random.rand(n_p) / 9    # add random noise
enz = np.c_[east, north, elev]
sr = SphereReg(enz)
r = Ransac(sr, 0.3)
enz_sphere = r.ransac_filter()
print("-" * 80)
print("Sphere")
print(f"{enz_sphere.shape[0]} points fit")
# calculate final params
sr1 = SphereReg(enz_sphere)
params = sr1.lkn_reg()
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f}")
print(f"RMS: {sr1.RMS():.3f}")
# test for ellipse
param_0 = np.array([4.0, -3.5, 7.0, 3.0, np.pi / 6])
t = np.random.rand(n_p) * 2 * np.pi
east = param_0[0] + param_0[2] * np.cos(t) * np.cos(param_0[4]) - \
                    param_0[3] * np.sin(t) * np.sin(param_0[4])
north = param_0[1] + param_0[2] * np.cos(t) * np.sin(param_0[4]) + \
                     param_0[3] * np.sin(t) * np.cos(param_0[4])
north += np.random.rand(n_p) / 9    # add random noise
en = np.c_[east, north]
er = EllipseReg(en)
r = Ransac(er, 0.2)
en_ellipse = r.ransac_filter()
print("-" * 80)
print("Ellipse")
print(f"{en_ellipse.shape[0]} points fit")
# calculate final params
el1 = EllipseReg(en_ellipse)
params = el1.lkn_reg()
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f}")
print(f"RMS: {el1.RMS():.3f}")
# 3D line
param_0 = np.array([100, 200, 300, 0.4356, 0.3245, 0.8396])
t = np.random.rand(n_p) * 100
east = param_0[0] + t * param_0[3]
north = param_0[1] + t * param_0[4]
elev = param_0[2] + t * param_0[5]
north += np.random.rand(n_p) / 3    # add random noise
enz = np.c_[east, north, elev]
lr =Line3dReg(enz)
r = Ransac(lr, 0.2)
enz_line = r.ransac_filter()
print("-" * 80)
print("3D line")
print(f"{enz_line.shape[0]} points fit")
# calculate final params
enz1 = Line3dReg(enz_line)
params = enz1.lkn_reg()
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f} {params[5]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f} {param_0[5]:.3f}")
print(f"RMS: {enz1.RMS():.3f}")
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

# Cylinder surface
# points on axis for each height
axis_points = base_center + np.outer(heights, u0)  # (N,3)

enz = axis_points + (np.cos(angles)[:,None] * radius * v) + \
        (np.sin(angles)[:,None] * radius * w)
# add noise
noise_sigma = 0.01
enz += np.random.normal(scale=noise_sigma, size=enz.shape)
#enz = np.array([[-0.337, -0.516, -2.085],
#                [2.438, 4.302, 0.641],
#                [1.915, 1.146, 2.460],
#                [2.521, 0.307, 0.895],
#                [-0.104, -2.498, -3.092],
#                [0.526, -2.564, -3.100]])
cr =CylinderReg(enz)
r = Ransac(cr, 0.2)
enz_cyl = r.ransac_filter()
print("-" * 80)
print("Cylinder")
print(f"{enz_cyl.shape[0]} points fit")
# calculate final params
cr1 = CylinderReg(enz_cyl)
params = cr1.lkn_reg()
if params is not None:
    print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f} {params[5]:.3f} {params[6]:.3f}")
    print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f} {param_0[5]:.3f} {param_0[6]:.3f}")
    print(f"RMS: {cr1.RMS():.3f}")
else:
    print("Cylinder fit failed")
#res = cr1.dist()
#print("Points X Y Z v")
#for i in range(enz.shape[0]):
#    print(f"{i:5} {enz[i,0]:10.3f} {enz[i,1]:10.3f} {enz[i,2]:10.3f} {res[i]:10.6f}")

# Cone
param_0 = np.array([0, 0, height_max, 0, 0, 1, np.pi/6])
apex = param_0[:3]
v = param_0[3:6]
v = v / np.linalg.norm(v)
alpha = param_0[6]
heights = np.random.rand(n_p) * height_max
angles = np.random.rand(n_p) * 2 * np.pi
r = np.abs(height_max - heights) * np.tan(alpha)
enz = np.array([r * np.cos(angles), r * np.sin(angles), heights])
cr =ConeReg(enz, param_0)
r = Ransac(lr, 0.2)
enz_cone = r.ransac_filter()
print("-" * 80)
print("Cone")
print(f"{enz_line.shape[0]} points fit")
# calculate final params
enz1 = ConeReg(enz_cone, param_0)
params = enz1.lkn_reg()
print(f"Calculated params: {params[0]:.3f} {params[1]:.3f} {params[2]:.3f} {params[3]:.3f} {params[4]:.3f} {params[5]:.3f} {params[6]:.3f}")
print(f"Original   params: {param_0[0]:.3f} {param_0[1]:.3f} {param_0[2]:.3f} {param_0[3]:.3f} {param_0[4]:.3f} {param_0[5]:.3f} {params[6]:.3f}")
print(f"RMS: {enz1.RMS():.3f}")
