from ransac import Ransac
from regression import LinearReg, CircleReg, SphereReg
import numpy as np
from matplotlib import pyplot as plt

max_coo = 1000  # coordinate range [0..max_coo]
n_p = 200       # number of points
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
elev += np.random.rand(n_p)    # add random noise
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
