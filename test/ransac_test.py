from ransac import Ransac
from regression import LinearReg, CircleReg
import numpy as np

max_coo = 1000  # coordinate range [..max_coo]
n_p = 200       # number of points
# test for plane
param_0 = np.array([0.57735027, 0.80829038, 0.11547005, 12.45])
east = np.random.rand(n_p) * max_coo
north = np.random.rand(n_p) * max_coo
elev = -(param_0[0] * east + param_0[1] * north + param_0[3]) / param_0[2]
enz = np.c_[east, north, elev]
#enz = np.array([[1, -3, 9], [5, 7, -2], [3, 9, 2], [5, 7, 8]])
lr = LinearReg(enz)
r = Ransac(lr, 1.0)
enz_plane = r.ransac_filter()
print(f"{enz_plane.shape[0]} points fit")
# calculate final params
lr1 = LinearReg(enz_plane)
params = lr1.lkn_reg()
print(f"params: {params}")
print(f"params: {param_0}")
# test for line

