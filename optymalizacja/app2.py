import numpy as np
from scipy.optimize import minimize
import random

def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

#nwm czy klimczak nie pisal zeby byla 10 elementowa
#x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
vector_size = 1
x0 = np.random.rand(vector_size, 10)
print(x0)

res = minimize(rosen, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x)

res2 = minimize(rosen, x0, method='powell', options={'xatol': 1e-8, 'disp': True})
print(res2.x)


res3 = minimize(rosen, x0, method='CG', options={'xatol': 1e-8, 'disp': True})
print(res3.x)
