# coding: utf-8
import numpy as np
from scipy.linalg import solve_banded
from adi import *

#for pentadiag
d1 = np.array([-5.,-5.,-5.,-5.,-5.])
Q1 = np.array([1.0,2.0,3.0,4.0,5.0])
c1 = np.array([-2.,-2.,-2.,-2.])
a1 = np.array([-2.,-2.,-2.,-2.])
f1 = np.array([-3.,-3.,-3.])
e1 = np.array([-3.,-3.,-3.])


# for banded
d = np.array([-5.,-5.,-5.,-5.,-5.])
Q = np.array([1.0,2.0,3.0,4.0,5.0])
c = np.array([0.,-2.,-2.,-2.,-2.])
a = np.array([-2.,-2.,-2.,-2.,0,])
f = np.array([0.0, 0.0, -3.,-3.,-3.])
e = np.array([-3.,-3.,-3.0,0,0.0])


print("pentadiag algorithm")
print(pentadiag(d1,f1,c1,a1,e1,Q1))


print("banded solver")
ab = np.matrix([f,c, d, a, e])                  # simplified matrix
ans = solve_banded((2, 2), ab, Q)
print(ans) 
#print("should be")
#print([-0.621,-0.174,0.818,-0.508,-1.288])

