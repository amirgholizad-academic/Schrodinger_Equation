#Adding necessary libraries
import numpy as np
#import panda as pd
import scipy as sp
import matplotlib as mp
import decimal
import math

def f(t):
    return math.cos(t)

def Discretizator(N, a, b):

	F = np.zeros(N+1)
	T = np.zeros(N+1)
	H = (b-a)/N
	for i in range(0,len(T)):
		T[i] = a+(i*H)

	for j in range(0,len(F)):
		F[j] = f(T[j])

	return [H,T,F]

"""
def RK4():

    K1 = f(y[n] , t[n])
    K2 = f(y[n] + K1*(h/2) , t[n] + h/2)
    K3 = f(y[n] + K2*(h/2) , t[n] + h/2)
    K4 = f(y[n] + K3*h , t[n] + h)
"""

arrays = Discretizator(10, -math.pi, math.pi)
h = arrays[0]
t = arrays[1]
f = arrays[2]

print(h)
