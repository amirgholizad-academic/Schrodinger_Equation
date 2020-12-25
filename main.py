#Adding necessary libraries
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import decimal
import math


def Discretizator(N, a, b):

	#F = np.zeros(N+1)
	T = np.zeros(N+1)
	H = (b-a)/N
	for i in range(0,len(T)):
		T[i] = a+(i*H)

	#for j in range(0,len(F)):
	#	F[j] = f(T[j])

	return [H,T]


def f(y,t):
    return 2*t*y

def RK4(y0,N,t,h):

	y = np.zeros(N+1)
	y[0] = y0
	for n in range(1,N+1):
		K1 = f(y[n-1] , t[n-1])
		K2 = f(y[n-1] + K1*(h/2) , t[n-1] + h/2)
		K3 = f(y[n-1] + K2*(h/2) , t[n-1] + h/2)
		K4 = f(y[n-1] + K3*h , t[n-1] + h)
		y[n] = float(y[n-1] + (K1+2*K2+2*K3+K4)*h/6)

	return y

a = 0
b = 10
N = 100
y0 = 3
t = Discretizator(N, a, b)[1]
h = Discretizator(N, a, b)[0]

Y = RK4(y0,N,t,h)

print(Y)
plt.plot(t,Y,"o",color = 'black')
plt.show()