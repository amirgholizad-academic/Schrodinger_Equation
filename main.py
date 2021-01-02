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
	H = (abs(b-a))/N
	for i in range(0,len(T)):
		T[i] = a+(i*H)

	#for j in range(0,len(F)):
	#	F[j] = f1(T[j])

	return [H,T]


def f1(y, x, t):
	F1 = x
	return F1

def f2(y, x, t):
	F2 = 6 * t
	return F2

def RK4(y0,x0,N,t,h):

	y = np.zeros(N + 1)
	x = np.zeros(N + 1)
	y[0] = y0
	x[0] = x0

	for n in range(1,N+1):

		K1y = f1(y[n - 1], x[n - 1], t[n - 1])
		K1x = f2(y[n - 1], x[n - 1], t[n - 1])


		K2y = f1(y[n - 1] + K1y * (h / 2), x[n - 1] + (K1x * (h / 2)), t[n - 1] + (h / 2))
		K2x = f2(y[n - 1] + K1y * (h / 2), x[n - 1] + K1x * (h / 2), t[n - 1] + h / 2)

		K3y = f1(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2)
		K3x = f2(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2)

		K4y = f1(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h)
		K4x = f2(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h)


		y[n] = float(y[n - 1] + (K1y + 2 * K2y + 2 * K3y + K4y) * h / 6)
		x[n] = float(x[n - 1] + (K1x + 2 * K2x + 2 * K3x + K4x) * h / 6)

	return [y,x]

a = 0
b = 8
N = 8
y0 = 0
x0 = 0
t = Discretizator(N, a, b)[1]
h = Discretizator(N, a, b)[0]

Y = RK4(y0,x0,N,t,h)[0]
X = RK4(y0,x0,N,t,h)[1]

print(Y)
plt.plot(t,Y,"o",color = 'black')
plt.show()