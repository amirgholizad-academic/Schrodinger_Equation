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
	#	F[j] = f1(T[j])

	return [H,T]


def f1(y, x, t):
	F1 = -4*t*math.exp(-2*(t**2))
	return F1

def f2(y, x, t):
	F2 = 4*(t**2-1)*y
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

a1 = 0
b1 = 5
N1 = 50
y01 = 10
x01 = 0
t1 = Discretizator(N1, a1, b1)[1]
h1 = Discretizator(N1, a1, b1)[0]

Y1 = RK4(y01,x01,N1,t1,h1)[0]
Y1 = np.delete(Y1,0)
X1 = RK4(y01,x01,N1,t1,h1)[1]
t1 = np.delete(t1,0)

a2 = 0
b2 = -5
N2 = 50
y02 = 10
x02 = 0
t2 = Discretizator(N2, a2, b2)[1]
h2 = Discretizator(N2, a2, b2)[0]

Y2 = RK4(y02,x02,N2,t2,h2)[0]
X2 = RK4(y02,x02,N2,t2,h2)[1]
Y2 = Y2[::-1]
t2 = t2[::-1]

Y = np.concatenate((Y2,Y1))
T = np.concatenate((t2,t1))

print(Y)
plt.plot(T,Y,"o",color = 'black')
plt.show()