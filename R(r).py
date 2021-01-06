#Adding necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

#Taking an interval of [a,b] for t and discretizating it to N+1 numbers
def Discretizator(N, a, b):

	T = np.zeros(N+1)
	#T is an array of numbers for values that t is allowed to have
	H = (b-a)/N
	#H is the difference between two numbers inside T
	for i in range(0,len(T)):
		T[i] = a+(i*H)

	return [H,T]

#f1 is the righ hand side of the first equation
def f1(y, x, t):
	F1 = x
	return F1

#f2 is the righ hand side of the second equation
def f2(y, x, t):
	#V = -14.39998
	#M = 1 / 18*(10**(-16))
	#h = 7.6199682*M*(10**(-20))
	n = 2
	#E = -13.6056 / (n ** 2)
	k = 0
	c = 9
	a0 = (1/2)
	landa = 1/(n*a0)
	F2 = y*((1/(n*a0)**2)-2/(t*a0)+k/(t**2))
	return F2

#the RK4 takes the h, t, N(the number of discretization)
# and initial conditions and returns the solution as two arrays
# using Runge-Kutta algorithm
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

		y[n] = y[n - 1] + (K1y + 2 * K2y + 2 * K3y + K4y) * h / 6
		x[n] = x[n - 1] + (K1x + 2 * K2x + 2 * K3x + K4x) * h / 6
	#as the original form of the equation was U(r)=r*R(r)
	# we should multiply every y[n] by t[n]
	for p in range(0,len(y)):
		y[n] = y[n] / t[n]

	return [y,x]

#the original ODE is : d^2(U)/d(rho)^2 = U*(k/p**2 + 1/4 - landa/rho) - (2/rho)*d(U)/d(rho)
# by changing the variables we get : d(U)/d(rho) = x , U = y , rho = t
# finally we get :  d(y)/d(t) = f1 = x , d(x)/d(t) = y*(k/t**2 + 1/4 - landa/t) - (2/t)*x = f2

#because the given interval for t = rho contains negative numbers we are going to need to
# spilit it to two arrays and then concatenate them
a = 0.000000000001
b = 5
N = 1000
a0 = (1/2)
y0 = 0
x0 = a0 ** (-3 / 2)
t = Discretizator(N, a, b)[1]
h = Discretizator(N, a, b)[0]

Y = RK4(y0, x0, N, t, h)[0]
X = RK4(y0, x0, N, t, h)[1]


print(Y)
plt.plot(t, Y, "o", color ='black')
plt.show()