#Adding necessary libraries
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

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
def f2(y, x, t, k, s, a0):
	F2 = y*((1/(s*a0)**2)-2/(t*a0)+k/(t**2))
	return F2

#the RK4 takes the h, t, N(the number of discretization), coefficients(k, n1234, a0)
# and initial conditions, then returns the solution as two arrays
# using Runge-Kutta algorithm
def RK4(y0, x0, N, t, h, k, n1234, a0):

	y = np.zeros(N + 1)
	x = np.zeros(N + 1)
	y[0] = y0
	x[0] = x0

	for n in range(1,N+1):

		K1y = f1(y[n - 1], x[n - 1], t[n - 1])
		K1x = f2(y[n - 1], x[n - 1], t[n - 1], k, n1234, a0)


		K2y = f1(y[n - 1] + K1y * (h / 2), x[n - 1] + K1x * (h / 2), t[n - 1] + h / 2)
		K2x = f2(y[n - 1] + K1y * (h / 2), x[n - 1] + K1x * (h / 2), t[n - 1] + h / 2, k, n1234, a0)

		K3y = f1(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2)
		K3x = f2(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2, k, n1234, a0)

		K4y = f1(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h)
		K4x = f2(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h, k, n1234, a0)

		y[n] = y[n - 1] + (K1y + 2 * K2y + 2 * K3y + K4y) * h / 6
		x[n] = x[n - 1] + (K1x + 2 * K2x + 2 * K3x + K4x) * h / 6
	#as the original form of the equation was U(r)=r*R(r)
	# we should multiply every y[n] by t[n]
	for p in range(0,len(y)):
		y[p] = y[p] / t[p]

	return [y,x]

#the original ODE is : d^2(U)/d(rho)^2 = U*(k/p**2 + 1/4 - landa/rho) - (2/rho)*d(U)/d(rho)
# by changing the variables we get : d(U)/d(rho) = x , U = y , rho = t
# finally we get :  d(y)/d(t) = f1 = x , d(x)/d(t) = y*(k/t**2 + 1/4 - landa/t) - (2/t)*x = f2

#instead of normal intervals and to be get more efficient plots, we start from 20 and go to 0,0001
a = 20
b = 0.0001
N = 1000
t = Discretizator(N, a, b)[1]
h = Discretizator(N, a, b)[0]

y01 = 0.0000000004
x01 = -0.0000000004
n1 = 1
k1 = 0
a01 = 0.8
Y1 = RK4(y01, x01, N, t, h, k1, n1, a01)[0]


y02 = 0.0002
x02 = -0.0001
n2 = 2
k2 = 2
a02 = 0.8
Y2 = RK4(y02, x02, N, t, h, k2, n2, a02)[0]

y03 = -0.0003
x03= 0.00013
n3 = 2
k3 = 0
a03 = 0.82
Y3 = RK4(y03, x03, N, t, h, k3, n3, a03)[0]

y04 = 0.0085
x04 = -0.0017
n4 = 3
k4 = 0
a04 = 0.5
Y4 = RK4(y04, x04, N, t, h, k4, n4, a04)[0]

# now we plot all 4 graphs
fig = plt.figure(figsize=(6,4))
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.size'] = 10
axes = fig.add_axes([0.1,0.1,0.8,0.8])
line_style = ['-', '--', ':', '-.']

axes.plot(t, Y1, label=fr'$R_{1}^{0}$'+'(r)', linestyle=line_style[0], color='black')
axes.plot(t, Y2, label=fr'$R_{2}^{1}$'+'(r)', linestyle=line_style[1], color='black')
axes.plot(t, Y3, label=fr'$R_{2}^{0}$'+'(r)', linestyle=line_style[2], color='black')
axes.plot(t, Y4, label=fr'$R_{3}^{2}$'+'(r)', linestyle=line_style[3], color='black')

axes.set_xlabel('r')
axes.legend(loc=1)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.xticks([0,2,4,7,10,14,20],
		   [0,"$2a_{0}$","4$a_{0}$","7$a_{0}$","10$a_{0}$","14$a_{0}$","20$a_{0}$"])
plt.ylim(ymax=2)
plt.xlim(xmin=0.0001)
plt.show()