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

#f2 is the right hand side of the second equation
def f2(y, x, t, k, m):
	F2 = y*( (m/math.sin(t))**2 - k ) - x/math.tan(t)
	return F2

#the RK4 takes the h, t, N(the number of discretization), coefficients(k,m) and
# initial conditions, then returns the solution as two arrays
# using Runge-Kutta algorithm
def RK4(y0,x0,N,t,h,k,m):

	y = np.zeros(N + 1)
	x = np.zeros(N + 1)
	y[0] = y0
	x[0] = x0

	for n in range(1,N+1):

		K1y = f1(y[n - 1], x[n - 1], t[n - 1])
		K1x = f2(y[n - 1], x[n - 1], t[n - 1], k, m)


		K2y = f1(y[n - 1] + K1y * (h / 2), x[n - 1] + K1x * (h / 2), t[n - 1] + h / 2)
		K2x = f2(y[n - 1] + K1y * (h / 2), x[n - 1] + K1x * (h / 2), t[n - 1] + h / 2, k, m)

		K3y = f1(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2)
		K3x = f2(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2, k, m)

		K4y = f1(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h)
		K4x = f2(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h, k, m)


		y[n] = y[n - 1] + (K1y + 2 * K2y + 2 * K3y + K4y) * h / 6
		x[n] = x[n - 1] + (K1x + 2 * K2x + 2 * K3x + K4x) * h / 6

	return [y,x]

#the original ODE is : d^2(P)/d(theta)^2 = P( (m/sin(theta))**2 - k ) - cot(theta)*d(P)/d(theta)
# by changing the variables we get : d(P)/d(theta) = x , P = y , theta = t
# finally we get :  d(y)/d(t) = f1 = x , d(x)/d(t) = y( (m/sin(t))**2 - k ) - x/tan(t) = f2

a = math.pi/6
b = math.pi
N = 1000

t = Discretizator(N, a, b)[1]
h = Discretizator(N, a, b)[0]


# to get 4 different plots for m=0,1,2,3
# the RK4 needs to get variable m and k as arguments too
y01 = math.sqrt(1/2)
x01 = 0
k1 = 0
m1 = 0
Y1 = RK4(y01, x01, N, t, h, k1, m1)[0]

y02 = 0.43
x02 = 0.75
k2 = 2
m2 = -1
Y2 = RK4(y02, x02, N, t, h, k2, m2)[0]

y03 = -0.84
x03 = -0.97
k3 = 6
m3 = 1
Y3 = RK4(y03, x03, N, t, h, k3, m3)[0]

y04 = 0.61
x04 = -3.85
k4 = 12
m4 = 0
Y4 = RK4(y04, x04, N, t, h, k4, m4)[0]


# now we plot all 4 graphs
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.size'] = 10
fig = plt.figure(figsize=(6,4))
axes = fig.add_axes([0.1,0.1,0.8,0.8])
line_style = ['-', '--', ':', '-.']

axes.plot(t, Y1, label='P'+f'$^{0}_{0}$'+'(\u03B8)', linestyle=line_style[0], color='black')
axes.plot(t, Y2, label='P'+"\u207B"+f'$^{1}_{1}$'+'(\u03B8)', linestyle=line_style[1], color='black')
axes.plot(t, Y3, label='P'+f'$^{1}_{2}$'+'(\u03B8)', linestyle=line_style[2], color='black')
axes.plot(t, Y4, label='P'+f'$^{0}_{3}$'+'(\u03B8)', linestyle=line_style[3], color='black')

axes.set_xlabel('\u03B8')
axes.legend(loc=3)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.xticks([0,math.pi/6,math.pi/4,1,math.pi/2,2,math.pi-0.011],[0,'\u03C0/6','\u03C0/4','1','\u03C0/2','2','\u03C0'])
plt.ylim(ymin=-2, ymax=1.5)
plt.xlim(xmin=0, xmax=math.pi-0.011)
plt.show()