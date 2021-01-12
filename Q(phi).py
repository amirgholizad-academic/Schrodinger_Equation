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
def f2(y, x, t, m):
	F2 = -y*(m**2)
	return F2

#the RK4 takes the h, t, N(the number of discretization)and
# coefficient m and initial conditions, then
# returns the solution as two arrays using Runge-Kutta algorithm
def RK4(y0, x0, N, t, h, m):

	y = np.zeros(N + 1)
	x = np.zeros(N + 1)
	y[0] = y0
	x[0] = x0

	for n in range(1,N+1):

		K1y = f1(y[n - 1], x[n - 1], t[n - 1])
		K1x = f2(y[n - 1], x[n - 1], t[n - 1], m)


		K2y = f1(y[n - 1] + K1y * (h / 2), x[n - 1] + K1x * (h / 2), t[n - 1] + h / 2)
		K2x = f2(y[n - 1] + K1y * (h / 2), x[n - 1] + K1x * (h / 2), t[n - 1] + h / 2, m)

		K3y = f1(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2)
		K3x = f2(y[n - 1] + K2y * (h / 2), x[n - 1] + K2x * (h / 2), t[n - 1] + h / 2, m)

		K4y = f1(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h)
		K4x = f2(y[n - 1] + K3y * h, x[n - 1] + K3x * h, t[n - 1] + h, m)


		y[n] = y[n - 1] + (K1y + 2 * K2y + 2 * K3y + K4y) * h / 6
		x[n] = x[n - 1] + (K1x + 2 * K2x + 2 * K3x + K4x) * h / 6

	return [y,x]


#the original ODE is : d^2(Q)/d(phi)^2 = -Q*(m**2)
# by changing the variables we get : d(Q)/d(phi) = x , Q = y , phi = t
# finally we get : d(y)/d(t) = f1 = x , d(x)/d(t) = -y*(m**2) = f2

#take the intial conditions
a = 0
b = 2*math.pi
N = 200
y0 = (1/(2*math.pi))**0.5
x0 = 0
t = Discretizator(N, a, b)[1]
h = Discretizator(N, a, b)[0]

Y = np.zeros((4,N+1))
X = np.zeros((4,N+1))

# to get 4 different plots for m=0,1,2,3
#    the RK4 needs to get variable m as an argument too
for m in range(0,4):
	Y[m] = RK4(y0, x0, N, t, h, m)[0]


# now we plot all 4 graphs
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.size'] = 10
fig = plt.figure(figsize=(6,4))
axes = fig.add_axes([0.1,0.1,0.8,0.8])
line_style = ['-', '--', ':', '-.']

for m in range(0,4):
	axes.plot(t, Y[m], label=fr'$Q_{m}$'+'(\u03C6)', linestyle=line_style[m], color='black')

axes.set_xlabel('\u03C6')
axes.legend(loc=4)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.xticks([0,math.pi/3,math.pi/2,2*math.pi/3,math.pi,4*math.pi/3,3*math.pi/2,2*math.pi],
		   [0,'\u03C0/3','\u03C0/2','2\u03C0/3','\u03C0','4\u03C0/3','3\u03C0/2','2\u03C0'])
plt.yticks([-1,-(1/(2*math.pi))**0.5,0,(1/(2*math.pi))**0.5,1],
		   ["-1","$-(2\u03C0)^{-0.5}$",'0',"$(2\u03C0)^{-0.5}$",'1'])
plt.show()