# this is a script to numerically solve ODES and then save that data
# we'll attempt to create a model which estimates period from the length of the pendulum
# the bit that is interesting is that the ODE is nonlinear so there is no analytic solution to the ODE
# of course, there are plenty of numerical ODE solvers out there which can do this easily but this is mainly just to do something cool with tensorflow

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# we'll have a dataset of size approx. 5000 elements
# should be enough for one parameter
length = np.arange(0.01, 50.0, 0.01)
g = 9.81 # gravity...

def dtheta(y, t, length_g):
    theta, omega = y
    dy = [omega, -length_g*np.sin(theta)]
    return dy

t = np.linspace(0, 20, 1000)
y_0 = [np.pi-0.1, 0]
sol = odeint(dtheta, y_0, t, args=(10,))

#plt.plot(t, sol[:, 0])
#plt.show()

ft = np.fft.rfft(sol[:, 0])
freqs = np.fft.rfftfreq(len(sol[:, 0]), t[1]-t[0])
mags = abs(ft)

print(1/freqs[mags.argmax()])

plt.plot(t, sol[:, 0])
plt.show()
