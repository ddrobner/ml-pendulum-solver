# this is a script to numerically solve ODES and then save that data
# we'll attempt to create a model which estimates period from the length of the pendulum
# the bit that is interesting is that the ODE is nonlinear so there is no analytic solution to the ODE
# of course, there are plenty of numerical ODE solvers out there which can do this easily but this is mainly just to do something cool with tensorflow

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# we'll have a dataset of size approx. 2000 elements
# should be enough for one parameter
length = np.arange(0.1, 100, 0.1)
g = 9.81 # gravity...

# define our function which returns dy/dt
def dy(y, t, length_g):
    theta, omega = y
    dy = [omega, -length_g*np.sin(theta)]
    return dy

# set up our sample space
t = np.linspace(0, 60, 8000)
y_0 = [np.pi-0.1, 0]
"""
#this block was testing the method I was using

sol = odeint(dy, y_0, t, args=(0.23513,))

# perform a fft to find the period
ft = np.fft.rfft(sol[:, 0])
freqs = np.fft.rfftfreq(len(sol[:, 0]), t[1]-t[0])
mags = abs(ft)
"""
# ok... we can do it for a single omega
# let's generate a dataset now
periods = np.empty(len(length))

k = 0
for l in length:
    sol_l = odeint(dy, y_0, t, args=((g/l),), hmax=0.001)
    fft = np.fft.rfft(sol_l[:, 0])
    freq = np.fft.rfftfreq(len(sol_l[:, 0]), t[1]-t[0])
    mag = abs(fft)
    periods[k] = 1/freq[mag.argmax()]
    k += 1

out = pd.DataFrame({"Length" : length, "Period": periods})
out.to_csv("shm_data.csv", index=False)
