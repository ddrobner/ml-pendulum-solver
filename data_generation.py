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
length = np.arange(1, 20, 0.1)
g = 9.81 # gravity...

# define our function which returns dy/dt
def dy(y, t, length_g):
    theta, omega = y
    dy = [omega, -length_g*np.sin(theta)]
    return dy

# set up our sample space
sample_rate = 0.01
t_max = 60
t = np.linspace(0, t_max, int(t_max/sample_rate))
y_0 = [np.pi-0.1, 0]

#this block was testing the method I was using
sol = odeint(dy, y_0, t, args=((g/1),))

"""
# perform a fft to find the period
ft = np.fft.rfft(sol[:, 0])
freqs = np.fft.rfftfreq(len(sol[:, 0]), t[1]-t[0])
mags = abs(ft)

theta_t = sol[:, 0]

# using autocorrelation to find the period of the solution
acf = np.correlate(theta_t, theta_t, 'full')[-len(theta_t):]

inflection = np.diff(np.sign(np.diff(acf)))
peaks = (inflection < 0).nonzero()[0] + 1
delay = peaks[acf[peaks].argmax()]
print(t[delay]-t[0])


"""
# ok... we can do it for a single omega
# let's generate a dataset now
periods = np.empty(len(length))

k = 0
for l in length:
    # solve ode
    sol_l = odeint(dy, y_0, t, args=((g/l),))
    theta_t = sol_l[:, 0]
    # autocorrelate
    acf = np.correlate(theta_t, theta_t, 'full')[-len(theta_t):]
    inflection = np.diff(np.sign(np.diff(acf)))
    peaks = (inflection < 0).nonzero()[0] + 1
    delay = peaks[acf[peaks].argmax()]
    periods[k] = (t[delay]-t[0])
    k += 1
out = pd.DataFrame({"Length" : length, "Period": periods})
out.to_csv("shm_data.csv", index=False)