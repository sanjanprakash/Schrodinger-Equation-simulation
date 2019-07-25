import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from schrod_class import Schrodinger

#Helper functions for Gaussian wave-packets
def gauss_x(x, a, x0, k0):
    #A Gaussian wave packet of width a, centered at x0, with momentum k0
    return ((a * np.sqrt(np.pi)) ** (-0.5) * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

def gauss_k(k, a, x0, k0):
    #Fourier transform of gauss_x(x)
    return ((a / np.sqrt(np.pi)) ** 0.5 * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))

#Helper function to define the potential barrier
def theta(x):
    #Return 0 if x <= 0, and 1 if x > 0
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y

#The potential barrier
def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))

#Create the animation
#Time steps and duration (to be used as parameters for the FFT and the animation)
dt = 0.01
N_steps = 50
t_max = 120
frames = int(t_max / float(N_steps * dt))

#Constants
hbar = 1.0            #Planck's constant
m = 1.9               #Particle mass

#Range of values of x
N = 2 ** 11
dx = 0.1
x = dx * (np.arange(N) - 0.5 * N)

#Potential specification
V0 = 1.5            #Barrier height
L = hbar / np.sqrt(2 * m * V0)
a = 3 * L           #Barrier width
x0 = -60 * L
V_x = square_barrier(x, a, V0)
V_x[x < -98] = 1E6
V_x[x > 98] = 1E6

#Specify initial momentum and quantities derived from it
p0 = np.sqrt(2 * m * 0.2 * V0)
dp2 = p0 * p0 * 1. / 80
d = hbar / np.sqrt(2 * dp2)

k0 = p0 / hbar
v0 = p0 / m
psi_x0 = gauss_x(x, d, x0, k0)

#Define the Schrodinger object which performs the calculations
S = Schrodinger(x = x, psi_x0 = psi_x0, V_x = V_x, hbar = hbar, m = m, k0 = -28)


#Setting up the plot...
fig = plt.figure()

#Axis limits
xlim = (-100, 100)
klim = (-5, 5)

#First plot (x-space) :
ymin = 0
ymax = V0
ax1 = fig.add_subplot(211, xlim = xlim, ylim = (ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin)))
psi_x_line, = ax1.plot([], [], c = 'r', label = r'$|\psi(x)|$')
V_x_line, = ax1.plot([], [], c = 'k', label = r'$V(x)$')
center_line = ax1.axvline(0, c = 'k', ls = ':', label = r"$x_0 + v_0t$")

title = ax1.set_title("")
ax1.legend(prop = dict(size = 12))
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$|\psi(x)|$')

#Second plot (k-space) :
ymin = abs(S.psi_k).min()
ymax = abs(S.psi_k).max()
ax2 = fig.add_subplot(212, xlim = klim, ylim = (ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin)))
psi_k_line, = ax2.plot([], [], c = 'r', label = r'$|\psi(k)|$')
p0_line1 = ax2.axvline(-p0 / hbar, c = 'k', ls = ':', label = r'$\pm p_0$')
p0_line2 = ax2.axvline(p0 / hbar, c = 'k', ls = ':')
mV_line = ax2.axvline(np.sqrt(2 * V0) / hbar, c = 'k', ls = '--', label = r'$\sqrt{2mV_0}$')

ax2.legend(prop = dict(size = 12))
ax2.set_xlabel('$k$')
ax2.set_ylabel(r'$|\psi(k)|$')

V_x_line.set_data(S.x, S.V_x)             #Feeding the plot with the data

#Functions to help in animation
def init():
    psi_x_line.set_data([], [])
    V_x_line.set_data([], [])
    center_line.set_data([], [])

    psi_k_line.set_data([], [])
    title.set_text("")
    return (psi_x_line, V_x_line, center_line, psi_k_line, title)

def animate(i):
    S.time_step(dt, N_steps)
    psi_x_line.set_data(S.x, 4 * abs(S.psi_x))
    V_x_line.set_data(S.x, S.V_x)
    center_line.set_data(2 * [x0 + S.t * p0 / m], [0, 1])

    psi_k_line.set_data(S.k, abs(S.psi_k))
    title.set_text("t = %.2f" % S.t)
    return (psi_x_line, V_x_line, center_line, psi_k_line, title)

anim = animation.FuncAnimation(fig, animate, init_func = init, frames = frames, interval = 30, blit = True)

plt.show()