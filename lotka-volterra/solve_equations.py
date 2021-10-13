"""Credits to:
    https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def dX_dt(X, t=0):
    """Returns the growth rate of bears and fishes."""

    res = np.array([
        a * X[0] - b * X[0] * X[1],
        (c * X[0] - d) * X[1] ])
    return res

# Set seaborn theme.
sns.set_theme(style='darkgrid')

# Parameters.
a = 1.
b = 0.1
c = 0.075
d = 1.5

# Equilibrium points.
X_f0 = np.array([0. , 0.])
X_f1 = np.array([d / c, a / b])

# Time.
t = np.linspace(0, 15,  1000)

# Initials conditions.
X0 = np.array([10, 5])

# Integrate the system.
X = integrate.odeint(dX_dt, X0, t)
fishes, bears = X.T

#
# Plot the timeline of populations.
#
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
plt.plot(t, fishes, 'r-', label = 'Fishes')
plt.plot(t, bears, 'b-', label = 'Bears')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Time evolution')
fig.savefig('populations.png')


#
# Plot the phase plot.
#
fig, ax = plt.subplots(figsize = a4_dims)

# Scale factor for initial conditions.
values  = np.linspace(0.2, 1, 5)

# Colors for each trajectory.
vcolors = plt.cm.autumn_r(np.linspace(0.2, 1., len(values)))

# Plot all the trajectories.
for value, col in zip(values, vcolors):

    # Solve a new system.
    X0 = value * X_f1
    X = integrate.odeint(dX_dt, X0, t)

    # Plot the solution.
    plt.plot( X[:,0], X[:,1], lw = 3.5 * value, color = col,
        label = f'X0=({X0[0]:.2f}, {X0[1]:.2f})')

# Define a grid and compute direction for each point.
n_points = 20
ymax = plt.ylim(ymin = 0)[1]
xmax = plt.xlim(xmin = 0)[1]
x = np.linspace(0, xmax, n_points)
y = np.linspace(0, ymax, n_points)
X1, Y1 = np.meshgrid(x, y)

# Growth rate for the whole grid.
DX1, DY1 = dX_dt([X1, Y1])

# Normalize and stabilize.
M = np.hypot(DX1, DY1)
M[M == 0] = 1.
DX1 /= M
DY1 /= M


plt.title('Phase plot')
Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot = 'mid', cmap = plt.cm.jet)
plt.xlabel('Fishes')
plt.ylabel('Bears')
plt.legend()
plt.grid()
plt.xlim(0, xmax)
plt.ylim(0, ymax)
fig.savefig('phases.png')
