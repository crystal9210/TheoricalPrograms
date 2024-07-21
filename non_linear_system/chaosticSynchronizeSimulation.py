import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Synchronization strength
k_values = [5, 1]

# Initial conditions
initial_conditions = [1.0, 1.0, 1.0, 1.1, 1.1, 1.1]

# Define the Lorenz system with synchronization feedback
def lorenz_system(t, state, sigma, rho, beta, k):
    x1, y1, z1, x2, y2, z2 = state
    dx1dt = sigma * (y1 - x1)
    dy1dt = x1 * (rho - z1) - y1
    dz1dt = x1 * y1 - beta * z1
    dx2dt = sigma * (y2 - x2) + k * (x1 - x2)
    dy2dt = x2 * (rho - z2) - y2 + k * (y1 - y2)
    dz2dt = x2 * y2 - beta * z2 + k * (z1 - z2)
    return [dx1dt, dy1dt, dz1dt, dx2dt, dy2dt, dz2dt]

# Time span for simulation
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)

# Run simulations and plot results
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, k in enumerate(k_values):
    sol = solve_ivp(lorenz_system, t_span, initial_conditions, args=(sigma, rho, beta, k), t_eval=t_eval)
    x1, y1, z1, x2, y2, z2 = sol.y

    # Plot x1 and x2
    axs[i, 0].plot(t_eval, x1, label='x1')
    axs[i, 0].plot(t_eval, x2, label='x2')
    axs[i, 0].set_title(f'k = {k}: x1 and x2')
    axs[i, 0].legend()

    # Plot y1 and y2
    axs[i, 1].plot(t_eval, y1, label='y1')
    axs[i, 1].plot(t_eval, y2, label='y2')
    axs[i, 1].set_title(f'k = {k}: y1 and y2')
    axs[i, 1].legend()

    # Plot z1 and z2
    axs[i, 2].plot(t_eval, z1, label='z1')
    axs[i, 2].plot(t_eval, z2)
    axs[i, 2].set_title(f'k = {k}: z1 and z2')
    axs[i, 2].legend()

plt.tight_layout()
plt.show()
