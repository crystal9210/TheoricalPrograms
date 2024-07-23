import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ローレンツシステムの2つのセットのカオス同期を調べるためのプログラム
# 確認：ローレンツシステム：カオス的な挙動を示す3次元の動的システム
# シミュレーションは時間 t=0から100 まで実行され、10000の評価ポイントを持つ

# データプロットからわかること：k=5の方が収束が早いが、1の場合も各種パラメータの差分Δの数値が時間経過とともに0に収束する
# ポイント：差分が0に収束しない場合、システムは同期しない、収束する場合、同期する

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

# Run simulations and plot delta signals
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

for i, k in enumerate(k_values):
    sol = solve_ivp(lorenz_system, t_span, initial_conditions, args=(sigma, rho, beta, k), t_eval=t_eval)
    x1, y1, z1, x2, y2, z2 = sol.y
    delta_x = x1 - x2
    delta_y = y1 - y2
    delta_z = z1 - z2

    # Plot delta_x
    axs[0, i].plot(t_eval, delta_x, label=f'k = {k}')
    axs[0, i].set_title(f'Delta x(t) for k = {k}')
    axs[0, i].legend()

    # Plot delta_y
    axs[1, i].plot(t_eval, delta_y, label=f'k = {k}')
    axs[1, i].set_title(f'Delta y(t) for k = {k}')
    axs[1, i].legend()

    # Plot delta_z
    axs[2, i].plot(t_eval, delta_z, label=f'k = {k}')
    axs[2, i].set_title(f'Delta z(t) for k = {k}')
    axs[2, i].legend()

plt.tight_layout()
plt.show()
