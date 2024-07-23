import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# solve_ivp関数を用いて時間t=1から100までの、二つのローレンツシステムの、二つの同期強度k(=1,5)に対するシミュレーションを実行し
# カオス同期が成立するか確認するためのプログラムー＞二つの図において、それぞれプロットされているがそれぞれどのような挙動からどういう性質が言えるのかがわからない
# 条件付きリアプノフ指数をプロットする->k=5のとき、システムは強く同期する、k=1のときもlog_delta_xが時間と共に減少し、同期することが期待される

# 一般的な事実：条件付きリアプノフ指数が負の値を取るときシステムは同期する

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Synchronization strength
k_values = [5, 1]

# Initial conditions | システム1：各数値：1.0、システム2；各数値：1.1
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

# Function to estimate the conditional Lyapunov exponent
def estimate_lyapunov(t_eval, delta):
    log_delta = np.log(np.abs(delta))
    poly_fit = np.polyfit(t_eval, log_delta, 1)
    return poly_fit[0]

# Run simulations and plot Lyapunov exponents
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

for i, k in enumerate(k_values):
    sol = solve_ivp(lorenz_system, t_span, initial_conditions, args=(sigma, rho, beta, k), t_eval=t_eval)
    x1, y1, z1, x2, y2, z2 = sol.y
    delta_x = x1 - x2

    lyapunov_exp = estimate_lyapunov(t_eval, delta_x)
    print(f'Estimated conditional Lyapunov exponent for k = {k}: {lyapunov_exp}')

    # Plot log(delta_x)
    log_delta_x = np.log(np.abs(delta_x) + 1e-10)  # Add epsilon to avoid log(0) ー＞ 差分が0になる場合に対応する＋np.absでlogに入る数値が負にならないように調整する
    axs[i].set_title(f'Log(Delta x(t)) for k = {k}')
    axs[i].plot(t_eval, log_delta_x, label=f'k = {k}')
    axs[i].set_title(f'Log(Delta x(t)) for k = {k}')
    axs[i].legend()

plt.tight_layout()
plt.show()
