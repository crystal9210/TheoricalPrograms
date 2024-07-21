import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ローレンツモデルの微分方程式
def lorenz(t, state, sigma, r, b):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (r - z) - y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

# ローレンツモデルを数値的に解く関数
def solve_lorenz(sigma, r, b, initial_state=[1.0, 1.0, 1.0], t_max=100.0, dt=0.01):
    t_span = [0, t_max]
    t_eval = np.arange(0, t_max, dt)
    solution = solve_ivp(lorenz, t_span, initial_state, args=(sigma, r, b), t_eval=t_eval)
    return solution.t, solution.y

# プロット関数
def plot_lorenz(t, states, r):
    x, y, z = states
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(221, projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_title(f'Lorenz Attractor (r={r})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax = fig.add_subplot(222)
    ax.plot(t, x, lw=0.5)
    ax.set_title('X vs Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('X')

    ax = fig.add_subplot(223)
    ax.plot(t, y, lw=0.5)
    ax.set_title('Y vs Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Y')

    ax = fig.add_subplot(224)
    ax.plot(t, z, lw=0.5)
    ax.set_title('Z vs Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Z')

    plt.tight_layout()

# 初期パラメータ
sigma = 10.0
b = 8.0 / 3.0

# パラメータrを変化させながらシミュレーション
r_values = [10, 23.74, 28, 35, 40] # 代表的なrの値を用いている

for r in r_values:
    t, states = solve_lorenz(sigma, r, b)
    plot_lorenz(t, states, r)

plt.show() # すべてのプロットを同時に表示
