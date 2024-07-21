import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

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
def plot_lorenz_2d(t, states, r):
    x, y, z = states
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    ax[0].plot(x, y, lw=0.5)
    ax[0].set_title(f'Lorenz Attractor (r={r}) in 2D Projection (X-Y Plane)')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].plot(t, x, lw=0.5)
    ax[1].set_title('X vs Time')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('X')

    # パワースペクトル解析
    N = len(t)
    T = t[1] - t[0]
    yf = fft(x - np.mean(x))
    xf = fftfreq(N, T)[:N//2]
    ax[2].plot(xf, 2.0/N * np.abs(yf[:N//2]))
    ax[2].set_title('Power Spectrum of X')
    ax[2].set_xlabel('Frequency')
    ax[2].set_ylabel('Power')

    plt.tight_layout()
    plt.show()

# 初期パラメータ
sigma = 10.0
b = 8.0 / 3.0

# パラメータ r を準周期が観察される値に設定
r = 21.1

# ローレンツモデルのシミュレーション
t, states = solve_lorenz(sigma, r, b)
plot_lorenz_2d(t, states, r)
