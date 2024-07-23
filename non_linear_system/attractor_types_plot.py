import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 点アトラクタ：安定状態のアトラクタ
def damped_oscillator(t, state):
    x, y = state
    dx = y
    dy = -0.5 * y - x  # 減衰項付きの単純振動
    return [dx, dy]

# 2. リミットサイクル：周期振動のアトラクタ
def van_der_pol(t, state, mu):
    x, y = state
    dx = y
    dy = mu * (1 - x**2) * y - x
    return [dx, dy]

# 3. トーラス：準周期振動のアトラクタ
def torus(t, state, omega1, omega2):
    x, y, z = state
    dx = np.cos(omega1 * t)
    dy = np.sin(omega2 * t)
    dz = np.cos(omega1 * t + omega2 * t)
    return [dx, dy, dz]

# 4. カオスアトラクタ（ストレンジアトラクタ）：カオス振動のアトラクタ
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# 初期条件とパラメータの設定
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# 各システムの初期条件
initial_state_damped = [1.0, 0.0]
initial_state_vdp = [1.0, 0.0]
initial_state_torus = [0.0, 0.0, 0.0]
initial_state_lorenz = [1.0, 1.0, 1.0]

# 時間範囲
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 数値解を求める
solution_damped = solve_ivp(damped_oscillator, t_span, initial_state_damped, t_eval=t_eval)
solution_vdp = solve_ivp(van_der_pol, t_span, initial_state_vdp, args=(1.0,), t_eval=t_eval)

t_span_torus = (0, 50)
t_eval_torus = np.linspace(t_span_torus[0], t_span_torus[1], 1000)
solution_torus = solve_ivp(torus, t_span_torus, initial_state_torus, args=(1.0, np.sqrt(2)), t_eval=t_eval_torus)

t_span_lorenz = (0, 50)
t_eval_lorenz = np.linspace(t_span_lorenz[0], t_span_lorenz[1], 1000)
solution_lorenz = solve_ivp(lorenz, t_span_lorenz, initial_state_lorenz, args=(sigma, rho, beta), t_eval=t_eval_lorenz)

# プロット
fig = plt.figure(figsize=(14, 12))

# 点アトラクタ
ax1 = fig.add_subplot(221)
ax1.plot(solution_damped.y[0], solution_damped.y[1])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Point Attractor: Damped Oscillator')
ax1.grid()

# リミットサイクル
ax2 = fig.add_subplot(222)
ax2.plot(solution_vdp.y[0], solution_vdp.y[1])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Limit Cycle: Van der Pol Oscillator')
ax2.grid()

# トーラス
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(solution_torus.y[0], solution_torus.y[1], solution_torus.y[2])
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Torus: Quasiperiodic Oscillation')

# カオスアトラクタ
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(solution_lorenz.y[0], solution_lorenz.y[1], solution_lorenz.y[2])
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('Lorenz Attractor: Chaotic Oscillation')

plt.tight_layout()
plt.show()
