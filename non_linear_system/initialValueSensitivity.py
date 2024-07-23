import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ローレンツ方程式の定義
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

initial_state_1 = [1.0, 1.0, 1.0]
initial_state_2 = [1.001, 1.0, 1.0]  # わずかに異なる初期条件

t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# 数値解を求める
solution_1 = solve_ivp(lorenz, t_span, initial_state_1, args=(sigma, rho, beta), t_eval=t_eval)
solution_2 = solve_ivp(lorenz, t_span, initial_state_2, args=(sigma, rho, beta), t_eval=t_eval)

# プロット
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(solution_1.t, solution_1.y[0], label='x1(t)')
plt.plot(solution_2.t, solution_2.y[0], label='x2(t)')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.title('Comparison of X(t) for Two Initial Conditions')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(solution_1.t, np.abs(solution_1.y[0] - solution_2.y[0]), label='|x1(t) - x2(t)|')
plt.xlabel('Time')
plt.ylabel('Difference in X(t)')
plt.title('Difference in X(t) over Time')
plt.legend()

plt.tight_layout()
plt.show()
