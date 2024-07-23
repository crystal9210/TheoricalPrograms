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

# パラメータの設定
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# 初期条件
initial_state = [1.0, 1.0, 1.0]

# 時間範囲の設定
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# 数値解を求める
solution = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)

# ローレンツアトラクタをプロットするコード
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# プロット
ax.plot(solution.y[0], solution.y[1], solution.y[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')

plt.show()
