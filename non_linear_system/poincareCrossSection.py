import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# ローレンツアトラクタ
# ローレンツアトラクター＞カオス的なシステムの代表例、そのフラクタル次元は約2.06から2.1とされており、非常に複雑な軌跡を持つ。位相空間内の軌跡は一見同じ移送状態を取ると思えるが、細かいところまで見ると同じ状態はとらない
# 位相空間での軌跡は、異なるスケール(全体と部分で相似的な形が見られるということ)でも自己相似的なパターンを示します。

# ヘノンマップ
# ヘノンマップもカオス的な挙動を示すことが知られています。そのフラクタル次元は約1.2から1.3であり、シンプルな2次元マップながら複雑なカオス軌跡を持ちます。

# カオスの性質
# 複雑性：軌跡は長期的には予測が困難
# 非周期性：システムは同じ状態には戻らず常に新しいパターンを描く
# フラクタル構造：異なるスケールで自己相似性を示し、軌跡がフラクタル構造(自己相似性を示す軌跡構造)を示す
# 初期値鋭敏性：初期値のわずかな差異が時間経過とともに指数関数的にシステムの挙動に大きな差異を生む

def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def henon_map(n, a=1.4, b=0.3):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = 0.1, 0.1
    for i in range(1, n):
        x[i] = 1 - a * x[i-1]**2 + y[i-1]
        y[i] = b * x[i-1]
    return x, y

# 初期条件とパラメータの設定
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# ローレンツアトラクタの数値解を求める
solution = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)

# ヘノンマップの数値解を求める
n_points = 10000
x_henon, y_henon = henon_map(n_points)

# 位相空間とポアンカレ断面のプロット
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# ローレンツアトラクタ位相空間プロット
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(solution.y[0], solution.y[1], solution.y[2])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Lorenz Attractor Phase Space')

# ローレンツアトラクタポアンカレ断面プロット
z_section = 27
tolerance = 0.5
indices = np.where((solution.y[2] > z_section - tolerance) & (solution.y[2] < z_section + tolerance))[0]
ax2 = axes[0, 1]
ax2.scatter(solution.y[0][indices], solution.y[1][indices], c=solution.y[2][indices], cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(f'Lorenz Attractor Poincaré Section at Z={z_section}')

# ヘノンマップ位相空間プロット
ax3 = axes[1, 0]
ax3.plot(x_henon, y_henon, 'bo', markersize=0.5)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Henon Map Phase Space')

# ヘノンマップポアンカレ断面プロット
ax4 = axes[1, 1]
ax4.plot(x_henon[1:], x_henon[:-1], 'bo', markersize=0.5)
ax4.set_xlabel('X(n)')
ax4.set_ylabel('X(n+1)')
ax4.set_title('Henon Map Poincaré Section')

plt.tight_layout()
plt.show()

# フラクタル次元の計算
def box_counting(points, box_size):
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    bins_x = np.arange(min_x, max_x, box_size)
    bins_y = np.arange(min_y, max_y, box_size)
    count = 0

    for bx in bins_x:
        for by in bins_y:
            if np.any((points[:, 0] >= bx) & (points[:, 0] < bx + box_size) & (points[:, 1] >= by) & (points[:, 1] < by + box_size)):
                count += 1

    return count

# ローレンツアトラクタのポアンカレ断面の点の分布を計算
points_lorenz = np.vstack((solution.y[0][indices], solution.y[1][indices])).T
box_sizes = np.logspace(-2, 0, num=10)
counts_lorenz = [box_counting(points_lorenz, size) for size in box_sizes]

# ヘノンマップの点の分布を計算
points_henon = np.vstack((x_henon[1:], x_henon[:-1])).T
counts_henon = [box_counting(points_henon, size) for size in box_sizes]

# フラクタル次元のプロット
plt.figure(figsize=(14, 7))

plt.subplot(121)
plt.plot(np.log(1/box_sizes), np.log(counts_lorenz), 'bo-')
plt.xlabel('log(1/Box size)')
plt.ylabel('log(Count)')
plt.title('Box-counting method for Lorenz Attractor Poincaré Section')

plt.subplot(122)
plt.plot(np.log(1/box_sizes), np.log(counts_henon), 'ro-')
plt.xlabel('log(1/Box size)') # 箱のサイズの逆数の対数、箱のサイズが小さくなるにつれて値としては大きくなる
plt.ylabel('log(Count)') # 各箱サイズで少なくとも一つのデータポイントが含まれる箱の数の対数を示す
plt.title('Box-counting method for Henon Map Poincaré Section')

plt.tight_layout()
plt.show()

# フラクタル次元の計算
coefficients_lorenz = np.polyfit(np.log(1/box_sizes), np.log(counts_lorenz), 1)
fractal_dimension_lorenz = coefficients_lorenz[0]
print(f'Fractal Dimension (Lorenz): {fractal_dimension_lorenz}')

coefficients_henon = np.polyfit(np.log(1/box_sizes), np.log(counts_henon), 1)
fractal_dimension_henon = coefficients_henon[0]
print(f'Fractal Dimension (Henon): {fractal_dimension_henon}')

# カオスの判定
threshold = 1.2  # フラクタル次元の閾値（例）
if fractal_dimension_lorenz > threshold:
    print("The Lorenz system exhibits chaotic behavior.")
else:
    print("The Lorenz system does not exhibit chaotic behavior.")

if fractal_dimension_henon > threshold:
    print("The Henon map exhibits chaotic behavior.")
else:
    print("The Henon map does not exhibit chaotic behavior.")
