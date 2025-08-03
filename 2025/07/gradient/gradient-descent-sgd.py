# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# 訓練資料：新增三筆資料
# 原始資料：x1=1, x2=1, y=4
# 新增資料1：x1=2, x2=1, y=6
# 新增資料2：x1=1, x2=2, y=6
X = np.array([[1, 1], [2, 1], [1, 2]])  # 第一筆資料  # 第二筆資料  # 第三筆資料
y = np.array([4, 6, 6])  # 對應的目標值


# 定義預測函數：y_hat = w1 * x1 + w2 * x2
def predict(w, X):
    """預測函數"""
    return X @ w


# 定義損失函數：MSE (Mean Squared Error)
def loss(w):
    """計算所有訓練資料的 MSE 損失（用於視覺化）"""
    y_pred = predict(w, X)
    return np.mean((y - y_pred) ** 2)


def loss_single(w, x_i, y_i):
    """計算單筆資料的 MSE 損失（SGD 使用）"""
    y_pred = w @ x_i
    # print(f"w: {w}, x_i: {x_i}, y_i: {y_i}, y_pred: {y_pred}")
    return (y_i - y_pred) ** 2


# 計算單筆資料的梯度：SGD 使用隨機選擇的一筆資料
def compute_sgd_gradient(w, x_i, y_i):
    """計算單筆資料的梯度（SGD）"""
    y_pred = w @ x_i
    error = y_i - y_pred
    # 梯度 = -2 * error * x_i
    grad = -2 * error * x_i
    return grad


# 初始點
w = np.array([0.0, 0.0])
lr = 0.1
path = [w.copy()]

# 設定隨機種子以確保結果可重現
np.random.seed(42)

# 預先計算所有步驟（SGD：前三筆固定使用 data 1, 2, 3，之後隨機）
num_iterations = 10
for i in range(num_iterations):
    # 前三筆固定使用 data 1, 2, 3
    if i < 3:
        idx = i  # 固定順序：0, 1, 2
    else:
        # 之後隨機選擇一筆訓練資料
        idx = np.random.randint(0, len(X))

    x_i = X[idx]
    y_i = y[idx]

    # 使用選中的資料計算梯度
    grad = compute_sgd_gradient(w, x_i, y_i)
    w = w - lr * grad
    path.append(w.copy())

path = np.array(path)

# 設置圖形
fig, ax = plt.subplots(figsize=(12, 8))

# 準備網格數據
w1_vals = np.linspace(-1, 4.5, 100)
w2_vals = np.linspace(-1, 4.5, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = np.zeros_like(W1)

# 計算網格上每個點的損失
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w_grid = np.array([W1[i, j], W2[i, j]])
        Z[i, j] = loss(w_grid)

# 初始化熱度圖
heatmap = ax.imshow(
    Z,
    extent=[w1_vals.min(), w1_vals.max(), w2_vals.min(), w2_vals.max()],
    origin="lower",
    cmap="viridis",
    aspect="auto",
)
plt.colorbar(heatmap, label="Loss Value")

# 添加等高線
contours = ax.contour(W1, W2, Z, levels=15, colors="white", alpha=0.7, linewidths=0.8)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

# 初始化繪圖元素
(line,) = ax.plot([], [], "r--", linewidth=2, alpha=0.8)
(point,) = ax.plot([], [], "ro", markersize=8, zorder=5)
start_point = ax.scatter([], [], color="blue", s=100, label="Start (0, 0)", zorder=5)
current_point = ax.scatter([], [], color="red", s=100, label="Current Step", zorder=5)
step_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

ax.set_xlabel("$w_1$")
ax.set_ylabel("$w_2$")
ax.legend()
ax.grid(True, alpha=0.3)

# 設置軸範圍
ax.set_xlim(w1_vals.min(), w1_vals.max())
ax.set_ylim(w2_vals.min(), w2_vals.max())


def animate(frame):
    """動畫函數"""
    if frame == 0:
        # 第一步：只顯示藍色起點，不顯示紅點
        line.set_data([], [])
        point.set_data([], [])
        start_point.set_offsets([path[0, 0], path[0, 1]])
        current_point.set_offsets([np.nan, np.nan])  # 隱藏紅點

        # 顯示初始數據
        current_w = path[0]
        current_loss = loss(current_w)  # 顯示所有資料的損失
        text = "Iteration 0 (SGD):\nw1={:.4f}, w2={:.4f}\nAll data Loss={:.4f}\nNo gradient yet".format(
            current_w[0], current_w[1], current_loss
        )
        step_text.set_text(text)

    else:
        # 顯示到當前步驟的路徑
        current_path = path[: frame + 1]
        line.set_data(current_path[:, 0], current_path[:, 1])
        point.set_data(current_path[:, 0], current_path[:, 1])
        start_point.set_offsets([path[0, 0], path[0, 1]])
        current_point.set_offsets([current_path[-1, 0], current_path[-1, 1]])

        # 顯示當前步驟的數據
        current_w = current_path[-1]
        all_data_loss = loss(current_w)  # 所有資料的損失（用於視覺化）

        # 計算當前使用的訓練資料（與訓練過程一致）
        if frame <= 3:
            # 前三筆固定使用 data 1, 2, 3
            current_idx = frame - 1  # 0, 1, 2
        else:
            # 之後隨機選擇（重現訓練過程）
            np.random.seed(42)
            for i in range(3, frame):  # 跳過前3次固定選擇
                idx = np.random.randint(0, len(X))
            current_idx = np.random.randint(0, len(X))

        x_i = X[current_idx]
        y_i = y[current_idx]

        # 計算當前步驟使用的權重（應該是更新前的權重）
        if frame == 1:
            # 第一次迭代：使用初始權重 [0, 0]
            w_for_gradient = path[0]
        else:
            # 其他迭代：使用前一步的權重
            w_for_gradient = path[frame - 1]

        single_loss = loss_single(w_for_gradient, x_i, y_i)  # 單筆資料的損失
        grad = compute_sgd_gradient(w_for_gradient, x_i, y_i)

        text = "Iteration {} (SGD):\nw1={:.4f}, w2={:.4f}\nAll data Loss={:.4f}\nSingle data Loss={:.4f}\nUsed data {}: x1={}, x2={}, y={}\nGradient=[{:.4f}, {:.4f}]".format(
            frame,
            current_w[0],
            current_w[1],
            all_data_loss,
            single_loss,
            current_idx + 1,
            x_i[0],
            x_i[1],
            y_i,
            grad[0],
            grad[1],
        )
        step_text.set_text(text)

    return line, point, start_point, current_point, step_text


# 創建動畫
anim = FuncAnimation(
    fig, animate, frames=len(path), interval=1000, repeat=True, blit=False
)

plt.tight_layout()

# 保存為 GIF 檔案
print("正在匯出 GIF 檔案...")
try:
    anim.save("sgd_animation.gif", writer="pillow", fps=1, dpi=100)
    print("GIF 檔案已成功保存為 'sgd_animation.gif'")
except Exception as e:
    print(f"保存 GIF 時發生錯誤: {e}")
    print("請確保已安裝 pillow 套件: pip install pillow")

plt.show()

# 打印所有步驟的位置
print("All SGD iteration positions:")
for i, pos in enumerate(path):
    if i == 0:
        print(
            f"Iteration {i}: ({pos[0]:.4f}, {pos[1]:.4f}), Loss={loss(pos):.4f}, No gradient"
        )
    else:
        # 重新計算使用的資料（與訓練過程一致）
        if i <= 3:
            # 前三筆固定使用 data 1, 2, 3
            current_idx = i - 1  # 0, 1, 2
        else:
            # 之後隨機選擇（重現訓練過程）
            np.random.seed(42)
            for j in range(3, i):  # 跳過前3次固定選擇
                idx = np.random.randint(0, len(X))
            current_idx = np.random.randint(0, len(X))

        x_i = X[current_idx]
        y_i = y[current_idx]

        # 計算當前步驟使用的權重（應該是更新前的權重）
        if i == 1:
            # 第一次迭代：使用初始權重 [0, 0]
            w_for_gradient = path[0]
        else:
            # 其他迭代：使用前一步的權重
            w_for_gradient = path[i - 1]

        single_loss = loss_single(w_for_gradient, x_i, y_i)
        grad = compute_sgd_gradient(w_for_gradient, x_i, y_i)
        print(
            f"Iteration {i}: ({pos[0]:.4f}, {pos[1]:.4f}), All data Loss={loss(pos):.4f}, "
            f"Single data Loss={single_loss:.4f}, Used data {current_idx + 1}, "
            f"Gradient=[{grad[0]:.4f}, {grad[1]:.4f}]"
        )

# 打印最終結果和訓練資料
print("\n最終結果:")
final_w = path[-1]
final_loss = loss(final_w)
print(f"最終權重: w1={final_w[0]:.4f}, w2={final_w[1]:.4f}")
print(f"最終損失: {final_loss:.4f}")

print("\n訓練資料:")
for i, (x, target) in enumerate(zip(X, y)):
    prediction = predict(final_w, x.reshape(1, -1))[0]
    print(f"資料 {i+1}: x1={x[0]}, x2={x[1]}, y={target}, y_pred={prediction:.4f}")
