# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


# 定義損失函數 L(w1, w2) = (4 - (w1 + w2))^2
def loss(w1, w2):
    return (4 - (w1 + w2)) ** 2


# 計算梯度
def compute_gradient(w):
    return -2 * (4 - w[0] - w[1]) * np.array([1.0, 1.0])


# 初始點
w = np.array([0.0, 0.0])
lr = 0.1
path = [w.copy()]

# 預先計算所有步驟
for _ in range(10):
    grad = compute_gradient(w)
    w = w - lr * grad
    path.append(w.copy())

path = np.array(path)

# 設置圖形
fig, ax = plt.subplots(figsize=(12, 8))

# 準備網格數據
w1_vals = np.linspace(-1, 4.5, 100)
w2_vals = np.linspace(-1, 4.5, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = loss(W1, W2)

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
    fontsize=12,
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
        current_loss = loss(current_w[0], current_w[1])
        grad = compute_gradient(current_w)
        text = "Iteration 0:\nw1={:.4f}, w2={:.4f}\nLoss={:.4f}\nGradient=[{:.4f}, {:.4f}]".format(
            current_w[0], current_w[1], current_loss, grad[0], grad[1]
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
        current_loss = loss(current_w[0], current_w[1])
        grad = compute_gradient(current_w)
        text = "Iteration {}:\nw1={:.4f}, w2={:.4f}\nLoss={:.4f}\nGradient=[{:.4f}, {:.4f}]".format(
            frame, current_w[0], current_w[1], current_loss, grad[0], grad[1]
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
    anim.save("gradient_descent_animation.gif", writer="pillow", fps=1, dpi=100)
    print("GIF 檔案已成功保存為 'gradient_descent_animation.gif'")
except Exception as e:
    print(f"保存 GIF 時發生錯誤: {e}")
    print("請確保已安裝 pillow 套件: pip install pillow")

plt.show()

# 打印所有步驟的位置
print("All iteration positions:")
for i, pos in enumerate(path):
    current_loss = loss(pos[0], pos[1])
    grad = compute_gradient(pos)
    print(
        "Iteration {}: ({:.4f}, {:.4f}), Loss={:.4f}, Gradient=[{:.4f}, {:.4f}]".format(
            i, pos[0], pos[1], current_loss, grad[0], grad[1]
        )
    )
