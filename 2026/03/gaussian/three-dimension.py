#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三維高斯分布 3D 等值面圖繪圖程式
繪製三維高斯分布（三個隨機變量 x, y, z）的等密度椭球体表面
展示不同概率密度等級的椭球形狀
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2, multivariate_normal
from matplotlib import cm

# 設置參數：三維均值和協方差矩陣
mu = [0, 0, 0]  # 三維均值向量 [μx, μy, μz]
# 協方差矩陣（3x3）- 可以調整來改變分布形狀
cov = np.array(
    [
        [1.0, 0.0, 0.0],  # x 的方差和與其他變量的協方差
        [0.0, 1.0, 0.0],  # y 的方差和與其他變量的協方差
        [0.0, 0.0, 1.0],  # z 的方差和與其他變量的協方差
    ]
)


def gaussian_3d(points, mu, cov):
    """
    三維高斯分布函數

    參數:
    points: Nx3 的點陣列，每行是一個 [x, y, z] 點
    mu: 三維均值向量 [μx, μy, μz]
    cov: 3x3 協方差矩陣

    返回:
    每個點的概率密度值
    """
    return multivariate_normal.pdf(points, mean=mu, cov=cov)


def plot_gaussian_ellipsoid(ax, mu, cov, n_std=1, alpha=0.3, color=None):
    """
    繪製三維高斯分布的等密度椭球体表面

    參數:
    ax: 3D 坐標軸
    mu: 均值向量
    cov: 協方差矩陣
    n_std: 標準差倍數（控制椭球体大小）
    alpha: 透明度
    color: 顏色
    """
    # 生成球面座標
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    # 將球面點堆疊成 (3, N) 的形式
    sphere_points = np.stack([x_sphere.ravel(), y_sphere.ravel(), z_sphere.ravel()])

    # 對協方差矩陣進行特徵分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 創建變換矩陣：縮放 + 旋轉
    # 縮放因子為 sqrt(eigenvalue) * n_std
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues) * n_std)

    # 將單位球面變換成椭球体
    ellipsoid_points = transform @ sphere_points

    # 加上均值偏移
    x_ellipsoid = ellipsoid_points[0, :].reshape(x_sphere.shape) + mu[0]
    y_ellipsoid = ellipsoid_points[1, :].reshape(y_sphere.shape) + mu[1]
    z_ellipsoid = ellipsoid_points[2, :].reshape(z_sphere.shape) + mu[2]

    # 繪製椭球体表面
    return ax.plot_surface(
        x_ellipsoid,
        y_ellipsoid,
        z_ellipsoid,
        alpha=alpha,
        color=color,
        edgecolor="white",
        linewidth=0.3,
        antialiased=True,
    )


def main():
    """主函數"""
    # 設置中文字體
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 創建3D圖表
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")

    # 定義要繪製的等密度椭球体層級（標準差倍數）
    n_std_levels = [3, 2, 1]  # 從外到內繪製：3σ, 2σ, 1σ 椭球体
    colors = ["#FFD700", "#FFA500", "#FF6B6B"]  # 從黃到紅（熱圖配色）
    alphas = [0.3, 0.5, 0.7]  # 外層更透明，內層較不透明（但不完全遮擋）

    # 繪製多個等密度椭球体表面
    for i, (n_std, color, alpha) in enumerate(zip(n_std_levels, colors, alphas)):
        plot_gaussian_ellipsoid(ax, mu, cov, n_std=n_std, alpha=alpha, color=color)

    # 添加從均值向外延伸的參考線
    line_length = 1.5
    ax.plot(
        [mu[0], mu[0]],
        [mu[1], mu[1]],
        [mu[2] - line_length, mu[2] + line_length],
        "k--",
        linewidth=1.5,
        alpha=0.3,
        zorder=999,
    )
    ax.plot(
        [mu[0] - line_length, mu[0] + line_length],
        [mu[1], mu[1]],
        [mu[2], mu[2]],
        "k--",
        linewidth=1.5,
        alpha=0.3,
        zorder=999,
    )
    ax.plot(
        [mu[0], mu[0]],
        [mu[1] - line_length, mu[1] + line_length],
        [mu[2], mu[2]],
        "k--",
        linewidth=1.5,
        alpha=0.3,
        zorder=999,
    )

    # 添加坐標軸箭頭（顯示主軸方向）
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    scale = 3.0
    colors_axes = ["red", "green", "blue"]
    for i in range(3):
        direction = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * scale
        ax.quiver(
            mu[0],
            mu[1],
            mu[2],
            direction[0],
            direction[1],
            direction[2],
            color=colors_axes[i],
            alpha=0.6,
            arrow_length_ratio=0.15,
            linewidth=2,
        )

    # 設置圖表屬性
    ax.set_title(
        "3D Gaussian Distribution",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("X", fontsize=14, labelpad=10)
    ax.set_ylabel("Y", fontsize=14, labelpad=10)
    ax.set_zlabel("Z", fontsize=14, labelpad=10)

    # 設置相等的縮放比例
    max_range = 3.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # 設置視角
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.2)

    # 設置背景顏色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 添加參數說明和圖例（3D 橢球機率由卡方分布計算）
    color_labels = ["淡黃（外層）", "深黃（中層）", "暗黃（內層）"]
    level_lines = []
    for n_std, color_label in zip(n_std_levels, color_labels):
        prob = chi2.cdf(n_std**2, df=3)
        level_lines.append(f"  {color_label}：{n_std}σ，P(內含)≈{prob:.1%}")
    level_lines.reverse()  # 文字敘述改為由內層到外層

    sigma_lines = [
        f"[{cov[0,0]:.1f}, {cov[0,1]:.1f}, {cov[0,2]:.1f}]",
        f"[{cov[1,0]:.1f}, {cov[1,1]:.1f}, {cov[1,2]:.1f}]",
        f"[{cov[2,0]:.1f}, {cov[2,1]:.1f}, {cov[2,2]:.1f}]",
    ]

    param_text = f"μ = [{mu[0]}, {mu[1]}, {mu[2]}]\n" f"Σ =\n" + "\n".join(
        sigma_lines
    ) + "\n\n" + f"三維等密度橢球（以馬氏距離定義）：\n" + "\n".join(level_lines)
    ax.text2D(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.95, edgecolor="gray", pad=0.8
        ),
        zorder=10,
    )

    # 保存圖表
    plt.tight_layout()
    plt.savefig(
        "gaussian_3d.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    # 顯示圖表
    plt.show()

    print("三維高斯分布等密度椭球体圖已生成並保存為 'gaussian_3d.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
