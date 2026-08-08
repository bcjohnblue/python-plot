#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二維高斯分布的降維投影示範
1) 圖一：二維高斯 N(mu, Sigma) 散點圖 + 1σ/2σ 等密度橢圓 + 投影到 x 軸的虛線
2) 圖二：投影後的一維高斯 N(1, 4) 密度曲線與投影點
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 設置參數（對應文章例子）
MU = np.array([1.0, 2.0])
COV = np.array([[4.0, 1.0], [1.0, 3.0]])
A = np.array([[1.0, 0.0]])  # 1x2 投影矩陣，只取出第一個分量
N_SAMPLES = 100

# 投影後的一維高斯參數：mu_Y = A mu = 1, sigma_Y^2 = A Sigma A^T = 4
MU_Y = float((A @ MU)[0])
VAR_Y = float((A @ COV @ A.T)[0, 0])
SIGMA_Y = np.sqrt(VAR_Y)


def add_cov_ellipses(ax, mean, cov, ks=(1, 2)):
    """在 ax 上疊加 kσ 等密度橢圓（k=1, 2）。"""
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh 回傳特徵值由小到大，取最大特徵值對應的特徵向量當主軸
    angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
    for k in ks:
        ellipse = Ellipse(
            xy=mean,
            width=2 * k * np.sqrt(eigvals[-1]),
            height=2 * k * np.sqrt(eigvals[0]),
            angle=angle,
            fill=False,
            edgecolor="#222",
            linewidth=1.2,
            linestyle="-" if k == 1 else (0, (4, 3)),
            alpha=0.8,
        )
        ax.add_patch(ellipse)


def main():
    """主函數"""
    # 設置中文字體
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 固定亂數種子，讓每次結果可重現
    np.random.seed(43)

    # 生成二維高斯散點
    points = np.random.multivariate_normal(mean=MU, cov=COV, size=N_SAMPLES)

    # 投影到 x 軸（即 Y = A X，只保留第一個分量）
    projected_x = points @ A.T  # shape: (N, 1)
    projected_x = projected_x[:, 0]

    # 建立上下兩張子圖（共用同一段 x 範圍，垂直對齊）
    # height_ratios 給足各自 box aspect 需要的高度，兩個子圖才會維持相同寬度（x 軸對齊）
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 10), dpi=160, gridspec_kw={"height_ratios": [2.1, 1]}
    )
    ax1, ax2 = axes
    ax1.set_anchor("S")  # 上圖貼齊下緣、下圖貼齊上緣，縮小兩圖間距
    ax2.set_anchor("N")

    # 第一張：二維高斯散點 + 等密度橢圓 + 投影過程
    # 投影虛線：每個點垂直落到 x 軸
    for x, y in points:
        ax1.plot([x, x], [y, 0], color="gray", alpha=0.25, linewidth=0.6)
    ax1.scatter(points[:, 0], points[:, 1], s=24, color="navy", alpha=0.85)
    ax1.scatter(projected_x, np.zeros_like(projected_x), s=24, color="crimson", alpha=0.85)
    add_cov_ellipses(ax1, MU, COV)
    ax1.axhline(0, color="#444", linewidth=1.2)
    ax1.axvline(0, color="#444", linewidth=1.2)
    ax1.grid(True, linestyle=(0, (2, 2)), alpha=0.45)
    ax1.set_xlim(-5.5, 7.5)
    ax1.set_ylim(-3.5, 7.5)
    # 用 box aspect 固定「資料範圍比例 = 畫布比例」，等同 equal aspect，
    # 但維持與下圖相同的軸寬度，讓上下兩圖的 x 軸對齊
    ax1.set_box_aspect(11 / 13)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # 第二張：投影後的一維高斯 N(1, 4)
    x_grid = np.linspace(-5.5, 7.5, 400)
    pdf = np.exp(-((x_grid - MU_Y) ** 2) / (2 * VAR_Y)) / (SIGMA_Y * np.sqrt(2 * np.pi))
    ax2.plot(x_grid, pdf, color="navy", linewidth=1.8)
    ax2.fill_between(x_grid, pdf, color="navy", alpha=0.15)
    ax2.scatter(projected_x, np.zeros_like(projected_x), s=24, color="crimson", alpha=0.85)
    ax2.axvline(MU_Y, color="#444", linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.7)
    ax2.axhline(0, color="#444", linewidth=1.2)
    ax2.grid(True, linestyle=(0, (2, 2)), alpha=0.45)
    ax2.set_xlim(-5.5, 7.5)
    ax2.set_ylim(-0.02, 0.25)
    ax2.set_box_aspect(0.4)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    # 保存圖表
    plt.tight_layout(pad=1.4)
    plt.savefig(
        "gaussian_2d_projection.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 顯示圖表
    plt.show()

    print("圖表已生成並保存為 'gaussian_2d_projection.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
