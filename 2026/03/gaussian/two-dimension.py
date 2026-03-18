#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二維高斯分布函數繪圖程式
繪製不同參數的二維高斯分布，展示均值(μ)和協方差矩陣(Σ)對分布的影響
"""

import numpy as np
import matplotlib.pyplot as plt

# 設置參數
mu = [0, 0]
sigma_x = 1
sigma_y = 1
rho = 0


def gaussian_2d(x, y, mu=[0, 0], sigma_x=1, sigma_y=1, rho=0):
    """
    二維高斯分布函數

    參數:
    x, y: 輸入的網格座標
    mu: 均值向量 [μx, μy] (默認: [0, 0])
    sigma_x: x方向的標準差 (默認: 1)
    sigma_y: y方向的標準差 (默認: 1)
    rho: 相關係數 (默認: 0，表示獨立)

    返回:
    二維高斯分布的概率密度值
    """
    mu_x, mu_y = mu
    # 構建協方差矩陣
    cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ]
    )
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    # 計算 (x - μ) 和 (y - μ)
    diff_x = x - mu_x
    diff_y = y - mu_y

    # 計算指數部分: -0.5 * (x - μ)^T * Σ^(-1) * (x - μ)
    # 展開為: -0.5 * [diff_x, diff_y] * Σ^(-1) * [diff_x; diff_y]
    a = cov_inv[0, 0]
    b = cov_inv[0, 1]
    c = cov_inv[1, 1]
    exponent = -0.5 * (a * diff_x**2 + 2 * b * diff_x * diff_y + c * diff_y**2)

    # 計算正規化常數
    norm = 1 / (2 * np.pi * np.sqrt(det_cov))

    return norm * np.exp(exponent)


def main():
    """主函數"""
    # 設置中文字體
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 創建網格數據
    x_range = np.linspace(-3, 3, 300)
    y_range = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x_range, y_range)

    # 創建圖表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 繪製標準二維高斯分布
    Z = gaussian_2d(X, Y, mu=mu, sigma_x=sigma_x, sigma_y=sigma_y, rho=rho)

    # 繪製填充等高線圖（熱圖風格）
    contourf = ax.contourf(
        X, Y, Z, levels=50, cmap="viridis", alpha=1.0, antialiased=False
    )

    # 繪製等高線
    # contour = ax.contour(X, Y, Z, levels=15, colors="white", alpha=0.3, linewidths=0.5)

    # 添加顏色條
    cbar = plt.colorbar(contourf, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label("概率密度", fontsize=12, rotation=270, labelpad=20)

    # 標示均值位置
    ax.plot(
        mu[0], mu[1], "r*", markersize=12, markeredgecolor="white", markeredgewidth=1
    )

    # 設置圖表屬性
    ax.set_title("2D Gaussian Distribution", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # 添加參數說明
    param_text = (
        f"$\\mu$ = [{mu[0]}, {mu[1]}]\n"
        f"$\\sigma_x$ = {sigma_x}, $\\sigma_y$ = {sigma_y}\n"
        f"$\\rho$ = {rho}"
    )
    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 保存圖表
    plt.tight_layout()
    plt.savefig(
        "gaussian_2d.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 顯示圖表
    plt.show()

    print("二維高斯分布圖表已生成並保存為 'gaussian_2d.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
