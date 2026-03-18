#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一維高斯分布函數繪圖程式
繪製不同參數的高斯分布曲線，展示均值(μ)和標準差(σ)對分布的影響
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian_1d(x, mu=0, sigma=1):
    """
    一維高斯分布函數

    參數:
    x: 輸入值
    mu: 均值 (默認: 0)
    sigma: 標準差 (默認: 1)

    返回:
    高斯分布的概率密度值
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    """主函數"""
    # 設置中文字體
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 創建 x 軸數據
    x = np.linspace(-5, 5, 1000)

    # 創建圖表
    plt.figure(figsize=(12, 8))

    # 繪製不同參數的高斯分布曲線
    # 標準高斯分布 (μ=0, σ=1)
    y1 = gaussian_1d(x, mu=0, sigma=1)
    plt.plot(x, y1, "b-", linewidth=2, label="標準高斯分布 (μ=0, σ=1)")

    # 不同均值的高斯分布
    # y2 = gaussian_1d(x, mu=1, sigma=1)
    # plt.plot(x, y2, "r-", linewidth=2, label="高斯分布 (μ=1, σ=1)")

    # y3 = gaussian_1d(x, mu=-1, sigma=1)
    # plt.plot(x, y3, "g-", linewidth=2, label="高斯分布 (μ=-1, σ=1)")

    # 不同標準差的高斯分布
    y4 = gaussian_1d(x, mu=0, sigma=0.5)
    plt.plot(x, y4, "m-", linewidth=2, label="高斯分布 (μ=0, σ=0.5)")

    y5 = gaussian_1d(x, mu=0, sigma=2)
    plt.plot(x, y5, "c-", linewidth=2, label="高斯分布 (μ=0, σ=2)")

    # 添加垂直線標示均值位置
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="μ=0")
    # plt.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="μ=1")
    # plt.axvline(x=-1, color="green", linestyle="--", alpha=0.5, label="μ=-1")

    # 設置圖表屬性
    plt.title("一維高斯分布函數", fontsize=16, fontweight="bold")
    plt.xlabel("x", fontsize=14)
    plt.ylabel("概率密度 f(x)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 設置軸範圍
    plt.xlim(-5, 5)
    plt.ylim(0, 0.8)

    # 添加數學公式說明
    plt.text(
        0.02,
        0.98,
        "$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}$",
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 保存圖表
    plt.tight_layout()
    plt.savefig(
        "gaussian_1d.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 顯示圖表
    plt.show()

    print("一維高斯分布圖表已生成並保存為 'gaussian_1d_plot.png'")


def plot_single_gaussian():
    """繪製單個高斯分布曲線的簡化版本"""
    x = np.linspace(-4, 4, 1000)
    y = gaussian_1d(x, mu=0, sigma=1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "b-", linewidth=3, label="高斯分布 (μ=0, σ=1)")
    plt.fill_between(x, y, alpha=0.3, color="blue")

    plt.title("標準高斯分布", fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("概率密度 f(x)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
        print("\n" + "=" * 50)
        print("是否要查看簡化版本？(y/n): ", end="")
        choice = input().lower()
        if choice == "y":
            plot_single_gaussian()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
