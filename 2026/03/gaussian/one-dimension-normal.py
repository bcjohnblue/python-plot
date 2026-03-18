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

    # 繪製標準高斯分布曲線 (μ=0, σ=1)
    y1 = gaussian_1d(x, mu=0, sigma=1)
    plt.plot(x, y1, "b-", linewidth=3, label="標準高斯分布 (μ=0, σ=1)")

    # 添加垂直分隔線
    # plt.axvline(x=0, color="gray", linestyle="-", alpha=0.7, linewidth=2)  # 均值線
    # plt.axvline(x=1, color="gray", linestyle="-", alpha=0.6, linewidth=1.5)  # 1σ
    # plt.axvline(x=-1, color="gray", linestyle="-", alpha=0.6, linewidth=1.5)  # -1σ
    # plt.axvline(x=2, color="gray", linestyle="-", alpha=0.6, linewidth=1.5)  # 2σ
    # plt.axvline(x=-2, color="gray", linestyle="-", alpha=0.6, linewidth=1.5)  # -2σ
    # plt.axvline(x=3, color="gray", linestyle="-", alpha=0.6, linewidth=1.5)  # 3σ
    # plt.axvline(x=-3, color="gray", linestyle="-", alpha=0.6, linewidth=1.5)  # -3σ

    # 添加各標準差範圍的陰影區域
    # 中心區域 (μ 到 1σ) - 34.1%
    x_center_right = np.linspace(0, 1, 100)
    y_center_right = gaussian_1d(x_center_right, mu=0, sigma=1)
    plt.fill_between(x_center_right, y_center_right, alpha=0.6, color="blue")

    # 中心區域 (μ 到 -1σ) - 34.1%
    x_center_left = np.linspace(-1, 0, 100)
    y_center_left = gaussian_1d(x_center_left, mu=0, sigma=1)
    plt.fill_between(x_center_left, y_center_left, alpha=0.6, color="blue")

    # 次級區域 (1σ 到 2σ) - 13.6%
    x_secondary_right = np.linspace(1, 2, 100)
    y_secondary_right = gaussian_1d(x_secondary_right, mu=0, sigma=1)
    plt.fill_between(x_secondary_right, y_secondary_right, alpha=0.4, color="blue")

    # 次級區域 (-1σ 到 -2σ) - 13.6%
    x_secondary_left = np.linspace(-2, -1, 100)
    y_secondary_left = gaussian_1d(x_secondary_left, mu=0, sigma=1)
    plt.fill_between(x_secondary_left, y_secondary_left, alpha=0.4, color="blue")

    # 外圍區域 (2σ 到 3σ) - 2.1%
    x_outer_right = np.linspace(2, 3, 100)
    y_outer_right = gaussian_1d(x_outer_right, mu=0, sigma=1)
    plt.fill_between(x_outer_right, y_outer_right, alpha=0.2, color="blue")

    # 外圍區域 (-2σ 到 -3σ) - 2.1%
    x_outer_left = np.linspace(-3, -2, 100)
    y_outer_left = gaussian_1d(x_outer_left, mu=0, sigma=1)
    plt.fill_between(x_outer_left, y_outer_left, alpha=0.2, color="blue")

    # 極端區域 (超出 3σ) - 0.1%
    x_extreme_right = np.linspace(3, 5, 100)
    y_extreme_right = gaussian_1d(x_extreme_right, mu=0, sigma=1)
    plt.fill_between(x_extreme_right, y_extreme_right, alpha=0.1, color="blue")

    # 極端區域 (超出 -3σ) - 0.1%
    x_extreme_left = np.linspace(-5, -3, 100)
    y_extreme_left = gaussian_1d(x_extreme_left, mu=0, sigma=1)
    plt.fill_between(x_extreme_left, y_extreme_left, alpha=0.1, color="blue")

    # 設置圖表屬性
    plt.title("標準高斯分佈", fontsize=16, fontweight="bold")
    # plt.xlabel("x", fontsize=14)
    plt.ylabel("概率密度 f(x)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 設置軸範圍
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)

    # 添加數學公式說明
    plt.text(
        0.02,
        0.98,
        "$f(x) = \\frac{1}{\\sqrt{2\\pi}} e^{-\\frac{x^2}{2}}$",
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.9),
    )

    # 添加詳細的百分比標註
    # 中心區域百分比 (34.1%)
    plt.text(
        0.5,
        0.25,
        "34.1%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )
    plt.text(
        -0.5,
        0.25,
        "34.1%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )

    # 次級區域百分比 (13.6%)
    plt.text(
        1.5,
        0.15,
        "13.6%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )
    plt.text(
        -1.5,
        0.15,
        "13.6%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )

    # 外圍區域百分比 (2.1%)
    plt.text(
        2.5,
        0.08,
        "2.1%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )
    plt.text(
        -2.5,
        0.08,
        "2.1%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )

    # 極端區域百分比 (0.1%)
    plt.text(
        3.5,
        0.03,
        "0.1%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )
    plt.text(
        -3.5,
        0.03,
        "0.1%",
        fontsize=11,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )

    # 添加標準差標記
    plt.text(
        0,
        -0.02,
        "μ",
        fontsize=12,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )
    plt.text(
        1,
        -0.02,
        "1σ",
        fontsize=10,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )
    plt.text(
        -1,
        -0.02,
        "-1σ",
        fontsize=10,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )
    plt.text(
        2,
        -0.02,
        "2σ",
        fontsize=10,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )
    plt.text(
        -2,
        -0.02,
        "-2σ",
        fontsize=10,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )
    plt.text(
        3,
        -0.02,
        "3σ",
        fontsize=10,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )
    plt.text(
        -3,
        -0.02,
        "-3σ",
        fontsize=10,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )

    # 保存圖表
    plt.tight_layout()
    plt.savefig(
        "gaussian_1d_normal.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 顯示圖表
    plt.show()

    print("一維高斯分布圖表已生成並保存為 'gaussian_1d_plot.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
