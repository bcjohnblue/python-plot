#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
橢球方程繪圖程式
繪製方程: 4x₁² + x₂² + (1/9)x₃² = 1
這是一個橢球面，半軸長度分別為: a = 1/2, b = 1, c = 3
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    """主函數"""
    # 設置中文字體
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 橢球參數
    # 方程: 4x₁² + x₂² + (1/9)x₃² = 1
    # 轉換為標準形式: (x₁/(1/2))² + (x₂/1)² + (x₃/3)² = 1
    a = 0.5  # x₁ 方向的半軸長度
    b = 1.0  # x₂ 方向的半軸長度
    c = 3.0  # x₃ 方向的半軸長度

    # 創建參數網格
    u = np.linspace(0, 2 * np.pi, 50)  # 方位角
    v = np.linspace(0, np.pi, 50)  # 極角

    # 參數化橢球方程
    # x₁ = a * sin(v) * cos(u)
    # x₂ = b * sin(v) * sin(u)
    # x₃ = c * cos(v)
    x1 = a * np.outer(np.sin(v), np.cos(u))
    x2 = b * np.outer(np.sin(v), np.sin(u))
    x3 = c * np.outer(np.cos(v), np.ones_like(u))

    # 創建 3D 圖形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 繪製橢球面（交換坐標軸：x3 顯示在 x 軸，使橢球橫向顯示）
    # 使用藍色系
    ax.plot_surface(
        x3,
        x1,
        x2,
        color="steelblue",  # 淺藍色
        alpha=0.95,  # 稍微降低透明度，讓橢球更明顯
        edgecolor="none",  # 深藍色邊緣線，更明顯
        linewidth=0.8,  # 增加邊緣線寬度，讓橢球更明顯
        shade=True,  # 啟用光影效果
        antialiased=False,
    )

    # 設置坐標軸標籤（對應交換後的坐標）
    ax.set_xlabel("$x_3$", fontsize=14, labelpad=10)
    ax.set_ylabel("$x_1$", fontsize=14, labelpad=10)
    ax.set_zlabel("$x_2$", fontsize=14, labelpad=10)

    # 設置標題
    # ax.set_title(
    #     "橢球: $4x_1^2 + x_2^2 + \\frac{1}{9}x_3^2 = 1$",
    #     fontsize=16,
    #     fontweight="bold",
    #     pad=0,  # 減小標題與圖形之間的距離
    # )

    # 設置坐標軸範圍（對應交換後的坐標：x3 在 x 軸，x1 在 y 軸，x2 在 z 軸）
    ax.set_xlim([-3.5, 3.5])  # x3 方向（現在是 x 軸）
    ax.set_ylim([-1, 1])  # x1 方向（現在是 y 軸）
    ax.set_zlim([-1.2, 1.2])  # x2 方向（現在是 z 軸）

    # 設置 y 軸（x1 方向）只顯示 -1, 0, 1 這三個刻度
    ax.set_yticks([-1, 0, 1])

    # 設置坐標軸比例（對應交換後的坐標，使橢球橫向顯示）
    ax.set_box_aspect([3, 0.5, 1])

    # 添加網格
    ax.grid(True, alpha=0.3)

    # 設置初始視角（仰角和方位角）
    # elev: 仰角（垂直角度），azim: 方位角（水平角度）
    ax.view_init(elev=20, azim=45)

    # 添加資訊文本
    # info_text = (
    #     f"半軸長度:\n"
    #     f"$a = {a}$ (x₁方向)\n"
    #     f"$b = {b}$ (x₂方向)\n"
    #     f"$c = {c}$ (x₃方向)"
    # )
    # ax.text2D(
    #     0.02,
    #     0.98,
    #     info_text,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    # )

    # 保存圖形
    plt.tight_layout()
    plt.savefig("same-axis.png", dpi=300, bbox_inches="tight")

    # 顯示圖形
    plt.show()

    print("橢球圖已生成並保存為 'same-axis.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
