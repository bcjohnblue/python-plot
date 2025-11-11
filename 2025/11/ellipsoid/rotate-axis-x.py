#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
橢圓方程繪圖程式
繪製方程: 5x₁² + 8x₁x₂ + 5x₂² = 1
這是一個包含交叉項的橢圓方程
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """主函數"""
    # 設置中文字體
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 橢圓方程: 5x₁² + 8x₁x₂ + 5x₂² = 1
    # 轉換為矩陣形式: [x₁, x₂] * A * [x₁, x₂]ᵀ = 1
    # 其中 A = [[5, 4], [4, 5]] (交叉項 8x₁x₂ 在矩陣中分成兩個 4)
    A = np.array([[5, 4], [4, 5]])

    # 使用特徵值分解找到主軸和半軸長度
    # A = P * D * Pᵀ，其中 D 是對角矩陣（特徵值），P 是特徵向量矩陣
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # 半軸長度 = 1 / sqrt(特徵值)
    # 因為在標準形式中，橢圓方程為 (x/a)² + (y/b)² = 1
    # 而特徵值對應的是 1/a² 和 1/b²
    a = 1.0 / np.sqrt(eigenvalues[0])  # 第一個半軸長度
    b = 1.0 / np.sqrt(eigenvalues[1])  # 第二個半軸長度

    # 旋轉角度（特徵向量給出的旋轉）
    # 第一個特徵向量對應主軸方向
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # 創建參數化橢圓（在標準坐標系中）
    t = np.linspace(0, 2 * np.pi, 200)
    x_standard = a * np.cos(t)
    y_standard = b * np.sin(t)

    # 旋轉到實際坐標系
    # 使用旋轉矩陣
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # 應用旋轉
    coords = np.vstack([x_standard, y_standard])
    coords_rotated = rotation_matrix @ coords
    x1 = coords_rotated[0, :]
    x2 = coords_rotated[1, :]

    # 創建 2D 圖形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # 繪製橢圓
    ax.plot(x1, x2, "b-", linewidth=2, label="$5x_1^2 + 8x_1x_2 + 5x_2^2 = 1$")

    # 設置坐標軸標籤
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)

    # 設置標題
    # ax.set_title(
    #     "$5x_1^2 + 8x_1x_2 + 5x_2^2 = 1$",
    #     fontsize=16,
    #     fontweight="bold",
    #     pad=20,
    # )

    # 設置坐標軸範圍（根據半軸長度自動調整）
    max_range = max(a, b) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])

    # 設置坐標軸比例相等，使橢圓不會被拉伸
    ax.set_aspect("equal")

    # 添加網格
    ax.grid(True, alpha=0.3)

    # 添加坐標軸
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    # 添加圖例
    ax.legend(fontsize=12, loc="upper right")

    # 保存圖形
    plt.tight_layout()
    plt.savefig("rotate-axis-x.png", dpi=300, bbox_inches="tight")

    # 顯示圖形
    plt.show()

    print("橢圓圖已生成並保存為 'rotate-axis.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
