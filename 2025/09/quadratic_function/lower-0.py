#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次型函式 f = ax² + 2bxy + cy² 的倒碗形狀繪圖
當 f 始終為負時，函式呈現倒碗形狀（橢圓拋物面）
條件：判別式 b² - ac < 0 且 a < 0
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 設置中文字體支援
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def quadratic_function(x, y, a, b, c):
    """
    計算二次型函式 f = ax² + 2bxy + cy²

    Args:
        x, y: 座標點
        a, b, c: 二次型係數

    Returns:
        函式值
    """
    return a * x**2 + 2 * b * x * y + c * y**2


def create_negative_definite_quadratic():
    """
    創建一個負定二次型函式的倒碗形狀圖
    """
    # 設定係數，確保 f 始終為負
    a, b, c = -1, 0, -1

    # 創建網格
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # 計算函式值
    Z = quadratic_function(X, Y, a, b, c)

    # 創建圖表
    fig = plt.figure(figsize=(10, 8))

    # 3D 表面圖
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z, cmap="plasma", alpha=0.9, linewidth=0.1, antialiased=True
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title(f"f = {a}x² + {2*b}xy + {c}y²")

    # 設置軸範圍和比例
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-50, 0)  # 限制 z 軸範圍，讓倒碗形狀更明顯

    # 設置相等的軸比例，增加 z 軸高度
    ax.set_box_aspect([1, 1, 1.5])  # [x, y, z] 比例，z 軸拉高

    ax.view_init(elev=30, azim=45)

    # 添加顏色條
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="f(x,y)")

    plt.tight_layout()

    # 顯示判別式信息
    discriminant = b**2 - a * c
    print(f"係數設定：a = {a}, b = {b}, c = {c}")
    print(f"判別式 b² - ac = {discriminant:.2f}")
    print(f"由於判別式 < 0 且 a < 0，函式 f 始終為負")
    print(f"最大值出現在 (0, 0) 點，值為 0")

    return fig


def main():
    """主函數"""
    print("=" * 60)
    print("二次型函式 f = ax² + 2bxy + cy² 的倒碗形狀繪圖")
    print("=" * 60)

    try:
        # 創建主要的倒碗形狀圖
        print("\n1. 創建主要倒碗形狀圖...")
        fig1 = create_negative_definite_quadratic()
        plt.savefig(
            "./quadratic_inverted_bowl.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("已保存主要圖表為: quadratic_inverted_bowl.png")

        # 顯示圖表
        plt.show()

        print("\n繪圖完成！")
        print("\n數學說明：")
        print("- 當判別式 b² - ac < 0 且 a < 0 時，二次型函式為負定")
        print("- 函式圖形呈現倒碗形狀（橢圓拋物面）")
        print("- 最大值出現在原點 (0,0)，值為 0")
        print("- 函式值隨著距離原點越遠而減小")

    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
