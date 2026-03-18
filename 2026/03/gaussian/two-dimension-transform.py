#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二維高斯分布點狀圖與線性變換示範
1) 圖一：標準二維高斯分布散點圖
2) 圖二：將圖一資料做 y 方向縮放 0.5 後，再逆時針旋轉 30 度
"""

import numpy as np
import matplotlib.pyplot as plt

# 設置參數
MU = np.array([0.0, 0.0])
COV = np.array([[1.0, 0.0], [0.0, 1.0]])  # 標準二維高斯 N(0, I)
N_SAMPLES = 100
SHIFT = np.array([1.0, 1.0])  # 第二張圖點雲往右上平移 1 單位


def transform_points(points, y_scale=0.5, ccw_deg=30):
    """先對 y 縮放，再逆時針旋轉指定角度。"""
    # y 方向縮放
    scale = np.array([[1.0, 0.0], [0.0, y_scale]])
    scaled = points @ scale.T

    # 逆時針旋轉 θ 度
    theta = np.deg2rad(ccw_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = scaled @ rot.T
    return rotated


def main():
    """主函數"""
    # 設置中文字體
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 固定亂數種子，讓每次結果可重現
    np.random.seed(43)

    # 生成標準二維高斯散點（第一張保持原始位置）
    base_points = np.random.multivariate_normal(mean=MU, cov=COV, size=N_SAMPLES)
    points = base_points

    # 第二張：先做縮放+旋轉，再整體平移 (+1, +1)
    transformed_points = transform_points(base_points, y_scale=0.5, ccw_deg=30) + SHIFT

    # 建立兩張子圖
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=160)
    ax1, ax2 = axes

    # 第一張：標準二維高斯散點
    ax1.scatter(points[:, 0], points[:, 1], s=24, color="navy", alpha=0.85)
    ax1.axhline(0, color="#444", linewidth=1.2)
    ax1.axvline(0, color="#444", linewidth=1.2)
    ax1.grid(True, linestyle=(0, (2, 2)), alpha=0.45)
    ax1.set_aspect("equal", "box")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # 第二張：變換後散點
    ax2.scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        s=24,
        color="navy",
        alpha=0.85,
    )
    ax2.axhline(0, color="#444", linewidth=1.2)
    ax2.axvline(0, color="#444", linewidth=1.2)
    ax2.grid(True, linestyle=(0, (2, 2)), alpha=0.45)
    ax2.set_aspect("equal", "box")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-4, 4)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    # 保存圖表
    plt.tight_layout(pad=1.4)
    plt.savefig(
        "gaussian_2d_transform.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 顯示圖表
    plt.show()

    print("圖表已生成並保存為 'gaussian_2d_transform.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"繪圖錯誤: {e}")
    finally:
        plt.close("all")  # 清理記憶體
