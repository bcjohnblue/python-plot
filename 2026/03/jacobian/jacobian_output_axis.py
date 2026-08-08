#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
繪製與附圖類似的輸出座標軸：
- 紅線中心在 x=1.5，a 標在紅線下方
- 每個 0.5 間隔點用虛黃線表示（除紅線）
- 在紅線左右第 3 個虛黃線稍右側加實黃線

輸出：`jacobian_output_axis.png`（同目錄）
"""

import os

import matplotlib.pyplot as plt


def format_tick_value(v: float) -> str:
    """將刻度格式成 1 位小數。"""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def main():
    # 設置中文字體（若系統沒有支援，也不會影響刻度數字/英文字母 a）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 版面配置：讓圖寬度接近附圖、背景為純黑
    fig, ax = plt.subplots(figsize=(14, 3.2))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_title(r"$f(x)=x^2$", fontsize=28, color="white", pad=8, y=0.78)

    # 刻度設定（1.1 到 1.9，步長 0.1）
    ticks = [round(1.1 + i * 0.1, 1) for i in range(9)]
    red_x = 1.5
    red = "#ff0000"
    gold = "#FFD700"
    sky_blue = "#00BFFF"
    white = "#ffffff"

    # 畫水平軸
    ax.axhline(0, color=white, linewidth=2.2)

    # 刻度線以水平軸為中心上下對稱，並縮短長度
    tick_half = 0.045

    for t in ticks:
        if abs(t - red_x) < 1e-9:
            # 紅線：中心參考線（實線）
            ax.vlines(t, -tick_half, tick_half, colors=red, linewidth=3)
            color = red
        else:
            # 一般間隔點：虛黃線
            ax.vlines(
                t,
                -tick_half,
                tick_half,
                colors=gold,
                linewidth=2.8,
                linestyles="--",
            )
            color = gold

        # 刻度數字放在軸線下方
        label_y = -0.10
        ax.text(
            t,
            label_y,
            "2.25" if abs(t - red_x) < 1e-9 else format_tick_value(t),
            color=color,
            ha="center",
            va="top",
            fontsize=18,
        )

    # 額外藍虛線標記
    for t in (1.21, 1.81):
        ax.vlines(
            t,
            -tick_half,
            tick_half,
            colors=sky_blue,
            linewidth=2.8,
        )

    # 藍線數值標注
    ax.text(1.21, -0.155, "1.96", color=sky_blue, ha="center", va="top", fontsize=16)
    ax.text(1.81, -0.155, "2.56", color=sky_blue, ha="center", va="top", fontsize=16)

    # 在紅線下方標示 a
    ax.text(
        red_x,
        -0.155,
        "f(a)",
        color=gold,
        ha="center",
        va="top",
        fontsize=28,
        fontstyle="italic",
    )

    # 清理座標軸細節，避免黑底上出現多餘白邊
    # 顯示區間聚焦在 1.1~1.9（左右留少許邊距）
    ax.set_xlim(1.08, 1.92)
    ax.set_ylim(-0.22, 0.22)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 盡量貼齊內容邊界
    plt.tight_layout(pad=0.2)

    out_path = os.path.join(os.path.dirname(__file__), "jacobian_output_axis.png")
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    print(f"已生成：{os.path.abspath(out_path)}")

    # 直接顯示視窗：在有 GUI backend（如 TkAgg/MacOSX）時會開窗；無 GUI 時通常不會真正顯示。
    plt.show()


if __name__ == "__main__":
    try:
        main()
    finally:
        plt.close("all")
