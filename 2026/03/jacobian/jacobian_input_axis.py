#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
繪製與附圖類似的輸入座標軸：
紅色：a 對應位置 x=1.5
黃金色：x=0.5 間隔的刻度線與刻度數字
a：標示在 x=1.5

輸出：`jacobian_input_axis.png`（同目錄）
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

    ax.set_title(r"$f(x) = x$", fontsize=28, color="white", pad=8, y=0.78)

    # 刻度設定（1.1 到 1.9，步長 0.1；包含 a=1.5）
    ticks = [round(1.1 + i * 0.1, 1) for i in range(9)]
    red_x = 1.5
    blue_marks = {1.4, 1.6}
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
            color = red
        elif t in blue_marks:
            color = sky_blue
        else:
            color = gold
        ax.vlines(t, -tick_half, tick_half, colors=color, linewidth=3)

        # 刻度數字放在軸線下方
        label_y = -0.10
        ax.text(
            t,
            label_y,
            format_tick_value(t),
            color=color,
            ha="center",
            va="top",
            fontsize=18,
        )

    # 在 x=1.5 額外標示 a（放在刻度線下方）
    ax.text(
        1.5,
        -0.155,
        "a",
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

    out_path = os.path.join(os.path.dirname(__file__), "jacobian_input_axis.png")
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
