#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公式 15、16 的「四格拆解」推導圖。

  公式 15：x₀ = u₀/u₂,  x₁ = u₁/u₂,  x₂ = ‖u‖
  公式 16：l = ‖(x₀,x₁,1)‖,  u = (x₂/l)(x₀,x₁,1)

四個子圖：
  ① u₀-u₂ 側視：相似三角形解釋 x₀ = u₀/u₂
  ② u₁-u₂ 側視：相似三角形解釋 x₁ = u₁/u₂
  ③ 3D 全景：標出 x₂ = ‖u‖（徑向距離）與投影平面 u₂ = 1
  ④ 3D 反向：由 (x₀,x₁,1) → 單位化 → 乘 x₂ 還原 u

本程式只產生四張獨立 PNG（與本檔同目錄）：

  - `camera_ray_derivation_4panel_01_u0_u2.png` — ①
  - `camera_ray_derivation_4panel_02_u1_u2.png` — ②
  - `camera_ray_derivation_4panel_03_x2_forward.png` — ③
  - `camera_ray_derivation_4panel_04_inverse.png` — ④

執行：`python3 camera_ray_derivation_4panel.py`

- 可選 `--dpi`。
- 預設在存完四個 PNG 後，依序**彈出四個視窗**（關掉目前視窗才會出現下一張）。
- 若只想要檔、不要彈窗：加 `--no-show` 或 `MPLBACKEND=Agg python3 ...`。
- 偵測到無圖形後端（如 Agg）且未加 `--no-show` 時，只會在第一次印一行提示，仍正常寫檔。
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def forward_m(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(3)
    if u[2] <= 0:
        raise ValueError("需 u₂ > 0")
    return np.array([u[0] / u[2], u[1] / u[2], np.linalg.norm(u)], dtype=float)


def inverse_m(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(3)
    v = np.array([x[0], x[1], 1.0], dtype=float)
    l = np.linalg.norm(v)
    return (x[2] / l) * v





def _draw_similar_triangles_2d(
    ax, ui: float, u2: float, axis_label: str, xi_label: str
) -> None:
    """畫一個側視圖：水平軸為 u_i，垂直軸為 u₂。

    顯示兩個相似三角形：
      大三角形 O–(0,u₂)–(u_i,u₂)
      小三角形 O–(0,1)–(x_i,1)
    """
    xi = ui / u2

    pad_h = max(abs(ui), abs(xi)) * 0.5 + 0.25
    pad_v = 0.4
    ax.set_xlim(-pad_h, max(ui, xi) + pad_h)
    ax.set_ylim(-0.15, u2 + pad_v)

    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)

    # 投影平面 u₂ = 1（一條水平線）
    ax.axhline(
        1.0, color="#2F8F88", linewidth=1.2, alpha=0.7, label=r"投影平面 $u_2=1$"
    )

    # 大三角形（u 所在）
    big = np.array([[0, 0], [0, u2], [ui, u2], [0, 0]])
    ax.plot(
        big[:, 0],
        big[:, 1],
        color="#E63946",
        linewidth=2.2,
        label=r"大三角形 ($\mathbf{u}$)",
    )
    ax.fill(big[:, 0], big[:, 1], color="#E63946", alpha=0.08)

    # 小三角形（投影到 u₂=1）
    small = np.array([[0, 0], [0, 1.0], [xi, 1.0], [0, 0]])
    ax.plot(
        small[:, 0],
        small[:, 1],
        color="#457B9D",
        linewidth=2.2,
        label=r"小三角形 ($x_i$)",
    )
    ax.fill(small[:, 0], small[:, 1], color="#457B9D", alpha=0.18)

    # 從原點往 u 的射線（延伸線）
    t = np.linspace(0, 1.15, 2)
    ax.plot(t * ui, t * u2, color="#555555", linestyle="--", linewidth=1.2)

    # 點與標註
    ax.scatter([0, ui, xi, 0, 0], [0, u2, 1.0, u2, 1.0], color="black", s=22, zorder=5)
    ax.annotate(
        r"$O$", (0, 0), textcoords="offset points", xytext=(-12, -10), fontsize=11
    )
    ax.annotate(
        rf"$({axis_label}, u_2)=({ui:.2f},{u2:.2f})$",
        (ui, u2),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=10,
        color="#9b1d2b",
    )
    ax.annotate(
        rf"$({xi_label},1)=({xi:.3f},1)$",
        (xi, 1.0),
        textcoords="offset points",
        xytext=(8, -14),
        fontsize=10,
        color="#1d3a52",
    )

    # 文字：相似三角形比例
    ax.text(
        -pad_h * 0.45,
        u2 * 0.45,
        rf"相似三角形：$\dfrac{{{xi_label}}}{{1}}=\dfrac{{{axis_label}}}{{u_2}}$"
        + "\n"
        + rf"$\Rightarrow {xi_label}={axis_label}/u_2={xi:.3f}$",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e6", ec="#d4a017", alpha=0.95),
    )

    ax.set_xlabel(rf"${axis_label}$")
    ax.set_ylabel(r"$u_2$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(rf"側視圖（${axis_label}$-$u_2$ 平面）：透視除法得到 ${xi_label}$")


def _setup_3d(ax, u: np.ndarray, p: np.ndarray, ray_end: np.ndarray) -> None:
    # 先計算資料範圍與座標軸邊界
    all_pts = np.vstack([np.zeros(3), u, p, ray_end])
    span = np.ptp(all_pts, axis=0)
    margin = 0.1 * np.maximum(span, 0.25)
    mins = all_pts.min(axis=0) - margin
    maxs = all_pts.max(axis=0) + margin
    
    # 將平面的範圍設定成與座標軸顯示範圍完全一致，才不會超出邊框產生錯覺
    u0_min, u0_max = mins[0], maxs[0]
    u1_min, u1_max = mins[1], maxs[1]

    # --- 投影平面 u₂=1：加上半透明背景色與網格線 ---
    U0, U1 = np.meshgrid([u0_min, u0_max], [u1_min, u1_max])
    Z = np.ones_like(U0)
    ax.plot_surface(U0, U1, Z, color="#4ECDC4", alpha=0.15, shade=False)

    # 邊框
    border_u0 = [u0_min, u0_max, u0_max, u0_min, u0_min]
    border_u1 = [u1_min, u1_min, u1_max, u1_max, u1_min]
    border_z = [1.0] * 5
    ax.plot(border_u0, border_u1, border_z,
            color="#2F8F88", linewidth=1.4, alpha=0.8)

    # 內部網格線（沿 u0 方向）
    n_grid = 8
    for v in np.linspace(u1_min, u1_max, n_grid + 1):
        ax.plot([u0_min, u0_max], [v, v], [1.0, 1.0],
                color="#4ECDC4", linewidth=0.4, alpha=0.45)
    # 內部網格線（沿 u1 方向）
    for v in np.linspace(u0_min, u0_max, n_grid + 1):
        ax.plot([v, v], [u1_min, u1_max], [1.0, 1.0],
                color="#4ECDC4", linewidth=0.4, alpha=0.45)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel(r"$u_0$")
    ax.set_ylabel(r"$u_1$")
    ax.set_zlabel(r"$u_2$")
    ax.view_init(elev=22, azim=-58)


def _draw_panel3_forward(ax, u: np.ndarray) -> None:
    """3D 全景：強調 x₂ = ‖u‖（徑向距離）。"""
    x = forward_m(u)
    p = np.array([x[0], x[1], 1.0])
    o = np.zeros(3)
    ray_end = 1.15 * u

    _setup_3d(ax, u, p, ray_end)

    ax.plot(*zip(o, ray_end), color="#888888", linestyle="--", linewidth=1.2)
    ax.plot(*zip(o, p), color="#457B9D", linewidth=2.2)
    ax.plot(*zip(o, u), color="#E63946", linewidth=3.0)

    ax.scatter(*o, color="black", s=45, depthshade=False)
    ax.scatter(*u, color="#E63946", s=70, depthshade=False)
    ax.scatter(*p, color="#457B9D", s=70, depthshade=False)

    # 將 u 文字精確往左上方偏移（-y 往左，+z 往上）
    ax.text(u[0], u[1] - 0.18, u[2] + 0.15, r"$\mathbf{u}$", fontsize=12)
    # 將 u 的座標精確標示在正下方（往下 -z）
    ax.text(u[0], u[1], u[2] - 0.15, r"$(u_0, u_1, u_2)$", fontsize=10, color="#E63946")
    ax.text(p[0] - 0.14, p[1] - 0.08, p[2] + 0.10, r"$(x_0,x_1,x_2)$", fontsize=10)

    ax.set_title(r"3D 全景")


def _draw_panel4_inverse(ax, u: np.ndarray) -> None:
    """3D 反向：由 (x₀,x₁,1) → 單位化 → 乘 x₂ 還原 u。"""
    x = forward_m(u)
    p = np.array([x[0], x[1], 1.0])
    o = np.zeros(3)
    l = float(np.linalg.norm(p))
    unit_dir = p / l
    u_back = x[2] * unit_dir
    ray_end = 1.15 * u

    _setup_3d(ax, u, p, ray_end)

    ax.plot(
        *zip(o, p),
        color="#457B9D",
        linewidth=2.0,
        label=rf"$(x_0,x_1,1)$，長度 $l={l:.3f}$",
    )
    ax.plot(
        *zip(o, unit_dir),
        color="#F4A261",
        linewidth=2.4,
        label=r"$(x_0,x_1,1)/l$（單位向量）",
    )
    ax.plot(
        *zip(o, u_back),
        color="#E63946",
        linewidth=3.0,
        alpha=0.9,
        label=r"$\dfrac{x_2}{l}(x_0,x_1,1)=\mathbf{u}$",
    )

    ax.scatter(*o, color="black", s=45, depthshade=False)
    ax.scatter(*p, color="#457B9D", s=60, depthshade=False)
    ax.scatter(*unit_dir, color="#F4A261", s=60, depthshade=False)
    ax.scatter(*u_back, color="#E63946", s=80, depthshade=False)

    # 將文字向左上（-x, -y, +z 方向）偏移，避免被線條遮擋
    ax.text(p[0] - 0.14, p[1] - 0.08, p[2] + 0.10, r"$(x_0,x_1,1)$", fontsize=10)
    ax.text(
        unit_dir[0] - 0.14,
        unit_dir[1] - 0.08,
        unit_dir[2] + 0.10,
        r"單位方向",
        fontsize=10,
        color="#a35a13",
    )
    ax.text(
        u_back[0] - 0.14,
        u_back[1] - 0.08,
        u_back[2] + 0.12,
        r"$\mathbf{u}$（還原）",
        fontsize=11,
        color="#9b1d2b",
    )

    err = float(np.linalg.norm(u_back - u))
    ax.text2D(
        0.02,
        0.97,
        rf"驗證：$\|m^{{-1}}(m(\mathbf{{u}}))-\mathbf{{u}}\|={err:.2e}$",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f0f7ff", ec="#457B9D", alpha=0.9),
    )

    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(r"反向（公式 16）：單位化方向 $\times\, x_2$ 還原 $\mathbf{u}$")




def _can_show_interactive() -> bool:
    """有互動圖形後端時可 `plt.show()` 彈窗；否則只適合寫檔。"""
    b = matplotlib.get_backend().lower()
    for token in (
        "agg",
        "template",
        "ps",
        "svg",
        "pdf",
        "cairop",
        "inline",
    ):
        if token in b:
            return False
    return True


def save_separate_panels(
    u: np.ndarray, base_dir: str, *, dpi: int = 200, show: bool = True
) -> list[str]:
    """各畫一張圖、各自存成 PNG。若 `show` 則在可彈窗時，每張圖關掉後顯示下一張。

    回傳已寫入的絕對路徑清單。
    """
    out_paths: list[str] = []
    _warned_no_gui = False
    if show and _can_show_interactive():
        print("將依序彈出四個視窗；關閉目前視窗後顯示下一張。\n")
    pre = "camera_ray_derivation_4panel"
    spec = [
        (f"{pre}_01_u0_u2.png", 6.0, 5.2, "2d0"),
        (f"{pre}_02_u1_u2.png", 6.0, 5.2, "2d1"),
        (f"{pre}_03_x2_forward.png", 11.5, 7.0, "3d_fwd"),
    ]
    for fname, w, h, which in spec:
        fig = plt.figure(figsize=(w, h))
        if which == "2d0":
            ax = fig.add_subplot(111)
            _draw_similar_triangles_2d(
                ax,
                ui=float(u[0]),
                u2=float(u[2]),
                axis_label="u_0",
                xi_label="x_0",
            )
        elif which == "2d1":
            ax = fig.add_subplot(111)
            _draw_similar_triangles_2d(
                ax,
                ui=float(u[1]),
                u2=float(u[2]),
                axis_label="u_1",
                xi_label="x_1",
            )
        elif which == "3d_fwd":
            ax = fig.add_subplot(111, projection="3d")
            _draw_panel3_forward(ax, u)
        plt.tight_layout()
        p = os.path.join(base_dir, fname)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        ap = os.path.abspath(p)
        out_paths.append(ap)
        print(f"已寫入：{ap}")
        if show and _can_show_interactive():
            # 每次阻塞直到使用者關閉視窗，再顯示下一張
            plt.show()
        elif show and not _warned_no_gui:
            print(
                "（目前圖形後端無法彈窗，例如 Agg；已略過視窗。）"
            )
            _warned_no_gui = True
        plt.close(fig)
    return out_paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="產生四張子圖 PNG（攝影機空間 → 射線空間 公式 15/16）"
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="輸出圖片 DPI（預設 200）",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="只寫出 PNG、不彈出視窗",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    u = np.array([0.5, 0.9, 1.2], dtype=float)
    x = forward_m(u)
    print(f"u             = {u}")
    print(f"x = m(u)      = {x}")
    print(f"m^-1(m(u))    = {inverse_m(x)}")

    base_dir = os.path.dirname(__file__)
    save_separate_panels(
        u,
        base_dir,
        dpi=int(args.dpi),
        show=not args.no_show,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        plt.close("all")
