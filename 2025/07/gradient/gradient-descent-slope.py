import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def f(x):
    """定義要最小化的函數 f(x) = (x-5)^2 + 10"""
    return (x - 5) ** 2 + 10


def f_prime(x):
    """函數的導數 f'(x) = 2(x-5)"""
    return 2 * (x - 5)


def gradient_descent_step(x, learning_rate=0.1):
    """執行一步梯度下降"""
    gradient = f_prime(x)
    new_x = x - learning_rate * gradient
    return new_x, gradient


def plot_gradient_descent_iteration(x_current, iteration, ax):
    """畫出單次迭代的梯度下降視覺化"""
    ax.clear()

    # 設定 x 軸範圍
    x_min, x_max = -2, 12
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = f(x_vals)

    # 畫出函數曲線（黑線）
    ax.plot(x_vals, y_vals, "k-", linewidth=3, label="f(x) = (x-5)² + 10")

    # 計算當前點的值和梯度
    y_current = f(x_current)
    gradient = f_prime(x_current)

    # 畫出當前點（紅點）
    ax.plot(
        x_current, y_current, "ro", markersize=10, label=f"當前點 x={x_current:.2f}"
    )

    # 畫出切線（藍線）
    # 切線方程：y = f'(x_current) * (x - x_current) + f(x_current)
    tangent_x = np.array([x_current - 3, x_current + 3])
    tangent_y = gradient * (tangent_x - x_current) + y_current
    ax.plot(
        tangent_x,
        tangent_y,
        "b-",
        linewidth=2,
        alpha=0.8,
        label=f"切線 (斜率 = {gradient:.2f})",
    )

    # 畫出水平參考線（紅線）
    ax.axhline(
        y=y_current,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"當前函數值 = {y_current:.2f}",
    )

    # 標示梯度方向
    if gradient > 0:
        # 正梯度，箭頭指向右上方
        ax.annotate(
            "",
            xy=(x_current + 1, y_current + gradient),
            xytext=(x_current, y_current),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2),
        )
        ax.text(
            x_current + 1.2,
            y_current + gradient + 0.5,
            f"梯度 = {gradient:.2f}\n(正數)",
            color="blue",
            fontsize=10,
        )
    else:
        # 負梯度，箭頭指向左下方
        ax.annotate(
            "",
            xy=(x_current - 1, y_current + gradient),
            xytext=(x_current, y_current),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2),
        )
        ax.text(
            x_current - 1.2,
            y_current + gradient - 0.5,
            f"梯度 = {gradient:.2f}\n(負數)",
            color="blue",
            fontsize=10,
        )

    # 標示下一步位置
    next_x = x_current - 0.1 * gradient
    next_y = f(next_x)
    ax.plot(next_x, next_y, "go", markersize=8, label=f"下一步 x={next_x:.2f}")

    # 畫出移動箭頭
    ax.annotate(
        "",
        xy=(next_x, next_y),
        xytext=(x_current, y_current),
        arrowprops=dict(arrowstyle="->", color="green", lw=2, alpha=0.7),
    )

    # 添加說明文字
    explanation = f"""
    梯度下降原理：
    1. 當前位置：x = {x_current:.2f}
    2. 函數值：f(x) = {y_current:.2f}
    3. 梯度：f'(x) = {gradient:.2f}
    4. 更新公式：x_new = x - 學習率 × 梯度
    5. 下一步：x_new = {x_current:.2f} - 0.1 × {gradient:.2f} = {next_x:.2f}
    
    為什麼要減掉梯度？
    • 當梯度 > 0 時，函數在增加，需要向左移動（減掉正數）
    """

    ax.text(
        0.02,
        0.98,
        explanation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    # ax.set_title(f"梯度下降視覺化")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 50)


def create_gradient_descent_animation():
    """創建梯度下降的動畫"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 初始點
    x_start = 8.0
    x_current = x_start

    def animate(frame):
        nonlocal x_current
        if frame == 0:
            x_current = x_start
        else:
            x_current, _ = gradient_descent_step(x_current)

        plot_gradient_descent_iteration(x_current, frame, ax)
        return (ax,)

    # 創建動畫
    anim = FuncAnimation(fig, animate, frames=10, interval=2000, repeat=True)
    plt.tight_layout()
    plt.show()

    return anim


def plot_multiple_iterations():
    """畫出多次迭代的靜態圖"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    x_start = 8.0
    x_current = x_start

    for i in range(6):
        ax = axes[i]
        plot_gradient_descent_iteration(x_current, i, ax)

        # 更新到下一個位置
        if i < 5:
            x_current, _ = gradient_descent_step(x_current)

    plt.tight_layout()
    plt.show()


def plot_convergence_explanation():
    """畫出收斂過程的解釋圖"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左圖：函數曲線和梯度方向
    x_vals = np.linspace(-2, 12, 200)
    y_vals = f(x_vals)

    ax1.plot(x_vals, y_vals, "k-", linewidth=3, label="f(x) = (x-5)² + 10")

    # 標示不同區域的梯度方向
    x_left = np.linspace(-2, 5, 50)
    x_right = np.linspace(5, 12, 50)

    # 左側區域（梯度為負）
    for x in x_left[::10]:
        y = f(x)
        grad = f_prime(x)
        ax1.arrow(
            x,
            y,
            -0.5,
            grad * (-0.5),
            head_width=0.2,
            head_length=0.1,
            fc="red",
            ec="red",
            alpha=0.6,
        )

    # 右側區域（梯度為正）
    for x in x_right[::10]:
        y = f(x)
        grad = f_prime(x)
        ax1.arrow(
            x,
            y,
            -0.5,
            grad * (-0.5),
            head_width=0.2,
            head_length=0.1,
            fc="blue",
            ec="blue",
            alpha=0.6,
        )

    # 標示最小值點
    ax1.plot(5, f(5), "go", markersize=12, label="最小值點 (5, 10)")

    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("梯度方向指示")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右圖：梯度值隨 x 的變化
    x_vals = np.linspace(-2, 12, 200)
    grad_vals = f_prime(x_vals)

    ax2.plot(x_vals, grad_vals, "b-", linewidth=3, label="f'(x) = 2(x-5)")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.axvline(x=5, color="g", linestyle="--", alpha=0.5, label="x = 5 (最小值)")

    # 標示不同區域
    ax2.fill_between(
        x_vals,
        grad_vals,
        0,
        where=(grad_vals < 0),
        alpha=0.3,
        color="red",
        label="負梯度區域",
    )
    ax2.fill_between(
        x_vals,
        grad_vals,
        0,
        where=(grad_vals > 0),
        alpha=0.3,
        color="blue",
        label="正梯度區域",
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("f'(x)")
    ax2.set_title("梯度值變化")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("梯度下降視覺化演示")
    print("=" * 50)

    # 1. 顯示單次迭代的詳細視覺化
    print("1. 顯示單次迭代的詳細視覺化...")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_gradient_descent_iteration(8.0, 1, ax)
    plt.show()

    # 2. 顯示多次迭代的靜態圖
    # print("2. 顯示多次迭代的靜態圖...")
    # plot_multiple_iterations()

    # 3. 顯示收斂過程的解釋
    # print("3. 顯示收斂過程的解釋...")
    # plot_convergence_explanation()

    # 4. 創建動畫（可選）
    # print("4. 創建動畫演示...")
    # create_gradient_descent_animation()

    print("演示完成！")
