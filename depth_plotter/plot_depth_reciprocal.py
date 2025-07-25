import numpy as np
import matplotlib.pyplot as plt

# 设置 near 和 far 的值
near = 1
far = 50
# near = 10**-6
# far = 10**19


# 创建 z 的范围
z = np.linspace(near, far, 100)
# z = np.logspace(-6, 19, 500)

# 计算 F_depth
F_depth = (1 / z - 1 / near) / (1 / far - 1 / near)

# 绘制折线图
plt.figure(figsize=(8, 5))

plt.plot(z, F_depth, label="near = 1, far = 50", color="red", linestyle="--")
plt.title("near = 1, far = 50")
# plt.plot(z, F_depth, label="near = 10^-6, far = 10^19", color="red", linestyle="--")
# plt.title("near = 10^-6, far = 10^19")

# plt.xscale("log")  # 设置 x 轴为对数刻度

plt.xlabel("Z-value", color="blue")
plt.ylabel("Depth value", color="green")
plt.xlim(1, far)  # 設置 x 軸範圍從 1 開始
plt.ylim(0, 1)
plt.tick_params(axis="x", colors="blue")  # x 軸 ticks 和 labels 顏色設置為藍色
plt.tick_params(axis="y", colors="green")  # y 軸 ticks 和 labels 顏色設置為綠色

# xticks = [10**-6, 10**-3, 10**0, 10**3, 10**6, 10**9, 10**12, 10**15, 10**18]
# xtick_labels = [
#     r"$10^{-6}$",
#     r"$10^{-3}$",
#     r"$10^{0}$",
#     r"$10^{3}$",
#     r"$10^{6}$",
#     r"$10^{9}$",
#     r"$10^{12}$",
#     r"$10^{15}$",
#     r"$10^{18}$",
# ]
# plt.xticks(xticks, xtick_labels)  # 设置 X 轴刻度和标签

plt.axvline(x=10, color="blue", linestyle="dotted", linewidth=1)

plt.xticks(np.concatenate(([1], np.arange(5, 51, 5))))
plt.yticks(np.arange(0.1, 1.1, 0.1))
plt.show()
