import numpy as np
import matplotlib.pyplot as plt

# 设置 near 和 far 的值
near = 1
far = 50

# 创建 z 的范围
z = np.linspace(near, far, 100)

# 计算 F_depth
F_depth = (z - near) / (far - near)

# 绘制折线图
plt.figure(figsize=(7, 4))

plt.plot(z, F_depth, label="near = 1, far = 50", color="red", linestyle="--")
plt.title("near = 1, far = 50")
plt.xlabel("Z-value", color="blue")
plt.ylabel("Depth value", color="green")
plt.xlim(1, far)  # 設置 x 軸範圍從 1 開始
plt.ylim(0, 1)
plt.tick_params(axis="x", colors="blue")  # x 軸 ticks 和 labels 顏色設置為藍色
plt.tick_params(axis="y", colors="green")  # y 軸 ticks 和 labels 顏色設置為綠色
plt.xticks(np.concatenate(([1], np.arange(5, 51, 5))))
plt.yticks(np.arange(0.1, 1.1, 0.1))
plt.show()
