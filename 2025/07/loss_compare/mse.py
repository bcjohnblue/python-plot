import numpy as np
import matplotlib.pyplot as plt

# 定義 y - f(x) 的範圍
x = np.linspace(-10, 10, 200)
# 均方誤差 (MSE) 計算公式
mse = x**2

plt.plot(x, mse)
plt.xlabel(r"$y_i - \hat{y}_i$")
plt.ylabel("MSE")
plt.title("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()
