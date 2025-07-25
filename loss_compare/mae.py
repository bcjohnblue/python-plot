import numpy as np
import matplotlib.pyplot as plt

# 定義 y - f(x) 的範圍
x = np.linspace(-10, 10, 200)
# 平均絕對誤差 (MAE) 計算公式
mae = np.abs(x)

plt.plot(x, mae)
plt.xlabel(r"$y_i - \hat{y}_i$")
plt.ylabel("MAE")
plt.title("Mean Absolute Error (MAE)")
plt.grid(True)
plt.show()
