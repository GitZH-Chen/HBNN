"""
Visualize the Lorentz-BFC solvability interval.

This script plots the lower bound function
    a_min(b; m) = sqrt((m-1)*(1 + m*b^2)) / m
for several fixed m values as b varies from 1 to 4.

The figure helps analyze how the interval [a_min, b] depends on m and b in the Lorentz Busemann FC solvability condition.

Usage:
    Run this file directly to display the plot.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- 参数设置 ----
b = np.linspace(1.0, 4.0, 300)  # b 从 1 到 4
m_values = [4, 8, 16, 32, 64, 128, 256, 512]   # 固定的一组 m
# ------------------

plt.figure(figsize=(7,5))

for m in m_values:
    a_min = np.sqrt((m - 1) * (1 + m * b**2)) / m
    plt.plot(b, a_min, label=f"m={m}")

plt.xlabel("b", fontsize=12)
plt.ylabel(r"$a_{\min}(b;m)$", fontsize=12)
plt.title(r"$a_{\min}(b;m)=\frac{\sqrt{(m-1)(1+m b^2)}}{m}$", fontsize=13)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
