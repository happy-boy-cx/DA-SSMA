import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
df = pd.read_csv('ablationx.csv', index_col=0)
methods = df.index.tolist()
ratios = df.columns.tolist()

# 样式配置
colors = ['#4E79A7', '#E69442', '#E15759', '#76B7B2', '#59A14F']  # 五色系
bar_width = 0.12  # 柱宽微调
x_base = np.arange(0, len(ratios) * 0.7, 0.7)  # 步长为 0.8

# 创建画布
plt.figure(figsize=(8, 6))
ax = plt.gca()

# 绘制柱状图
for idx, method in enumerate(methods):
    offset = bar_width * (idx - 2)  # 对称偏移计算
    ax.bar(x_base + offset, df.loc[method],
          width=bar_width, color=colors[idx],
          label=method, edgecolor='white')

# 装饰元素
ax.set_xticks(x_base)
ax.set_xticklabels(ratios, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)

# 图例位置优化
legend = ax.legend(
    title="Methods",
    loc="upper left",
    bbox_to_anchor=(0.01, 0.98),
    frameon=True,
    shadow=True,
    fontsize=8
)
legend.get_title().set_fontsize(12)

# 网格样式
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()