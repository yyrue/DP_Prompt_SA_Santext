#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 KNN Attack 结果：epsilon vs defense_rate
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv("knn_attack_results_SST2_v2.csv")

# 计算每个 epsilon 的平均值和标准差
stats = df.groupby('epsilon')['defense_rate'].agg(['mean', 'std']).reset_index()
stats.columns = ['epsilon', 'defense_rate_mean', 'defense_rate_std']

print("=" * 50)
print("各 epsilon 下 Defense Rate 统计")
print("=" * 50)
for _, row in stats.iterrows():
    print(f"ε={row['epsilon']:4.1f}: Defense Rate = {row['defense_rate_mean']:.4f} ± {row['defense_rate_std']:.4f}")
print("=" * 50)

# 绘图
plt.figure(figsize=(10, 6))

# 主曲线（带误差棒）
plt.errorbar(
    stats['epsilon'],
    stats['defense_rate_mean'],
    yerr=stats['defense_rate_std'],
    fmt='o-',           # 圆点+实线
    color='#1f77b4',    # 蓝色
    linewidth=2,
    markersize=8,
    capsize=4,          # 误差棒端帽大小
    capthick=1.5,
    label='Defense Rate (KNN Attack)'
)

# 标签和标题
plt.xlabel('Privacy Budget ε', fontsize=14)
plt.ylabel('Defense Rate', fontsize=14)
plt.title('KNN Attack: Defense Rate vs Privacy Budget (SST-2, BERT Embedding)', fontsize=14)

# X 轴刻度
plt.xticks(stats['epsilon'], fontsize=11)
plt.yticks(fontsize=11)

# 网格
plt.grid(True, linestyle='--', alpha=0.6)

# 图例
plt.legend(loc='upper right', fontsize=12)

# Y 轴范围
plt.ylim(-0.02, 1.05)

# 紧凑布局
plt.tight_layout()

# 保存图片
output_path = "knn_attack_defense_rate.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存至: {output_path}")

plt.show()