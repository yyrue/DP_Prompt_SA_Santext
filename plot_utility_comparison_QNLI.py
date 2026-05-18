#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Utility (Accuracy) 对比图 — QNLI 数据集
- X 轴: epsilon
- Y 轴: Accuracy
- 对比线:
  1. Baseline（基于指数机制）
  2. Mixed (ε_high=16)
  3. Mixed (ε_high=18)
  4. Mixed (ε_high=20)
  5. Mixed (ε_high=22)
  6. Mixed (ε_high=24)
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# 1. 数据读取工具函数
# ============================================================

def load_avg(csv_path, key_col="epsilon", val_col="accuracy"):
    """按 epsilon 分组，计算 accuracy 的均值和标准差"""
    groups = defaultdict(list)
    if not os.path.exists(csv_path):
        print(f"[WARN] 文件不存在，跳过: {csv_path}")
        return {}, {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = float(row[key_col])
                v = float(row[val_col])
                groups[k].append(v)
            except (ValueError, KeyError):
                continue
    result_mean = {k: np.mean(v) for k, v in groups.items()}
    result_std  = {k: np.std(v, ddof=1) if len(v) > 1 else 0.0 for k, v in groups.items()}
    return result_mean, result_std


# ============================================================
# 2. 数据文件路径
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

baseline_csv = os.path.join(PROJECT_DIR, "utility_results_QNLI_bert.csv")
mixed_csvs = {
    "Mixed (ε_high=16)": os.path.join(PROJECT_DIR, "utility_results_QNLI_bert_mixed_eps16.csv"),
    "Mixed (ε_high=18)": os.path.join(PROJECT_DIR, "utility_results_QNLI_bert_mixed_eps18.csv"),
    "Mixed (ε_high=20)": os.path.join(PROJECT_DIR, "utility_results_QNLI_bert_mixed_eps20.csv"),
    "Mixed (ε_high=22)": os.path.join(PROJECT_DIR, "utility_results_QNLI_bert_mixed_eps22.csv"),
    "Mixed (ε_high=24)": os.path.join(PROJECT_DIR, "utility_results_QNLI_bert_mixed_eps24.csv"),
}

# ============================================================
# 3. 加载数据
# ============================================================

# Baseline 数据
baseline_mean, baseline_std = load_avg(baseline_csv)

# Mixed 数据
mixed_data = {}
for label, csv_path in mixed_csvs.items():
    mean, std = load_avg(csv_path)
    mixed_data[label] = (mean, std)

# ============================================================
# 4. 收集每条线各自的所有 epsilon 值
# ============================================================

# Baseline 的所有 epsilon（排除 0）
baseline_epsilons = sorted([e for e in baseline_mean.keys() if e >= 0])

print("Baseline epsilon 值:", baseline_epsilons)
for label, (mean, std) in mixed_data.items():
    all_eps = sorted(mean.keys())
    print(f"{label} epsilon 值: {all_eps}")

# 收集所有出现过的 epsilon 值，用于 X 轴刻度
all_epsilons = set(baseline_epsilons)
for label, (mean, std) in mixed_data.items():
    all_epsilons.update(mean.keys())
all_epsilons = sorted(all_epsilons)

# ============================================================
# 5. 画图
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

# 颜色和标记设置
styles = {
    "Baseline":           {"color": "#1f77b4", "marker": "o",  "linestyle": "-"},
    "Mixed (ε_high=16)": {"color": "#ff7f0e", "marker": "s",  "linestyle": "--"},
    "Mixed (ε_high=18)": {"color": "#2ca02c", "marker": "^",  "linestyle": "--"},
    "Mixed (ε_high=20)": {"color": "#d62728", "marker": "D",  "linestyle": "--"},
    "Mixed (ε_high=22)": {"color": "#8c564b", "marker": "p",  "linestyle": "--"},
    "Mixed (ε_high=24)": {"color": "#9467bd", "marker": "v",  "linestyle": "--"},
}

# 画 Baseline 线（画出所有 epsilon 数据点）
eps_list = baseline_epsilons
means = [baseline_mean[e] for e in eps_list]
stds  = [baseline_std[e] for e in eps_list]
style = styles["Baseline"]
ax.errorbar(eps_list, means, yerr=stds,
            label="Baseline",
            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            linewidth=2, markersize=8, capsize=3, capthick=1.5)

# 画 Mixed 线（每条线画出自己所有的 epsilon 数据点）
for label, (mean, std) in mixed_data.items():
    # 使用该数据集自己的所有 epsilon 值
    own_eps = sorted(mean.keys())
    if not own_eps:
        print(f"[WARN] {label} 没有数据，跳过")
        continue

    m = [mean[e] for e in own_eps]
    s = [std[e] for e in own_eps]
    style = styles[label]
    ax.errorbar(own_eps, m, yerr=s,
                label=label,
                color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                linewidth=2, markersize=8, capsize=3, capthick=1.5)

# ============================================================
# 6. 图表美化
# ============================================================

ax.set_xlabel(r'$\varepsilon_{\mathrm{MLDP}}$  (Privacy Budget, actual DP budget = $\varepsilon_{\mathrm{MLDP}} \times d_{\max}$,  $d_{\max}=2.89$)', fontsize=13)
ax.set_ylabel("Accuracy", fontsize=14)
ax.set_title("Utility Comparison: Baseline vs Sample Amplification (QNLI, BERT)", fontsize=15)

# X 轴刻度：显示所有出现过的 epsilon 值
ax.set_xticks(all_epsilons)
ax.set_xticklabels([str(int(e)) if e == int(e) else str(e) for e in all_epsilons], fontsize=9, rotation=45)

# Y 轴范围
ax.set_ylim(0.40, 1.0)
ax.tick_params(axis='y', labelsize=11)

# 网格线
ax.grid(True, alpha=0.3, linestyle='--')

# 图例
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)

plt.tight_layout()

# ============================================================
# 7. 保存图片
# ============================================================
output_png = os.path.join(PROJECT_DIR, "utility_comparison_QNLI_bert.png")
output_pdf = os.path.join(PROJECT_DIR, "utility_comparison_QNLI_bert.pdf")

plt.savefig(output_png, dpi=200, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\n图片已保存:")
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")

plt.show()
