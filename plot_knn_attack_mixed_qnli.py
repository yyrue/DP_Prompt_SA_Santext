#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 KNN Attack Defense Rate 曲线（QNLI 数据集）
横坐标：ε (MLDP)，标注实际隐私预算为 ε·d_max
纵坐标：Defense Rate
六条曲线：SanText baseline + eps_high = 16, 18, 20, 22, 24
"""

import csv
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 13

# ---- 配置 ----
PROJECT_DIR = "/data/youyaru/youyaru/SanText-main"
D_MAX = 2.892667
TOPK = 10
EPS_HIGH_LIST = [16, 18, 20, 22, 24]

# 颜色和标记
STYLES = {
    16: {'color': '#e74c3c', 'marker': 'o',  'label': r'$\varepsilon_0 = 16$'},
    18: {'color': '#3498db', 'marker': 's',  'label': r'$\varepsilon_0 = 18$'},
    20: {'color': '#2ecc71', 'marker': '^',  'label': r'$\varepsilon_0 = 20$'},
    22: {'color': '#8c564b', 'marker': 'p',  'label': r'$\varepsilon_0 = 22$'},
    24: {'color': '#9b59b6', 'marker': 'D',  'label': r'$\varepsilon_0 = 24$'},
}

# ---- 读取 Mixed 数据 ----
all_data = {}  # {eps_high: {epsilon': [defense_rate, ...]}}

for eps_high in EPS_HIGH_LIST:
    csv_path = f"{PROJECT_DIR}/knn_attack_results_QNLI_mixed_eps{eps_high}_Top{TOPK}.csv"
    data = {}
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                eps = float(row['epsilon'])
                dr = float(row['defense_rate'])
                data.setdefault(eps, []).append(dr)
    except FileNotFoundError:
        print(f"⚠️  文件不存在: {csv_path}")
        continue
    all_data[eps_high] = data

# ---- 读取 SanText baseline 数据 ----
santext_csv = f"{PROJECT_DIR}/knn_attack_results_QNLI_Top{TOPK}.csv"
santext_data = {}
try:
    with open(santext_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps = float(row['epsilon'])
            dr = float(row['defense_rate'])
            santext_data.setdefault(eps, []).append(dr)
except FileNotFoundError:
    print(f"⚠️  SanText baseline 文件不存在: {santext_csv}")

# ---- 画图 ----
fig, ax = plt.subplots(figsize=(10, 6))

# 先画 SanText baseline（黑色虚线）
if santext_data:
    epsilons_st = sorted(santext_data.keys())
    means_st = [statistics.mean(santext_data[e]) for e in epsilons_st]
    stds_st = [statistics.stdev(santext_data[e]) if len(santext_data[e]) > 1 else 0 for e in epsilons_st]
    ax.errorbar(
        epsilons_st, means_st, yerr=stds_st,
        color='black',
        marker='x',
        markersize=8,
        linewidth=2,
        linestyle='--',
        capsize=3,
        capthick=1.5,
        label='baseline',
    )

# 再画 mixed 曲线
for eps_high in EPS_HIGH_LIST:
    if eps_high not in all_data:
        continue
    data = all_data[eps_high]

    epsilons = sorted(data.keys())
    means = [statistics.mean(data[e]) for e in epsilons]
    stds = [statistics.stdev(data[e]) if len(data[e]) > 1 else 0 for e in epsilons]

    style = STYLES[eps_high]
    ax.errorbar(
        epsilons, means, yerr=stds,
        color=style['color'],
        marker=style['marker'],
        markersize=7,
        linewidth=2,
        capsize=3,
        capthick=1.5,
        label=style['label'],
    )

# ---- 坐标轴与标注 ----
ax.set_xlabel(r'$\varepsilon_{\mathrm{MLDP}}$  (Privacy Budget, actual DP budget = $\varepsilon_{\mathrm{MLDP}} \times d_{\max}$'
              f',  $d_{{\\max}}={D_MAX:.2f}$)', fontsize=13)
ax.set_ylabel('Defense Rate', fontsize=14)
ax.set_title(f'KNN Attack (Top-{TOPK}) Defense Rate — QNLI, Sample Amplification', fontsize=15)

ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, 26)

# 网格
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best', framealpha=0.9)

plt.tight_layout()

# ---- 保存 ----
out_png = f"{PROJECT_DIR}/knn_attack_defense_rate_QNLI_mixed_Top{TOPK}.png"
out_pdf = f"{PROJECT_DIR}/knn_attack_defense_rate_QNLI_mixed_Top{TOPK}.pdf"
fig.savefig(out_png, dpi=200, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f"已保存: {out_png}")
print(f"已保存: {out_pdf}")

# ---- 打印数据表 ----
print(f"\n{'source':>12} {'epsilon':>10} {'Defense Rate':>20}")
print("-" * 48)

# SanText baseline
if santext_data:
    for eps in sorted(santext_data.keys()):
        avg = statistics.mean(santext_data[eps])
        std = statistics.stdev(santext_data[eps]) if len(santext_data[eps]) > 1 else 0
        print(f"{'SanText':>12} {eps:>10.2f} {avg:>12.4f} ± {std:.4f}")
    print()

# Mixed
for eps_high in EPS_HIGH_LIST:
    if eps_high not in all_data:
        continue
    data = all_data[eps_high]
    for eps in sorted(data.keys()):
        avg = statistics.mean(data[eps])
        std = statistics.stdev(data[eps]) if len(data[eps]) > 1 else 0
        print(f"{f'mixed_{eps_high}':>12} {eps:>10.2f} {avg:>12.4f} ± {std:.4f}")
    print()

plt.show()
