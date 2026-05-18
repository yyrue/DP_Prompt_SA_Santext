#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Privacy-Utility Tradeoff 图（双 Y 轴版本）
- X 轴:    epsilon (2, 4, 6, ..., 24) + No Sanitization
- 左 Y 轴: Defense Rate（隐私保护，越高越好）—— 包含 Mask Attack 和 KNN Attack 两条曲线
- 右 Y 轴: Accuracy / Utility（效用，越高越好）

数据来源:
  - utility_results_SST2_bert.csv                         → SanText 各 epsilon 的 accuracy
  - mask_attack_results_SST2_v2.csv                       → SanText 各 epsilon 的 defense_rate (Mask Attack)
  - knn_attack_results_SST2_v2.csv                        → SanText 各 epsilon 的 defense_rate (KNN Attack)
  - utility_results_SST2_bert_no_sanitization.csv         → 无脱敏 accuracy
  - output_SST2_no_sanitization/mask_attack_results_no_sanitization.csv → 无脱敏 defense_rate
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

# ============================================================
# 1. 读取数据工具函数
# ============================================================

def load_avg(csv_path, key_col, val_col):
    """按 key_col 分组，计算 val_col 的均值和标准差"""
    groups = defaultdict(list)
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


def load_no_san_avg(csv_path, val_col):
    """读取无 epsilon 列的 CSV，直接计算 val_col 均值和标准差"""
    values = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row[val_col]))
            except (ValueError, KeyError):
                continue
    if not values:
        return None, None
    mean = np.mean(values)
    std  = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std

# ============================================================
# 2. 加载数据
# ============================================================

# SanText: accuracy (utility)
acc_mean, acc_std = load_avg(
    "utility_results_SST2_bert.csv",
    key_col="epsilon", val_col="accuracy"
)

# SanText: defense_rate (Mask Attack)
mask_def_mean, mask_def_std = load_avg(
    "mask_attack_results_SST2_v2.csv",
    key_col="epsilon", val_col="defense_rate"
)

# SanText: defense_rate (KNN Attack)
knn_def_mean, knn_def_std = load_avg(
    "knn_attack_results_SST2_v2.csv",
    key_col="epsilon", val_col="defense_rate"
)

# No Sanitization: accuracy
no_san_acc_mean, no_san_acc_std = load_no_san_avg(
    "utility_results_SST2_bert_no_sanitization.csv",
    val_col="accuracy"
)

# No Sanitization: defense_rate (Mask Attack)
no_san_mask_def_mean, no_san_mask_def_std = load_no_san_avg(
    "output_SST2_no_sanitization/mask_attack_results_no_sanitization.csv",
    val_col="defense_rate"
)

# No Sanitization: defense_rate (KNN Attack) —— 理论上接近 0，因为无脱敏时 KNN 能轻易还原
# 如果有数据文件则读取，否则设为默认值
try:
    no_san_knn_def_mean, no_san_knn_def_std = load_no_san_avg(
        "output_SST2_no_sanitization/knn_attack_results_no_sanitization.csv",
        val_col="defense_rate"
    )
except:
    no_san_knn_def_mean = 0.0  # 无脱敏时 KNN Attack 几乎必中
    no_san_knn_def_std = 0.0

# ============================================================
# 3. 整理数据（按 epsilon 排序）
# ============================================================

epsilons = sorted(set(acc_mean.keys()) & set(mask_def_mean.keys()) & set(knn_def_mean.keys()))

san_acc        = np.array([acc_mean[e] for e in epsilons])
san_acc_err    = np.array([acc_std[e]  for e in epsilons])
san_mask_def   = np.array([mask_def_mean[e] for e in epsilons])
san_mask_def_err = np.array([mask_def_std[e]  for e in epsilons])
san_knn_def    = np.array([knn_def_mean[e] for e in epsilons])
san_knn_def_err = np.array([knn_def_std[e]  for e in epsilons])

# X 轴：epsilon 数值 + 末尾加一个 "No San" 虚拟点
x_eps    = list(range(len(epsilons)))          # 0..11 对应 ε=2..24
x_nosan  = len(epsilons)                        # 12 对应 No Sanitization
x_all    = x_eps + [x_nosan]
x_labels = [str(int(e)) for e in epsilons] + ['No\nSan']

print("=" * 75)
print(f"{'epsilon':>8}  {'Accuracy':>10}  {'Mask Def':>12}  {'KNN Def':>12}")
print("-" * 75)
for i, e in enumerate(epsilons):
    print(f"{e:>8.1f}  {san_acc[i]:>10.4f}  {san_mask_def[i]:>12.4f}  {san_knn_def[i]:>12.4f}")
print("-" * 75)
print(f"{'No-San':>8}  {no_san_acc_mean:>10.4f}  {no_san_mask_def_mean:>12.4f}  {no_san_knn_def_mean:>12.4f}")
print("=" * 75)

# ============================================================
# 4. 绘图（双 Y 轴）
# ============================================================

COLOR_MASK_DEF = '#2878B5'   # 蓝色   → Mask Attack Defense Rate（左轴）
COLOR_KNN_DEF  = '#9AC9DB'   # 浅蓝色 → KNN Attack Defense Rate（左轴）
COLOR_ACC      = '#F28522'   # 橙色   → Accuracy（右轴）
COLOR_NS       = '#C82423'   # 红色   → No Sanitization

fig, ax_left = plt.subplots(figsize=(12, 7))
ax_right = ax_left.twinx()   # 共享 X 轴的右侧 Y 轴

# ---- 左轴：Mask Attack Defense Rate（SanText 曲线 + No Sanitization 连线）----
x_mask_def_full    = x_eps + [x_nosan]
mask_def_full      = list(san_mask_def)  + [no_san_mask_def_mean]
mask_def_err_full  = list(san_mask_def_err) + [no_san_mask_def_std]

line_mask_def, = ax_left.plot(
    x_mask_def_full, mask_def_full,
    color=COLOR_MASK_DEF, linewidth=2.2, marker='o', markersize=7,
    label='Defense Rate (Mask Attack)', zorder=3
)
ax_left.errorbar(
    x_mask_def_full, mask_def_full, yerr=mask_def_err_full,
    fmt='none', color=COLOR_MASK_DEF, capsize=4, elinewidth=1.2, zorder=3
)

# ---- 左轴：KNN Attack Defense Rate（SanText 曲线 + No Sanitization 连线）----
x_knn_def_full    = x_eps + [x_nosan]
knn_def_full      = list(san_knn_def)  + [no_san_knn_def_mean]
knn_def_err_full  = list(san_knn_def_err) + [no_san_knn_def_std]

line_knn_def, = ax_left.plot(
    x_knn_def_full, knn_def_full,
    color=COLOR_KNN_DEF, linewidth=2.2, marker='^', markersize=7,
    linestyle='-.', label='Defense Rate (KNN Attack)', zorder=3
)
ax_left.errorbar(
    x_knn_def_full, knn_def_full, yerr=knn_def_err_full,
    fmt='none', color=COLOR_KNN_DEF, capsize=4, elinewidth=1.2, zorder=3
)

# No Sanitization 点单独用菱形标记（左轴）
ax_left.plot(
    x_nosan, no_san_mask_def_mean,
    marker='D', color=COLOR_NS, markersize=10, zorder=5
)

# ---- 右轴：Accuracy（SanText 曲线 + No Sanitization 连线）----
x_acc_full    = x_eps + [x_nosan]
acc_full      = list(san_acc)  + [no_san_acc_mean]
acc_err_full  = list(san_acc_err) + [no_san_acc_std]

line_acc, = ax_right.plot(
    x_acc_full, acc_full,
    color=COLOR_ACC, linewidth=2.2, marker='s', markersize=7,
    linestyle='--', label='Accuracy (Utility)', zorder=3
)
ax_right.errorbar(
    x_acc_full, acc_full, yerr=acc_err_full,
    fmt='none', color=COLOR_ACC, capsize=4, elinewidth=1.2, zorder=3
)
# No Sanitization 点单独用菱形标记（右轴）
ax_right.plot(
    x_nosan, no_san_acc_mean,
    marker='D', color=COLOR_NS, markersize=10, zorder=5
)

# ---- X 轴刻度与标签 ----
ax_left.set_xticks(x_all)
ax_left.set_xticklabels(x_labels, fontsize=10.5)
ax_left.set_xlim(-0.5, x_nosan + 0.5)
ax_left.set_xlabel('Privacy Budget ε  (No San = No Sanitization)', fontsize=12)

# ---- 左 Y 轴设置 ----
ax_left.set_ylabel('Defense Rate ↑', fontsize=12, color=COLOR_MASK_DEF)
ax_left.tick_params(axis='y', labelcolor=COLOR_MASK_DEF)
ax_left.set_ylim(-0.02, 1.05)
ax_left.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# ---- 右 Y 轴设置 ----
ax_right.set_ylabel('Accuracy (Utility) ↑', fontsize=12, color=COLOR_ACC)
ax_right.tick_params(axis='y', labelcolor=COLOR_ACC)
ax_right.set_ylim(0.30, 1.05)
ax_right.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# ---- 标题 ----
ax_left.set_title(
    'Privacy-Utility Tradeoff on SST-2  (SanText, BERT Embedding)\n'
    'Comparison: Mask Attack vs KNN Attack',
    fontsize=13, pad=12
)

# ---- 网格（仅左轴） ----
ax_left.grid(True, linestyle='--', alpha=0.35, zorder=0)

# ---- 图例（合并两轴） ----
lines  = [line_mask_def, line_knn_def, line_acc]
labels = [l.get_label() for l in lines]
# 手动加 No Sanitization 图例项
from matplotlib.lines import Line2D
no_san_patch = Line2D([0], [0], marker='D', color=COLOR_NS, linestyle='None',
                      markersize=9, label='No Sanitization')
lines.append(no_san_patch)
labels.append('No Sanitization')
ax_left.legend(lines, labels, loc='center right', fontsize=10.5, framealpha=0.9)

plt.tight_layout()

output_path = 'privacy_utility_tradeoff_SST2_combined.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\n图片已保存至: {output_path}")
plt.show()