#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Privacy-Utility Tradeoff 图（双 Y 轴版本）
- X 轴:    epsilon (2, 4, 6, ..., 24) + No Sanitization
- 左 Y 轴: Defense Rate（隐私保护，越高越好）
- 右 Y 轴: Accuracy / Utility（效用，越高越好）
数据来源:
  - utility_results_SST2_bert.csv          → SanText 各 epsilon 的 accuracy
  - mask_attack_results_SST2_v2.csv        → SanText 各 epsilon 的 defense_rate (Mask Attack)
  - knn_attack_results_SST2_v2.csv         → SanText 各 epsilon 的 defense_rate (KNN Attack)
  - utility_results_SST2_bert_no_sanitization.csv  → 无脱敏 accuracy
  - output_SST2_no_sanitization/mask_attack_results_no_sanitization.csv → 无脱敏 defense_rate
  - utility_results_SST2_bert_mixed.csv    → Mixed 各 epsilon 的 accuracy
  - mask_attack_results_SST2_mixed.csv     → Mixed 各 epsilon 的 defense_rate (Mask Attack)
"""

import csv
import os
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


def load_no_san_avg(csv_path, val_col):
    """读取无 epsilon 列的 CSV，直接计算 val_col 均值和标准差"""
    values = []
    if not os.path.exists(csv_path):
        print(f"[WARN] 文件不存在，跳过: {csv_path}")
        return None, None
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

# # SanText: defense_rate (KNN Attack Top-1)
# knn_top1_def_mean, knn_top1_def_std = load_avg(
#     "knn_attack_results_SST2_v2_Top1.csv",
#     key_col="epsilon", val_col="defense_rate"
# )
#
# # SanText: defense_rate (KNN Attack Top-5)
# knn_top5_def_mean, knn_top5_def_std = load_avg(
#     "knn_attack_results_SST2_v2_Top5.csv",
#     key_col="epsilon", val_col="defense_rate"
# )
#
# # SanText: defense_rate (KNN Attack Top-10)
# knn_top10_def_mean, knn_top10_def_std = load_avg(
#     "knn_attack_results_SST2_v2.csv",
#     key_col="epsilon", val_col="defense_rate"
# )
#
# # SanText: defense_rate (KNN Attack Top-300)
# knn_top300_def_mean, knn_top300_def_std = load_avg(
#     "knn_attack_results_SST2_v2_Top300.csv",
#     key_col="epsilon", val_col="defense_rate"
# )
#
# # SanText: defense_rate (KNN Attack Top-500)
# knn_top500_def_mean, knn_top500_def_std = load_avg(
#     "knn_attack_results_SST2_v2_Top500.csv",
#     key_col="epsilon", val_col="defense_rate"
# )

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

# # No Sanitization: defense_rate (KNN Attack) —— 无脱敏时 KNN 攻击几乎必中，defense rate 接近 0
# no_san_knn_top1_def_mean = 0.0
# no_san_knn_top1_def_std = 0.0
# no_san_knn_top5_def_mean = 0.0
# no_san_knn_top5_def_std = 0.0
# no_san_knn_top10_def_mean = 0.0
# no_san_knn_top10_def_std = 0.0
# no_san_knn_top300_def_mean = 0.0
# no_san_knn_top300_def_std = 0.0
# no_san_knn_top500_def_mean = 0.0
# no_san_knn_top500_def_std = 0.0

# ---- Mixed 数据 ----
# Mixed: accuracy (utility)
mixed_acc_mean, mixed_acc_std = load_avg(
    "utility_results_SST2_bert_mixed.csv",
    key_col="epsilon", val_col="accuracy"
)

# Mixed: defense_rate (Mask Attack)
mixed_mask_def_mean, mixed_mask_def_std = load_avg(
    "mask_attack_results_SST2_mixed.csv",
    key_col="epsilon", val_col="defense_rate"
)

# ============================================================
# 3. 整理数据（按 epsilon 排序）
# ============================================================

epsilons = sorted(set(acc_mean.keys()) & set(mask_def_mean.keys()))

san_acc          = np.array([acc_mean[e] for e in epsilons])
san_acc_err      = np.array([acc_std[e]  for e in epsilons])
san_mask_def     = np.array([mask_def_mean[e] for e in epsilons])
san_mask_def_err = np.array([mask_def_std[e]  for e in epsilons])
# san_knn_top1_def     = np.array([knn_top1_def_mean[e] for e in epsilons])
# san_knn_top1_def_err = np.array([knn_top1_def_std[e]  for e in epsilons])
# san_knn_top5_def     = np.array([knn_top5_def_mean[e] for e in epsilons])
# san_knn_top5_def_err = np.array([knn_top5_def_std[e]  for e in epsilons])
# san_knn_top10_def     = np.array([knn_top10_def_mean[e] for e in epsilons])
# san_knn_top10_def_err = np.array([knn_top10_def_std[e]  for e in epsilons])
# san_knn_top300_def     = np.array([knn_top300_def_mean[e] for e in epsilons])
# san_knn_top300_def_err = np.array([knn_top300_def_std[e]  for e in epsilons])
# san_knn_top500_def     = np.array([knn_top500_def_mean[e] for e in epsilons])
# san_knn_top500_def_err = np.array([knn_top500_def_std[e]  for e in epsilons])

# Mixed 数据整理（epsilon 范围可能不同，取交集）
mixed_epsilons = sorted(set(mixed_acc_mean.keys()) & set(mixed_mask_def_mean.keys()))
mixed_acc_arr      = np.array([mixed_acc_mean[e] for e in mixed_epsilons])
mixed_acc_err_arr  = np.array([mixed_acc_std[e]  for e in mixed_epsilons])
mixed_mask_def_arr     = np.array([mixed_mask_def_mean[e] for e in mixed_epsilons])
mixed_mask_def_err_arr = np.array([mixed_mask_def_std[e]  for e in mixed_epsilons])

# X 轴：epsilon 数值 + 末尾加一个 "No San" 虚拟点
# 用数字索引作为 X 坐标，最后一个点为 No Sanitization
x_eps    = list(range(len(epsilons)))          # 0..11 对应 ε=2..24
x_nosan  = len(epsilons)                        # 12 对应 No Sanitization
x_all    = x_eps + [x_nosan]
x_labels = [str(int(e)) for e in epsilons] + ['No\nSan']

# Mixed 的 X 坐标：需要映射到与 SanText 相同的 X 轴位置
eps_to_x = {e: i for i, e in enumerate(epsilons)}
mixed_x = [eps_to_x[e] for e in mixed_epsilons if e in eps_to_x]
# 过滤掉不在 SanText epsilon 列表中的 mixed epsilon
mixed_epsilons_filtered = [e for e in mixed_epsilons if e in eps_to_x]
mixed_acc_filtered      = np.array([mixed_acc_mean[e] for e in mixed_epsilons_filtered])
mixed_acc_err_filtered  = np.array([mixed_acc_std[e]  for e in mixed_epsilons_filtered])
mixed_def_filtered      = np.array([mixed_mask_def_mean[e] for e in mixed_epsilons_filtered])
mixed_def_err_filtered  = np.array([mixed_mask_def_std[e]  for e in mixed_epsilons_filtered])
mixed_x_filtered        = [eps_to_x[e] for e in mixed_epsilons_filtered]

print("=" * 70)
print(f"{'epsilon':>8}  {'SanText Acc':>12}  {'SanText Mask':>12}  {'Mixed Acc':>12}  {'Mixed Mask':>12}")
print("-" * 70)
for i, e in enumerate(epsilons):
    mixed_acc_val = mixed_acc_mean.get(e, float('nan'))
    mixed_def_val = mixed_mask_def_mean.get(e, float('nan'))
    print(f"{e:>8.1f}  {san_acc[i]:>12.4f}  {san_mask_def[i]:>12.4f}  {mixed_acc_val:>12.4f}  {mixed_def_val:>12.4f}")
print("-" * 70)
if no_san_acc_mean is not None:
    print(f"{'No-San':>8}  {no_san_acc_mean:>12.4f}  {no_san_mask_def_mean:>12.4f}  {'---':>12}  {'---':>12}")
print("=" * 70)

# ============================================================
# 4. 绘图（双 Y 轴）
# ============================================================

COLOR_SANTEXT     = '#2878B5'   # 蓝色    → SanText（Accuracy + Defense 同色）
COLOR_MIXED       = '#F28522'   # 橙色    → Mixed（Accuracy + Defense 同色）
COLOR_NS          = '#C82423'   # 红色    → No Sanitization

fig, ax_left = plt.subplots(figsize=(12, 7))
ax_right = ax_left.twinx()   # 共享 X 轴的右侧 Y 轴

# ---- 左轴：SanText Defense Rate（虚线 + 方块）----
x_mask_def_full    = x_eps + [x_nosan]
mask_def_full      = list(san_mask_def)  + [no_san_mask_def_mean]
mask_def_err_full  = list(san_mask_def_err) + [no_san_mask_def_std]

line_san_def, = ax_left.plot(
    x_mask_def_full, mask_def_full,
    color=COLOR_SANTEXT, linewidth=2.2, marker='s', markersize=7,
    linestyle='--', label='SanText Defense Rate', zorder=3
)
ax_left.errorbar(
    x_mask_def_full, mask_def_full, yerr=mask_def_err_full,
    fmt='none', color=COLOR_SANTEXT, capsize=4, elinewidth=1.2, zorder=3
)

# ---- 左轴：Mixed Defense Rate（虚线 + 菱形）----
line_mixed_def, = ax_left.plot(
    mixed_x_filtered, mixed_def_filtered,
    color=COLOR_MIXED, linewidth=2.2, marker='D', markersize=7,
    linestyle='--', label='Mixed Defense Rate', zorder=4
)
ax_left.errorbar(
    mixed_x_filtered, mixed_def_filtered, yerr=mixed_def_err_filtered,
    fmt='none', color=COLOR_MIXED, capsize=4, elinewidth=1.2, zorder=4
)

# # ---- 左轴：KNN Attack Top-1 Defense Rate ----
# x_knn_top1_def_full    = x_eps + [x_nosan]
# knn_top1_def_full      = list(san_knn_top1_def)  + [no_san_knn_top1_def_mean]
# knn_top1_def_err_full  = list(san_knn_top1_def_err) + [no_san_knn_top1_def_std]
#
# line_knn_top1_def, = ax_left.plot(
#     x_knn_top1_def_full, knn_top1_def_full,
#     color=COLOR_KNN_TOP1, linewidth=2.2, marker='v', markersize=6,
#     linestyle=':', label='Defense (KNN Top-1)', zorder=3
# )
# ax_left.errorbar(
#     x_knn_top1_def_full, knn_top1_def_full, yerr=knn_top1_def_err_full,
#     fmt='none', color=COLOR_KNN_TOP1, capsize=3, elinewidth=1.0, zorder=3
# )
#
# # ---- 左轴：KNN Attack Top-5 Defense Rate ----
# x_knn_top5_def_full    = x_eps + [x_nosan]
# knn_top5_def_full      = list(san_knn_top5_def)  + [no_san_knn_top5_def_mean]
# knn_top5_def_err_full  = list(san_knn_top5_def_err) + [no_san_knn_top5_def_std]
#
# line_knn_top5_def, = ax_left.plot(
#     x_knn_top5_def_full, knn_top5_def_full,
#     color=COLOR_KNN_TOP5, linewidth=2.2, marker='<', markersize=6,
#     linestyle='-.', label='Defense (KNN Top-5)', zorder=3
# )
# ax_left.errorbar(
#     x_knn_top5_def_full, knn_top5_def_full, yerr=knn_top5_def_err_full,
#     fmt='none', color=COLOR_KNN_TOP5, capsize=3, elinewidth=1.0, zorder=3
# )
#
# # ---- 左轴：KNN Attack Top-10 Defense Rate ----
# x_knn_top10_def_full    = x_eps + [x_nosan]
# knn_top10_def_full      = list(san_knn_top10_def)  + [no_san_knn_top10_def_mean]
# knn_top10_def_err_full  = list(san_knn_top10_def_err) + [no_san_knn_top10_def_std]
#
# line_knn_top10_def, = ax_left.plot(
#     x_knn_top10_def_full, knn_top10_def_full,
#     color=COLOR_KNN_TOP10, linewidth=2.2, marker='^', markersize=6,
#     linestyle='--', label='Defense (KNN Top-10)', zorder=3
# )
# ax_left.errorbar(
#     x_knn_top10_def_full, knn_top10_def_full, yerr=knn_top10_def_err_full,
#     fmt='none', color=COLOR_KNN_TOP10, capsize=3, elinewidth=1.0, zorder=3
# )
#
# # ---- 左轴：KNN Attack Top-300 Defense Rate ----
# x_knn_top300_def_full    = x_eps + [x_nosan]
# knn_top300_def_full      = list(san_knn_top300_def)  + [no_san_knn_top300_def_mean]
# knn_top300_def_err_full  = list(san_knn_top300_def_err) + [no_san_knn_top300_def_std]
#
# line_knn_top300_def, = ax_left.plot(
#     x_knn_top300_def_full, knn_top300_def_full,
#     color=COLOR_KNN_TOP300, linewidth=2.2, marker='p', markersize=7,
#     linestyle=':', label='Defense (KNN Top-300)', zorder=3
# )
# ax_left.errorbar(
#     x_knn_top300_def_full, knn_top300_def_full, yerr=knn_top300_def_err_full,
#     fmt='none', color=COLOR_KNN_TOP300, capsize=3, elinewidth=1.0, zorder=3
# )
#
# # ---- 左轴：KNN Attack Top-500 Defense Rate ----
# x_knn_top500_def_full    = x_eps + [x_nosan]
# knn_top500_def_full      = list(san_knn_top500_def)  + [no_san_knn_top500_def_mean]
# knn_top500_def_err_full  = list(san_knn_top500_def_err) + [no_san_knn_top500_def_std]
#
# line_knn_top500_def, = ax_left.plot(
#     x_knn_top500_def_full, knn_top500_def_full,
#     color=COLOR_KNN_TOP500, linewidth=2.2, marker='h', markersize=7,
#     linestyle='-.', label='Defense (KNN Top-500)', zorder=3
# )
# ax_left.errorbar(
#     x_knn_top500_def_full, knn_top500_def_full, yerr=knn_top500_def_err_full,
#     fmt='none', color=COLOR_KNN_TOP500, capsize=3, elinewidth=1.0, zorder=3
# )

# No Sanitization 点单独用菱形标记（左轴）
ax_left.plot(
    x_nosan, no_san_mask_def_mean,
    marker='D', color=COLOR_NS, markersize=10, zorder=5
)

# ---- 右轴：SanText Accuracy（实线 + 圆点）----
x_acc_full  = x_eps + [x_nosan]
acc_full     = list(san_acc)  + [no_san_acc_mean]
acc_err_full = list(san_acc_err) + [no_san_acc_std]

line_san_acc, = ax_right.plot(
    x_acc_full, acc_full,
    color=COLOR_SANTEXT, linewidth=2.2, marker='o', markersize=7,
    linestyle='-', label='SanText Accuracy', zorder=3
)
ax_right.errorbar(
    x_acc_full, acc_full, yerr=acc_err_full,
    fmt='none', color=COLOR_SANTEXT, capsize=4, elinewidth=1.2, zorder=3
)

# ---- 右轴：Mixed Accuracy（实线 + 星号）----
line_mixed_acc, = ax_right.plot(
    mixed_x_filtered, mixed_acc_filtered,
    color=COLOR_MIXED, linewidth=2.2, marker='*', markersize=10,
    linestyle='-', label='Mixed Accuracy', zorder=4
)
ax_right.errorbar(
    mixed_x_filtered, mixed_acc_filtered, yerr=mixed_acc_err_filtered,
    fmt='none', color=COLOR_MIXED, capsize=4, elinewidth=1.2, zorder=4
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
ax_left.set_ylabel('Defense Rate ↑  (dashed lines)', fontsize=12)
ax_left.tick_params(axis='y')
ax_left.set_ylim(-0.02, 1.05)
ax_left.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# ---- 右 Y 轴设置 ----
ax_right.set_ylabel('Accuracy (Utility) ↑  (solid lines)', fontsize=12)
ax_right.tick_params(axis='y')
ax_right.set_ylim(0.30, 1.05)
ax_right.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# ---- 标题 ----
ax_left.set_title(
    'Privacy-Utility Tradeoff on SST-2  (SanText vs Mixed, BERT Embedding,sample between 0 and 24)',
    fontsize=13, pad=12
)

# ---- 网格（仅左轴） ----
ax_left.grid(True, linestyle='--', alpha=0.35, zorder=0)

# ---- 图例（合并两轴） ----
# ---- 图例（合并两轴） ----
lines  = [line_san_acc, line_san_def, line_mixed_acc, line_mixed_def]
labels = [l.get_label() for l in lines]
# 手动加 No Sanitization 图例项
from matplotlib.lines import Line2D
no_san_patch = Line2D([0], [0], marker='D', color=COLOR_NS, linestyle='None',
                      markersize=9, label='No Sanitization')
lines.append(no_san_patch)
labels.append('No Sanitization')
ax_left.legend(lines, labels, loc='center right', fontsize=10.5, framealpha=0.9)

plt.tight_layout()

output_path = 'privacy_utility_tradeoff_SST2.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\n图片已保存至: {output_path}")
plt.show()
