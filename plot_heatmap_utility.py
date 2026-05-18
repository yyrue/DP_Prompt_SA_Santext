"""
热力图绘制脚本：可视化固定 eps_target 下不同 (eps_low, eps_high) 组合的效用

X 轴: eps_high
Y 轴: eps_low
颜色: 平均 Accuracy（效用）

用法:
    python3 plot_heatmap_utility.py --eps_target 10
    python3 plot_heatmap_utility.py --eps_target 10 --csv utility_results_SST2_bert_heatmap_target10.csv
"""

import os
import csv
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

# -------------------------------------------------------
# 常量
# -------------------------------------------------------
D_MAX = 2.892667


def mldp_to_pure(eps_mldp: float) -> float:
    return eps_mldp * D_MAX


def calc_p_general(eps_low_mldp: float, eps_high_mldp: float, eps_target_mldp: float) -> float:
    """通用混合采样概率公式（论文 Lemma 3.3）"""
    eps1 = mldp_to_pure(eps_low_mldp)
    eps2 = mldp_to_pure(eps_high_mldp)
    eps_prime = mldp_to_pure(eps_target_mldp)

    e1 = math.exp(eps1)
    ep = math.exp(eps_prime)

    numerator = ep - e1
    denominator = (math.exp((eps1 + eps2) / 2) - e1) + (1 - math.exp((eps1 - eps2) / 2)) * ep

    if denominator == 0:
        return 0.0

    p = numerator / denominator
    return max(0.0, min(1.0, p))


def main():
    parser = argparse.ArgumentParser(description="热力图：不同 (eps_low, eps_high) 组合的效用")
    parser.add_argument("--eps_target", type=float, default=10.0,
                        help="目标 MLDP epsilon")
    parser.add_argument("--csv", type=str, default=None,
                        help="效用结果 CSV 文件路径（默认自动推断）")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="输出图片文件名前缀（默认自动生成）")
    parser.add_argument("--show_p", action="store_true", default=True,
                        help="在热力图格子中标注采样概率 p")
    args = parser.parse_args()

    # CSV 文件路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if args.csv is None:
        args.csv = os.path.join(
            project_dir,
            f"utility_results_SST2_bert_heatmap_target{int(args.eps_target)}.csv"
        )

    if args.output_prefix is None:
        args.output_prefix = f"heatmap_utility_target{int(args.eps_target)}"

    # -------------------------------------------------------
    # 读取数据
    # -------------------------------------------------------
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV 文件不存在: {args.csv}")
        print("请先运行效用评估实验: bash run_utility_bert_heatmap.sh")
        return

    results = {}  # {(eps_low, eps_high): [accuracy, ...]}
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps_low = float(row['eps_low'])
            eps_high = float(row['eps_high'])
            acc = row['accuracy']
            if acc == 'N/A':
                continue
            try:
                results.setdefault((eps_low, eps_high), []).append(float(acc))
            except ValueError:
                continue

    if not results:
        print("[ERROR] 没有有效的实验结果")
        return

    # 提取所有 eps_low 和 eps_high 值
    eps_low_values = sorted(set(k[0] for k in results.keys()))
    eps_high_values = sorted(set(k[1] for k in results.keys()))

    print(f"eps_target = {args.eps_target}")
    print(f"eps_low 值: {eps_low_values}")
    print(f"eps_high 值: {eps_high_values}")
    print(f"有效组合数: {len(results)}")

    # -------------------------------------------------------
    # 构建热力图矩阵
    # -------------------------------------------------------
    n_low = len(eps_low_values)
    n_high = len(eps_high_values)

    # 平均 accuracy 矩阵
    acc_matrix = np.full((n_low, n_high), np.nan)
    # 标准差矩阵
    std_matrix = np.full((n_low, n_high), np.nan)
    # 采样概率 p 矩阵
    p_matrix = np.full((n_low, n_high), np.nan)

    for i, el in enumerate(eps_low_values):
        for j, eh in enumerate(eps_high_values):
            if (el, eh) in results:
                accs = results[(el, eh)]
                acc_matrix[i, j] = np.mean(accs)
                std_matrix[i, j] = np.std(accs) if len(accs) > 1 else 0.0
            # 计算理论 p 值
            p_matrix[i, j] = calc_p_general(el, eh, args.eps_target)

    # -------------------------------------------------------
    # 绘图设置
    # -------------------------------------------------------
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.unicode_minus'] = False

    # ==========================================
    # 图1：Accuracy 热力图
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    # 使用 imshow 绘制热力图
    im = ax1.imshow(acc_matrix, cmap='RdYlGn', aspect='auto',
                    origin='lower', interpolation='nearest')

    # 设置坐标轴
    ax1.set_xticks(range(n_high))
    ax1.set_xticklabels([f"{v:.0f}" for v in eps_high_values], fontsize=11)
    ax1.set_yticks(range(n_low))
    ax1.set_yticklabels([f"{v:.0f}" for v in eps_low_values], fontsize=11)

    ax1.set_xlabel(r"$\varepsilon_{high}$", fontsize=14)
    ax1.set_ylabel(r"$\varepsilon_{low}$", fontsize=14)
    ax1.set_title(
        r"Utility (Accuracy) for Different $(\varepsilon_{low}, \varepsilon_{high})$ "
        f"Combinations\n"
        r"Fixed $\varepsilon_{target}$" + f" = {args.eps_target}",
        fontsize=14, fontweight='bold'
    )

    # 颜色条
    cbar = plt.colorbar(im, ax=ax1, shrink=0.85)
    cbar.set_label("Average Accuracy", fontsize=12)

    # 在每个格子中标注数值
    for i in range(n_low):
        for j in range(n_high):
            if not np.isnan(acc_matrix[i, j]):
                # 根据背景色选择文字颜色
                val = acc_matrix[i, j]
                text_color = 'white' if val < 0.6 else 'black'

                # 标注 accuracy
                ax1.text(j, i, f"{val:.3f}",
                         ha='center', va='center', fontsize=9,
                         color=text_color, fontweight='bold')

                # 标注 p 值（小字）
                if args.show_p and not np.isnan(p_matrix[i, j]):
                    ax1.text(j, i - 0.3, f"p={p_matrix[i, j]:.4f}",
                             ha='center', va='center', fontsize=7,
                             color=text_color, alpha=0.8)
            else:
                ax1.text(j, i, "N/A",
                         ha='center', va='center', fontsize=9,
                         color='gray')

    plt.tight_layout()
    out_path1 = os.path.join(project_dir, f"{args.output_prefix}_accuracy.png")
    plt.savefig(out_path1, dpi=200, bbox_inches='tight')
    plt.savefig(out_path1.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
    print(f"\n图1已保存: {out_path1}")

    # ==========================================
    # 图2：采样概率 p 热力图（理论值）
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    im2 = ax2.imshow(p_matrix, cmap='viridis', aspect='auto',
                     origin='lower', interpolation='nearest',
                     vmin=0, vmax=1)

    ax2.set_xticks(range(n_high))
    ax2.set_xticklabels([f"{v:.0f}" for v in eps_high_values], fontsize=11)
    ax2.set_yticks(range(n_low))
    ax2.set_yticklabels([f"{v:.0f}" for v in eps_low_values], fontsize=11)

    ax2.set_xlabel(r"$\varepsilon_{high}$", fontsize=14)
    ax2.set_ylabel(r"$\varepsilon_{low}$", fontsize=14)
    ax2.set_title(
        r"Sampling Probability $p$ for Different $(\varepsilon_{low}, \varepsilon_{high})$ "
        f"Combinations\n"
        r"Fixed $\varepsilon_{target}$" + f" = {args.eps_target}",
        fontsize=14, fontweight='bold'
    )

    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label("Sampling Probability p", fontsize=12)

    # 标注 p 值
    for i in range(n_low):
        for j in range(n_high):
            if not np.isnan(p_matrix[i, j]):
                val = p_matrix[i, j]
                text_color = 'white' if val < 0.5 else 'black'
                ax2.text(j, i, f"{val:.4f}",
                         ha='center', va='center', fontsize=9,
                         color=text_color)

    plt.tight_layout()
    out_path2 = os.path.join(project_dir, f"{args.output_prefix}_probability.png")
    plt.savefig(out_path2, dpi=200, bbox_inches='tight')
    plt.savefig(out_path2.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
    print(f"图2已保存: {out_path2}")

    # ==========================================
    # 图3：Accuracy vs p 散点图（揭示 p 与效用的关系）
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(9, 6))

    p_vals = []
    acc_vals = []
    labels = []
    for i, el in enumerate(eps_low_values):
        for j, eh in enumerate(eps_high_values):
            if not np.isnan(acc_matrix[i, j]) and not np.isnan(p_matrix[i, j]):
                p_vals.append(p_matrix[i, j])
                acc_vals.append(acc_matrix[i, j])
                labels.append(f"({el:.0f},{eh:.0f})")

    scatter = ax3.scatter(p_vals, acc_vals, c=acc_vals, cmap='RdYlGn',
                          s=100, edgecolors='black', linewidth=0.5, zorder=5)

    # 标注每个点
    for x, y, label in zip(p_vals, acc_vals, labels):
        ax3.annotate(label, (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, alpha=0.8)

    ax3.set_xlabel(r"Sampling Probability $p$", fontsize=14)
    ax3.set_ylabel("Average Accuracy", fontsize=14)
    ax3.set_title(
        r"Accuracy vs Sampling Probability $p$" + "\n"
        r"(Fixed $\varepsilon_{target}$" + f" = {args.eps_target})",
        fontsize=14, fontweight='bold'
    )
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, 1.05)

    plt.colorbar(scatter, ax=ax3, shrink=0.85, label="Accuracy")
    plt.tight_layout()
    out_path3 = os.path.join(project_dir, f"{args.output_prefix}_scatter.png")
    plt.savefig(out_path3, dpi=200, bbox_inches='tight')
    plt.savefig(out_path3.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
    print(f"图3已保存: {out_path3}")

    # ==========================================
    # 打印数据表格
    # ==========================================
    print("\n" + "=" * 70)
    print("数据汇总表")
    print("=" * 70)
    print(f"{'eps_low':>8}  {'eps_high':>9}  {'p':>10}  {'Accuracy':>10}  {'Std':>8}")
    print("-" * 55)
    for i, el in enumerate(eps_low_values):
        for j, eh in enumerate(eps_high_values):
            if not np.isnan(acc_matrix[i, j]):
                print(f"{el:>8.1f}  {eh:>9.1f}  {p_matrix[i, j]:>10.6f}  "
                      f"{acc_matrix[i, j]:>10.4f}  {std_matrix[i, j]:>8.4f}")

    plt.show()


if __name__ == "__main__":
    main()
