"""
Privacy-Utility Tradeoff 对比图
用法:
    python plot_privacy_utility_tradeoff.py                          # 画所有已有的 mixed 曲线
    python plot_privacy_utility_tradeoff.py --eps_high 24            # 只画 eps_high=24
    python plot_privacy_utility_tradeoff.py --eps_high 18 24         # 画 eps_high=18 和 24
    python plot_privacy_utility_tradeoff.py --eps_high 16 18 24      # 画 eps_high=16, 18, 24
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 13

# ============================================================
# 配置区：在这里添加新的 eps_high 数据
# ============================================================
# 格式: eps_high -> (utility_csv, attack_csv, 显示颜色, 标记形状)
# 添加新的 eps_high 时，只需在这里加一行即可

MIXED_DATA_CONFIG = {
    24: {
        'utility_csv': 'utility_results_SST2_bert_mixed_eps24.csv',
        'attack_csv':  'mask_attack_results_SST2_mixed_eps24.csv',
        'color':       '#E91E63',   # 粉红
        'marker':      's',         # 方形
    },
    18: {
        'utility_csv': 'utility_results_SST2_bert_mixed_eps18.csv',
        'attack_csv':  'mask_attack_results_SST2_mixed_eps18.csv',
        'color':       '#FF9800',   # 橙色
        'marker':      'D',         # 菱形
    },
    16: {
        'utility_csv': 'utility_results_SST2_bert_mixed_eps16.csv',
        'attack_csv':  'mask_attack_results_SST2_mixed_eps16.csv',
        'color':       '#4CAF50',   # 绿色
        'marker':      '^',         # 三角形
    },
    20: {
        'utility_csv': 'utility_results_SST2_bert_mixed_eps20.csv',
        'attack_csv':  'mask_attack_results_SST2_mixed_eps20.csv',
        'color':       '#9C27B0',   # 紫色
        'marker':      'v',         # 倒三角
    },
    22: {
        'utility_csv': 'utility_results_SST2_bert_mixed_eps22.csv',
        'attack_csv':  'mask_attack_results_SST2_mixed_eps22.csv',
        'color':       '#795548',   # 棕色
        'marker':      'p',         # 五角星
    },
    # ---- 添加新的 eps_high 只需复制下面的模板 ----
    # 14: {
    #     'utility_csv': 'utility_results_SST2_bert_mixed_eps14.csv',
    #     'attack_csv':  'mask_attack_results_SST2_mixed_eps14.csv',
    #     'color':       '#9C27B0',   # 紫色
    #     'marker':      'v',         # 倒三角
    # },

}

# SanText 基线数据（固定不变）
SANTEXT_CONFIG = {
    'utility_csv': 'utility_results_SST2_bert.csv',
    'attack_csv':  'mask_attack_results_SST2_v2.csv',
    'color':       '#2196F3',   # 蓝色
    'marker':      'o',         # 圆形
}


def compute_stats(df, value_col):
    """按 epsilon 分组，计算均值和标准差"""
    grouped = df.groupby('epsilon')[value_col].agg(['mean', 'std']).reset_index()
    grouped.columns = ['epsilon', 'mean', 'std']
    grouped['std'] = grouped['std'].fillna(0)
    return grouped


def load_and_merge(utility_csv, attack_csv):
    """读取 utility 和 attack 数据，合并为一个 DataFrame"""
    utility_df = pd.read_csv(utility_csv)
    attack_df = pd.read_csv(attack_csv)

    defense_stats = compute_stats(attack_df, 'defense_rate')
    acc_stats = compute_stats(utility_df, 'accuracy')
    merged = defense_stats.merge(acc_stats, on='epsilon', suffixes=('_defense', '_acc'))
    return merged.sort_values('mean_defense')


def main():
    parser = argparse.ArgumentParser(description='Plot Privacy-Utility Tradeoff')
    parser.add_argument('--eps_high', nargs='*', type=int, default=None,
                        help='要画的 eps_high 值列表，如 --eps_high 18 24。不指定则画所有已有数据。')
    parser.add_argument('--output', type=str, default='privacy_utility_tradeoff_santext_vs_mixed',
                        help='输出文件名前缀（不含扩展名）')
    parser.add_argument('--no_annotation', action='store_true',
                        help='不标注 epsilon 值')
    args = parser.parse_args()

    # 确定要画哪些 eps_high
    if args.eps_high is None:
        # 自动检测：画所有两个文件都存在的 eps_high
        eps_high_list = []
        for eps_h, cfg in sorted(MIXED_DATA_CONFIG.items()):
            if os.path.exists(cfg['utility_csv']) and os.path.exists(cfg['attack_csv']):
                eps_high_list.append(eps_h)
            else:
                print(f"[跳过] eps_high={eps_h}: 文件不完整")
                if not os.path.exists(cfg['utility_csv']):
                    print(f"       缺少: {cfg['utility_csv']}")
                if not os.path.exists(cfg['attack_csv']):
                    print(f"       缺少: {cfg['attack_csv']}")
    else:
        eps_high_list = args.eps_high

    if not eps_high_list:
        print("[ERROR] 没有可用的 mixed 数据！")
        return

    print(f"将画以下曲线: SanText + Mixed eps_high={eps_high_list}")
    print()

    # ============================================================
    # 1. 加载 SanText 基线数据
    # ============================================================
    santext_merged = load_and_merge(SANTEXT_CONFIG['utility_csv'], SANTEXT_CONFIG['attack_csv'])
    print("=== SanText 数据 ===")
    print(santext_merged[['epsilon', 'mean_defense', 'mean_acc']].to_string(index=False))
    print()

    # ============================================================
    # 2. 加载所有 Mixed 数据
    # ============================================================
    mixed_data = {}
    for eps_h in eps_high_list:
        if eps_h not in MIXED_DATA_CONFIG:
            print(f"[WARNING] eps_high={eps_h} 未在 MIXED_DATA_CONFIG 中配置，跳过")
            continue
        cfg = MIXED_DATA_CONFIG[eps_h]
        if not os.path.exists(cfg['utility_csv']) or not os.path.exists(cfg['attack_csv']):
            print(f"[WARNING] eps_high={eps_h} 数据文件不完整，跳过")
            continue
        merged = load_and_merge(cfg['utility_csv'], cfg['attack_csv'])
        mixed_data[eps_h] = merged
        print(f"=== Mixed Sample (eps_high={eps_h}) ===")
        print(merged[['epsilon', 'mean_defense', 'mean_acc']].to_string(index=False))
        print()

    # ============================================================
    # 3. 画图
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # --- SanText 基线 ---
    cfg = SANTEXT_CONFIG
    ax.errorbar(
        santext_merged['mean_defense'], santext_merged['mean_acc'],
        xerr=santext_merged['std_defense'], yerr=santext_merged['std_acc'],
        fmt=f"{cfg['marker']}-", color=cfg['color'], markersize=7, linewidth=2,
        capsize=3, capthick=1.2, elinewidth=1,
        label='Baseline(EM)', zorder=5
    )
    if not args.no_annotation:
        for _, row in santext_merged.iterrows():
            eps_val = row['epsilon']
            if eps_val in [2, 10, 12, 14, 16, 18, 20, 24]:
                ax.annotate(f'{eps_val:.0f}',
                            (row['mean_defense'], row['mean_acc']),
                            textcoords="offset points", xytext=(8, 5),
                            fontsize=8, color=cfg['color'], alpha=0.8)

    # --- Mixed Sample 曲线 ---
    for eps_h in sorted(mixed_data.keys()):
        cfg = MIXED_DATA_CONFIG[eps_h]
        merged = mixed_data[eps_h]
        ax.errorbar(
            merged['mean_defense'], merged['mean_acc'],
            xerr=merged['std_defense'], yerr=merged['std_acc'],
            fmt=f"{cfg['marker']}-", color=cfg['color'], markersize=7, linewidth=2,
            capsize=3, capthick=1.2, elinewidth=1,
            label=f'Mixed Sample ($\\varepsilon_{{high}}$={eps_h})', zorder=5
        )
        if not args.no_annotation:
            # 选择部分 epsilon 标注，避免拥挤
            all_eps = sorted(merged['epsilon'].unique())
            # 标注策略：标注首、尾、以及中间转折点附近的值
            label_eps = set()
            label_eps.add(all_eps[0])   # 最小
            label_eps.add(all_eps[-1])  # 最大
            # 找 accuracy 变化最大的点（转折点）
            acc_diff = merged['mean_acc'].diff().abs()
            if len(acc_diff) > 2:
                top_change_idx = acc_diff.nlargest(3).index
                for idx in top_change_idx:
                    if idx in merged.index:
                        label_eps.add(merged.loc[idx, 'epsilon'])

            for _, row in merged.iterrows():
                eps_val = row['epsilon']
                if eps_val in label_eps:
                    eps_str = f"{eps_val:.0f}" if eps_val == int(eps_val) else f"{eps_val}"
                    ax.annotate(eps_str,
                                (row['mean_defense'], row['mean_acc']),
                                textcoords="offset points", xytext=(8, -8),
                                fontsize=8, color=cfg['color'], alpha=0.8)

    # --- 参考线 ---
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.38, 0.505, 'Random Guess (Acc=0.5)', fontsize=8.5, color='gray', alpha=0.7)

    # --- 图表美化 ---
    ax.set_xlabel('Privacy (Defense Rate)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Utility (Accuracy)', fontsize=14, fontweight='bold')

    eps_high_str = ', '.join([str(e) for e in sorted(mixed_data.keys())])
    ax.set_title(f'Privacy-Utility Tradeoff: Baseline vs Mixed Sample\n(SST-2, BERT, Mask Token Attack)', fontsize=15, fontweight='bold')

    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.set_xlim(0.37, 0.98)
    ax.set_ylim(0.44, 0.96)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存
    png_path = f'{args.output}.png'
    pdf_path = f'{args.output}.pdf'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"图已保存为:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


if __name__ == '__main__':
    main()
