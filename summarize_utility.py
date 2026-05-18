"""
统计所有 utility 实验结果：每个 epsilon 的平均 accuracy 和标准差
用法: python summarize_utility.py [文件名]
  - 不带参数: 统计所有 utility_results_*.csv 文件
  - 带参数: 只统计指定文件
"""
import pandas as pd
import glob
import sys

def summarize(filepath):
    df = pd.read_csv(filepath)
    stats = df.groupby('epsilon')['accuracy'].agg(['mean', 'std', 'count']).sort_index()
    stats.columns = ['avg_accuracy', 'std', 'num_runs']
    stats['avg_accuracy_pct'] = (stats['avg_accuracy'] * 100).round(2)
    stats['std_pct'] = (stats['std'] * 100).round(2)
    return stats

if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob('utility_results_*bert*.csv'))

    for f in files:
        print(f"\n{'='*60}")
        print(f"文件: {f}")
        print(f"{'='*60}")
        stats = summarize(f)
        for eps, row in stats.iterrows():
            print(f"  ε={eps:<6} | Accuracy={row['avg_accuracy_pct']:>6.2f}% ± {row['std_pct']:.2f}% | 轮数={int(row['num_runs'])}")
