import csv
import statistics

input_file = "mask_attack_results_SST2_v2.csv"

# 读取数据，按 epsilon 分组
summary = {}
with open(input_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        eps = float(row['epsilon'])
        defense_rate = float(row['defense_rate'])
        summary.setdefault(eps, []).append(defense_rate)

# 输出结果
print(f"{'epsilon':<12} {'轮数':<8} {'平均 defense_rate':<20} {'标准差'}")
print("-" * 55)
for eps in sorted(summary.keys()):
    values = summary[eps]
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{eps:<12.1f} {len(values):<8} {avg:<20.4f} {std:.4f}")
