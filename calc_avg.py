import csv
from collections import defaultdict

CSV_FILE = "/data/youyaru/youyaru/SanText-main/experiment_results_SST2.csv"

# 读取数据
data = defaultdict(list)
with open(CSV_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        eps = row["epsilon"]
        acc = float(row["accuracy"])
        data[eps].append(acc)

# 打印结果
print("=" * 60)
print(f"{'epsilon':<12} {'各轮 accuracy':<35} {'平均值'}")
print("=" * 60)
for eps in sorted(data.keys(), key=float):
    vals = data[eps]
    avg = sum(vals) / len(vals)
    vals_str = "  ".join(f"{v:.4f}" for v in vals)
    print(f"{eps:<12} {vals_str:<35} {avg:.4f}")
print("=" * 60)
