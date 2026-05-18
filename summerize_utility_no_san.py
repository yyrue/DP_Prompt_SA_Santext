"""
统计无脱敏 baseline 的平均 accuracy 和标准差
"""
import pandas as pd

filepath = "utility_results_QNLI_bert_pretokenized.csv"
df = pd.read_csv(filepath)

# 去重（文件中可能有重复运行的结果追加）
df = df.drop_duplicates()

avg = df['accuracy'].mean()
std = df['accuracy'].std()
n = len(df)

print(f"文件: {filepath}")
print(f"轮数: {n}")
print(f"平均 Accuracy: {avg*100:.2f}% ± {std*100:.2f}%")
print(f"各轮结果:")
for _, row in df.iterrows():
    print(f"  run={int(row['run'])}, seed={int(row['seed'])}, accuracy={row['accuracy']*100:.2f}%")
