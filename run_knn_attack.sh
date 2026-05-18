#!/bin/bash
# ============================================================
# KNN Attack 实验脚本 (BERT Embedding 版本)
# 数据集: SST-2
# 攻击: KNN Attack (Song & Raghunathan, 2020), Top-10
# 流程: SanText 脱敏(BERT embedding) → KNN Attack → 统计 Defense Rate
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=5
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 路径配置 ----
PROJECT_DIR="/data/youyaru/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"

# ---- 实验配置 ----
TASK="SST-2"
TOPK=10
NUM_RUNS=5
THREADS=8
KNN_BATCH_SIZE=1024

# ---- epsilon 列表 ----
EPSILON_LIST=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0)

# ---- 结果文件 ----
RESULT_FILE="$PROJECT_DIR/knn_attack_results_SST2.csv"
OUTPUT_DIR="$PROJECT_DIR/output_SST2_knn_attack"

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " KNN Attack 实验 (SST-2, BERT Embedding)"
echo " Top-K       : $TOPK"
echo " Epsilon 列表: ${EPSILON_LIST[*]}"
echo " 每个 epsilon 重复: $NUM_RUNS 轮"
echo " 结果文件    : $RESULT_FILE"
echo "============================================"
echo ""

# ============================================================
# 遍历所有 epsilon
# ============================================================
for EPSILON in "${EPSILON_LIST[@]}"; do

    echo ""
    echo "============================================"
    echo " epsilon = $EPSILON"
    echo "============================================"

    for RUN in $(seq 1 $NUM_RUNS); do
        SEED=$((42 + RUN - 1))

        echo ""
        echo "---- epsilon=$EPSILON, 第 $RUN/$NUM_RUNS 轮 (seed=$SEED) ----"

        python3 "$PROJECT_DIR/knn_attack.py" \
            --task          "$TASK" \
            --data_dir      "$DATA_DIR" \
            --bert_model_path "$BERT_MODEL" \
            --output_dir    "$OUTPUT_DIR" \
            --result_file   "$RESULT_FILE" \
            --epsilon       "$EPSILON" \
            --seed          "$SEED" \
            --run           "$RUN" \
            --topk          "$TOPK" \
            --threads       "$THREADS" \
            --knn_batch_size "$KNN_BATCH_SIZE" \
            --sensitive_word_percentage 1.0

        echo "---- 第 $RUN 轮完成 ----"
    done

    # 当前 epsilon 的统计
    echo ""
    python3 - <<EOF
import csv, statistics
results = []
with open("$RESULT_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if float(row['epsilon']) == $EPSILON:
            try:
                results.append(float(row['defense_rate']))
            except:
                pass
if results:
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"  epsilon=$EPSILON | 轮数={len(results)} | 平均 Defense Rate={avg:.4f} ± {std:.4f}")
else:
    print("  暂无有效结果")
EOF

done

# ============================================================
# 所有实验完成，输出完整汇总
# ============================================================
echo ""
echo "============================================"
echo " 所有实验完成！最终汇总："
echo "============================================"

python3 - <<EOF
import csv, statistics

summary = {}
with open("$RESULT_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eps = float(row['epsilon'])
        try:
            summary.setdefault(eps, {'def': [], 'asr': []})
            summary[eps]['def'].append(float(row['defense_rate']))
            summary[eps]['asr'].append(float(row['attack_success_rate']))
        except:
            pass

topk = None
with open("$RESULT_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        topk = row.get('topk', '10')
        break

print(f"\n  KNN Attack (Top-{topk}) 结果汇总 — SST-2, BERT Embedding")
print(f"  {'epsilon':<10} {'轮数':<6} {'Defense Rate':>16} {'Attack Success Rate':>20}")
print("  " + "-" * 56)
for eps in sorted(summary.keys()):
    d = summary[eps]['def']
    a = summary[eps]['asr']
    d_avg = statistics.mean(d)
    d_std = statistics.stdev(d) if len(d) > 1 else 0.0
    a_avg = statistics.mean(a)
    print(f"  {eps:<10.1f} {len(d):<6} {d_avg:>10.4f} ± {d_std:.4f}   {a_avg:>14.4f}")
EOF

echo ""
echo "完整结果已保存至: $RESULT_FILE"
echo "============================================"
