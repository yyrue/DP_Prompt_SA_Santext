#!/bin/bash
# ============================================================
# KNN Attack v2 运行脚本
# 直接读取已有脱敏文件，跳过脱敏步骤，速度大幅提升
#
# 脱敏文件路径：
#   output_SST2_bert_utility/eps_{EPS}/run_{RUN}/eps_{EPS}/dev.tsv
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
SANITIZED_BASE="$PROJECT_DIR/output_SST2_bert_utility"

# ---- 实验配置 ----
TOPK=10
NUM_RUNS=5
KNN_BATCH_SIZE=2048

# ---- epsilon 列表 ----
EPSILON_LIST=(0)

# ---- 结果文件 ----
RESULT_FILE="$PROJECT_DIR/knn_attack_results_SST2_v2_Top${TOPK}.csv"

export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " KNN Attack v2 (直接读取已有脱敏文件)"
echo " Top-K       : $TOPK"
echo " Epsilon 列表: ${EPSILON_LIST[*]}"
echo " 每个 epsilon: $NUM_RUNS 轮"
echo " 结果文件    : $RESULT_FILE"
echo "============================================"
echo ""

# ============================================================
# 遍历所有 epsilon 和 run
# ============================================================
for EPSILON in "${EPSILON_LIST[@]}"; do
    # 格式化为两位小数，如 2.0 → 2.00
    EPS_DIR=$(python3 -c "print('%.2f' % $EPSILON)")

    echo ""
    echo "============================================"
    echo " epsilon = $EPSILON  (目录后缀: eps_${EPS_DIR})"
    echo "============================================"

    for RUN in $(seq 1 $NUM_RUNS); do
        SEED=$((42 + RUN - 1))

        # 脱敏文件目录
        SANITIZED_DIR="$SANITIZED_BASE/eps_${EPS_DIR}/run_${RUN}/eps_${EPS_DIR}"

        echo ""
        echo "---- epsilon=$EPSILON, run=$RUN/$NUM_RUNS (seed=$SEED) ----"
        echo "  脱敏文件: $SANITIZED_DIR/dev.tsv"

        # 检查文件是否存在
        if [ ! -f "$SANITIZED_DIR/dev.tsv" ]; then
            echo "  ⚠️  文件不存在，跳过: $SANITIZED_DIR/dev.tsv"
            continue
        fi

        python3 "$PROJECT_DIR/knn_attack_v2.py" \
            --bert_model_path   "$BERT_MODEL" \
            --original_data_dir "$DATA_DIR" \
            --sanitized_dir     "$SANITIZED_DIR" \
            --result_file       "$RESULT_FILE" \
            --epsilon           "$EPSILON" \
            --run               "$RUN" \
            --seed              "$SEED" \
            --topk              "$TOPK" \
            --knn_batch_size    "$KNN_BATCH_SIZE"

        echo "  ✅ 完成"
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
# 完整汇总
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

print(f"\n  KNN Attack (Top-$TOPK) 结果汇总 — SST-2, BERT Embedding")
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
