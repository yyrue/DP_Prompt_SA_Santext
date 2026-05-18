#!/bin/bash
# ============================================================
# SanText 实验脚本
# 数据集: SST-2
# 自动遍历 epsilon=1,2,3，每个跑 5 轮取平均值
# ============================================================

set -e  # 遇到错误立即停止

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=5
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/home/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
GLOVE_EMBEDDING="$PROJECT_DIR/data/glove.840B.300d.txt"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"
THREADS=4
TASK="SST-2"
METHOD="SanText"
EMBEDDING="glove"
EMBEDDING_SIZE=300

# ============================================================
# epsilon 列表：自动遍历所有值
# ============================================================
EPSILON_LIST=(12.0 14.0 16.0 18.0)

# 每个 epsilon 跑几轮（5轮取平均）
NUM_RUNS=5

# 结果汇总文件（追加模式，不覆盖之前的结果）
SUMMARY_FILE="$PROJECT_DIR/experiment_results_SST2.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "epsilon,run,seed,accuracy" > "$SUMMARY_FILE"
fi

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " SanText 实验 (SST-2)"
echo " Epsilon 列表: ${EPSILON_LIST[*]}"
echo " 每个 epsilon 重复: $NUM_RUNS 轮"
echo " 数据集: $TASK"
echo "============================================"
echo ""

# ============================================================
# 遍历所有 epsilon 值
# ============================================================
for EPSILON in "${EPSILON_LIST[@]}"; do
    EPSILON_DIR=$(python3 -c "print('%.2f' % $EPSILON)")

    echo ""
    echo "============================================"
    echo " 开始 epsilon = $EPSILON 的实验"
    echo "============================================"

    for RUN in $(seq 1 $NUM_RUNS); do
        echo ""
        echo "---- epsilon=$EPSILON, 第 $RUN/$NUM_RUNS 轮 ----"

        # 每轮使用不同的随机种子
        SEED=$((42 + RUN - 1))

        # 输出目录（包含 epsilon 和轮次信息）
        OUTPUT_SANITIZE="$PROJECT_DIR/output_SST2/eps_${EPSILON_DIR}/run_${RUN}"
        OUTPUT_FINETUNE="$PROJECT_DIR/output_SST2/finetune/eps_${EPSILON_DIR}/run_${RUN}"

        echo "  随机种子: $SEED"
        echo "  脱敏输出: $OUTPUT_SANITIZE"
        echo "  微调输出: $OUTPUT_FINETUNE"

        # ============================================================
        # 阶段一：文本脱敏
        # ============================================================
        echo ""
        echo "[阶段 1/2] 文本脱敏 (epsilon=$EPSILON, seed=$SEED)..."

        python3 run_SanText.py \
          --task "$TASK" \
          --method "$METHOD" \
          --embedding_type "$EMBEDDING" \
          --word_embedding_path "$GLOVE_EMBEDDING" \
          --word_embedding_size "$EMBEDDING_SIZE" \
          --epsilon "$EPSILON" \
          --data_dir "$DATA_DIR" \
          --output_dir "$OUTPUT_SANITIZE" \
          --threads "$THREADS" \
          --seed "$SEED"

        # ============================================================
        # 阶段二：微调 BERT
        # ============================================================
        echo ""
        echo "[阶段 2/2] 微调 BERT (seed=$SEED)..."

        mkdir -p "$OUTPUT_FINETUNE"

        # 实际脱敏数据目录（run_SanText.py 会自动追加 eps_X.XX 子目录）
        ACTUAL_SANITIZE_DIR="$OUTPUT_SANITIZE/eps_${EPSILON_DIR}"

        python3 run_glue.py \
          --model_name_or_path "$BERT_MODEL" \
          --task_name sst-2 \
          --do_train \
          --do_eval \
          --data_dir "$ACTUAL_SANITIZE_DIR" \
          --max_seq_length 128 \
          --per_device_train_batch_size 64 \
          --per_device_eval_batch_size 64 \
          --learning_rate 2e-5 \
          --num_train_epochs 3.0 \
          --output_dir "$OUTPUT_FINETUNE" \
          --overwrite_output_dir \
          --overwrite_cache \
          --save_steps 2000 \
          --save_total_limit 1 \
          --fp16 \
          --seed "$SEED"

        # 提取准确率
        ACCURACY=$(grep -oP "eval_acc\s*=\s*\K[0-9.]+" "$OUTPUT_FINETUNE/eval_results_sst-2.txt" 2>/dev/null || echo "N/A")
        if [ "$ACCURACY" = "N/A" ] || [ -z "$ACCURACY" ]; then
            ACCURACY=$(grep -oP "eval_accuracy[^0-9]*\K[0-9.]+" "$OUTPUT_FINETUNE/all_results.json" 2>/dev/null || echo "N/A")
        fi

        # 记录结果
        echo "$EPSILON,$RUN,$SEED,$ACCURACY" >> "$SUMMARY_FILE"

        echo ""
        echo "本轮结果: epsilon=$EPSILON, run=$RUN, seed=$SEED, accuracy=$ACCURACY"
        echo ""
    done

    # ============================================================
    # 计算当前 epsilon 的平均值和标准差
    # ============================================================
    echo "============================================"
    echo " epsilon = $EPSILON 的 $NUM_RUNS 轮实验完成，统计结果："
    echo "============================================"

    python3 - <<EOF
import csv, statistics

results = []
with open("$SUMMARY_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['epsilon'] == '$EPSILON' and row['accuracy'] != 'N/A':
            results.append(float(row['accuracy']))

if results:
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"  epsilon=$EPSILON | 轮数={len(results)} | 平均准确率={avg:.4f} | 标准差={std:.4f}")
else:
    print("  暂无有效结果")
EOF

done

# ============================================================
# 所有 epsilon 跑完后，输出完整汇总
# ============================================================
echo ""
echo "============================================"
echo " 所有实验完成！最终汇总："
echo "============================================"

python3 - <<EOF
import csv, statistics

summary = {}
with open("$SUMMARY_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eps = row['epsilon']
        if row['accuracy'] == 'N/A':
            continue
        summary.setdefault(eps, []).append(float(row['accuracy']))

print(f"{'epsilon':<12} {'轮数':<8} {'平均准确率':<14} {'标准差'}")
print("-" * 45)
for eps in sorted(summary.keys(), key=float):
    results = summary[eps]
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"{eps:<12} {len(results):<8} {avg:<14.4f} {std:.4f}")
EOF

echo ""
echo "完整结果已保存至: $SUMMARY_FILE"
