#!/bin/bash
# ============================================================
# BERT Utility 实验脚本 - 无脱敏 Baseline
# 数据集: SST-2
# 流程: 直接使用原始数据 → 微调BERT → 评估Accuracy
# 用途: 与 SanText 脱敏版本对比，作为 utility 上界
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=5
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/data/youyaru/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"

# 每个实验跑几轮
NUM_RUNS=5

# 结果汇总文件
SUMMARY_FILE="$PROJECT_DIR/utility_results_SST2_bert_no_sanitization.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "run,seed,accuracy" > "$SUMMARY_FILE"
fi

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " BERT Utility 实验 - 无脱敏 Baseline (SST-2)"
echo " 直接使用原始数据进行 BERT 微调"
echo " 重复轮数: $NUM_RUNS"
echo "============================================"
echo ""

# ============================================================
# 多轮实验（不同随机种子）
# ============================================================
for RUN in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "---- 第 $RUN/$NUM_RUNS 轮 ----"

    SEED=$((42 + RUN - 1))

    # 微调输出目录
    OUTPUT_FINETUNE="$PROJECT_DIR/output_SST2_bert_no_sanitization_utility/run_${RUN}"

    echo "  随机种子: $SEED"
    echo "  数据目录: $DATA_DIR (原始数据，无脱敏)"
    echo "  微调输出: $OUTPUT_FINETUNE"

    mkdir -p "$OUTPUT_FINETUNE"

    # ============================================================
    # 微调 BERT（直接在原始训练集上）
    # ============================================================
    echo ""
    echo "[微调 BERT] 使用原始数据 (seed=$SEED)..."

    python3 run_glue.py \
      --model_name_or_path "$BERT_MODEL" \
      --task_name sst-2 \
      --do_train \
      --do_eval \
      --data_dir "$DATA_DIR" \
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

    # ============================================================
    # 提取准确率
    # ============================================================
    ACCURACY="N/A"

    # 方式1：从 eval_results_sst-2.txt 提取
    if [ -f "$OUTPUT_FINETUNE/eval_results_sst-2.txt" ]; then
        ACCURACY=$(grep -oP "eval_acc\s*=\s*\K[0-9.]+" "$OUTPUT_FINETUNE/eval_results_sst-2.txt" 2>/dev/null || echo "N/A")
    fi

    # 方式2：从 all_results.json 提取（新版transformers）
    if [ "$ACCURACY" = "N/A" ] || [ -z "$ACCURACY" ]; then
        if [ -f "$OUTPUT_FINETUNE/all_results.json" ]; then
            ACCURACY=$(python3 -c "
import json
with open('$OUTPUT_FINETUNE/all_results.json') as f:
    data = json.load(f)
acc = data.get('eval_accuracy', data.get('eval_acc', 'N/A'))
print(acc)
" 2>/dev/null || echo "N/A")
        fi
    fi

    # 方式3：从 eval_results.json 提取
    if [ "$ACCURACY" = "N/A" ] || [ -z "$ACCURACY" ]; then
        if [ -f "$OUTPUT_FINETUNE/eval_results.json" ]; then
            ACCURACY=$(python3 -c "
import json
with open('$OUTPUT_FINETUNE/eval_results.json') as f:
    data = json.load(f)
acc = data.get('eval_accuracy', data.get('eval_acc', 'N/A'))
print(acc)
" 2>/dev/null || echo "N/A")
        fi
    fi

    # 记录结果
    echo "$RUN,$SEED,$ACCURACY" >> "$SUMMARY_FILE"

    echo ""
    echo "本轮结果: run=$RUN, seed=$SEED, accuracy=$ACCURACY"
    echo "----------------------------------------"
done

# ============================================================
# 所有轮次完成后，输出汇总统计
# ============================================================
echo ""
echo "============================================"
echo " 所有 $NUM_RUNS 轮实验完成！最终汇总："
echo "============================================"

python3 - <<EOF
import csv, statistics

results = []
with open("$SUMMARY_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['accuracy'] != 'N/A':
            try:
                results.append(float(row['accuracy']))
            except:
                pass

if results:
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"  轮数={len(results)} | 平均Accuracy={avg:.4f} | 标准差={std:.4f}")
    print(f"  各轮结果: {[round(r,4) for r in results]}")
else:
    print("  暂无有效结果")
EOF

echo ""
echo "完整结果已保存至: $SUMMARY_FILE"
echo ""
echo "============================================"
echo " 实验说明："
echo " - 本实验直接使用原始 SST-2 数据（无任何脱敏）"
echo " - Accuracy 代表 BERT 微调后在 dev 集上的分类准确率"
echo " - 此结果作为 Utility 的上界（最优参考值）"
echo " - 与 utility_results_SST2_bert.csv 对比可评估脱敏代价"
echo "============================================"
