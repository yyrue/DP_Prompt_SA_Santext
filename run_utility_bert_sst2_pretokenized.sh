#!/bin/bash
# ============================================================
# BERT Utility 实验脚本 - 预分词 Baseline
# 数据集: SST-2
# 流程: 原始数据 → BertTokenizer 预分词 → 微调 BERT → 评估 Accuracy
#
# 目的:
#   SanText 脱敏数据在进入 run_glue.py 之前已经过 BertTokenizer.tokenize()，
#   而 run_glue.py 中的 processor_glue.py 对输入做 .split(" ") 后直接
#   调用 convert_tokens_to_ids()，不会再次 tokenize。
#   因此原始数据如果不预分词，大量词汇会被映射为 [UNK]，导致不公平对比。
#   本脚本先对原始数据做同样的预分词，再跑微调，作为公平的 baseline。
#
# 用法:
#   bash run_utility_bert_sst2_pretokenized.sh
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置（可通过环境变量覆盖，默认 GPU 4）----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/data/youyaru/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"

# 预分词数据输出目录（不会修改原始数据）
PRETOKENIZED_DIR="$PROJECT_DIR/output_SST2_bert_pretokenized"

# 每个实验跑几轮
NUM_RUNS=5

# 结果汇总文件
SUMMARY_FILE="$PROJECT_DIR/utility_results_SST2_bert_pretokenized.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "run,seed,accuracy" > "$SUMMARY_FILE"
fi

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

# ============================================================
# 阶段一：预分词（只需执行一次，已有则跳过）
# ============================================================
echo "============================================"
echo " 阶段一：BERT 预分词 (SST-2)"
echo "============================================"

if [ -f "$PRETOKENIZED_DIR/train.tsv" ] && [ -f "$PRETOKENIZED_DIR/dev.tsv" ]; then
    echo "预分词数据已存在，跳过预分词步骤: $PRETOKENIZED_DIR"
else
    echo "对原始数据进行 BERT WordPiece 预分词..."
    echo "  原始数据: $DATA_DIR"
    echo "  输出目录: $PRETOKENIZED_DIR"
    echo ""

    python3 pretokenize_sst2.py \
      --data_dir "$DATA_DIR" \
      --bert_model_path "$BERT_MODEL" \
      --output_dir "$PRETOKENIZED_DIR"

    echo ""
    echo "预分词完成！"
fi

echo ""
echo "============================================"
echo " 阶段二：微调 BERT（使用预分词数据）"
echo " 数据目录: $PRETOKENIZED_DIR"
echo " 重复轮数: $NUM_RUNS"
echo "============================================"
echo ""

# ============================================================
# 阶段二：多轮微调实验（不同随机种子）
# ============================================================
for RUN in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "---- 第 $RUN/$NUM_RUNS 轮 ----"

    SEED=$((42 + RUN - 1))

    # 微调输出目录
    OUTPUT_FINETUNE="$PROJECT_DIR/output_SST2_bert_pretokenized_utility/run_${RUN}"

    echo "  随机种子: $SEED"
    echo "  数据目录: $PRETOKENIZED_DIR (预分词数据)"
    echo "  微调输出: $OUTPUT_FINETUNE"

    # 检查是否已有结果（避免重复运行）
    if [ -f "$SUMMARY_FILE" ]; then
        EXISTING=$(python3 -c "
import csv
with open('$SUMMARY_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['run']) == $RUN and row['accuracy'] != 'N/A':
            print('found')
            break
" 2>/dev/null)
        if [ "$EXISTING" = "found" ]; then
            echo "  已有结果，跳过"
            continue
        fi
    fi

    mkdir -p "$OUTPUT_FINETUNE"

    # ============================================================
    # 微调 BERT（在预分词数据上）
    # ============================================================
    echo ""
    echo "[微调 BERT] 使用预分词数据 (seed=$SEED)..."

    python3 run_glue.py \
      --model_name_or_path "$BERT_MODEL" \
      --task_name sst-2 \
      --do_train \
      --do_eval \
      --data_dir "$PRETOKENIZED_DIR" \
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
echo " - 本实验使用经过 BertTokenizer.tokenize() 预分词的 SST-2 数据"
echo " - 预分词数据目录: $PRETOKENIZED_DIR"
echo " - 原始数据未被修改: $DATA_DIR"
echo " - Accuracy 代表 BERT 微调后在 dev 集上的 SST-2 分类准确率"
echo " - 此结果作为公平对比的 Baseline（与脱敏数据处于同等分词条件）"
echo " - 对比文件："
echo "     无脱敏(未分词): utility_results_SST2_bert_no_sanitization.csv"
echo "     无脱敏(预分词): utility_results_SST2_bert_pretokenized.csv"
echo "     脱敏版本:       utility_results_SST2_bert.csv"
echo "============================================"
