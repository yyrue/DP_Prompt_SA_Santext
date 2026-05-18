#!/bin/bash
# ============================================================
# SanText Utility 实验脚本 (BERT Embedding 版本)
# 数据集: QNLI
# 流程: 脱敏(BERT embedding) → 微调BERT → 评估Accuracy
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=1
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/home/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/QNLI"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"
THREADS=8
TASK="QNLI"
METHOD="SanText"      # SanText = 所有词都是敏感词（run_SanText.py中的标准SanText）
EMBEDDING="bert"      # 使用 BERT embedding

# ============================================================
# epsilon 列表
# ============================================================
#EPSILON_LIST=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0)
EPSILON_LIST=(12.5 13.0 13.5 14.0 15.0 15.5)
# 每个 epsilon 跑几轮
NUM_RUNS=5

# 结果汇总文件
SUMMARY_FILE="$PROJECT_DIR/utility_results_QNLI_bert.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "epsilon,run,seed,accuracy" > "$SUMMARY_FILE"
fi

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " SanText Utility 实验 (QNLI, BERT Embedding)"
echo " Epsilon 列表: ${EPSILON_LIST[*]}"
echo " 每个 epsilon 重复: $NUM_RUNS 轮"
echo " Method: $METHOD"
echo " Embedding: $EMBEDDING"
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

        SEED=$((42 + RUN - 1))

        # 脱敏输出目录
        OUTPUT_SANITIZE="$PROJECT_DIR/output_QNLI_bert_utility/eps_${EPSILON_DIR}/run_${RUN}"
        # 微调输出目录
        OUTPUT_FINETUNE="$PROJECT_DIR/output_QNLI_bert_utility/finetune/eps_${EPSILON_DIR}/run_${RUN}"

        echo "  随机种子: $SEED"
        echo "  脱敏输出: $OUTPUT_SANITIZE"
        echo "  微调输出: $OUTPUT_FINETUNE"

        # ============================================================
        # 阶段一：文本脱敏（使用 BERT embedding）
        # ============================================================
        echo ""
        echo "[阶段 1/2] 文本脱敏 (epsilon=$EPSILON, seed=$SEED, embedding=bert)..."

        python3 run_SanText.py \
          --task "$TASK" \
          --method "$METHOD" \
          --embedding_type "$EMBEDDING" \
          --bert_model_path "$BERT_MODEL" \
          --epsilon "$EPSILON" \
          --data_dir "$DATA_DIR" \
          --output_dir "$OUTPUT_SANITIZE" \
          --threads "$THREADS" \
          --seed "$SEED"

        # ============================================================
        # 阶段二：微调 BERT（在脱敏后的训练集上）
        # ============================================================
        echo ""
        echo "[阶段 2/2] 微调 BERT (seed=$SEED)..."

        mkdir -p "$OUTPUT_FINETUNE"

        # run_SanText.py 会自动追加 eps_X.XX 子目录
        ACTUAL_SANITIZE_DIR="$OUTPUT_SANITIZE/eps_${EPSILON_DIR}"

        python3 run_glue.py \
          --model_name_or_path "$BERT_MODEL" \
          --task_name qnli \
          --do_train \
          --do_eval \
          --data_dir "$ACTUAL_SANITIZE_DIR" \
          --max_seq_length 128 \
          --per_device_train_batch_size 32 \
          --per_device_eval_batch_size 32 \
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

        # 方式1：从 eval_results_qnli.txt 提取
        if [ -f "$OUTPUT_FINETUNE/eval_results_qnli.txt" ]; then
            ACCURACY=$(grep -oP "eval_acc\s*=\s*\K[0-9.]+" "$OUTPUT_FINETUNE/eval_results_qnli.txt" 2>/dev/null || echo "N/A")
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
        echo "$EPSILON,$RUN,$SEED,$ACCURACY" >> "$SUMMARY_FILE"

        echo ""
        echo "本轮结果: epsilon=$EPSILON, run=$RUN, seed=$SEED, accuracy=$ACCURACY"
        echo "----------------------------------------"
    done

    # ============================================================
    # 计算当前 epsilon 的平均值和标准差
    # ============================================================
    echo ""
    echo "============================================"
    echo " epsilon = $EPSILON 的 $NUM_RUNS 轮实验完成"
    echo "============================================"

    python3 - <<EOF
import csv, statistics

results = []
with open("$SUMMARY_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['epsilon'] == '$EPSILON' and row['accuracy'] != 'N/A':
            try:
                results.append(float(row['accuracy']))
            except:
                pass

if results:
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"  epsilon=$EPSILON | 轮数={len(results)} | 平均Accuracy={avg:.4f} | 标准差={std:.4f}")
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
        try:
            summary.setdefault(eps, []).append(float(row['accuracy']))
        except:
            pass

print(f"{'epsilon':<12} {'轮数':<8} {'平均Accuracy':<16} {'标准差'}")
print("-" * 50)
for eps in sorted(summary.keys(), key=float):
    results = summary[eps]
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"{eps:<12} {len(results):<8} {avg:<16.4f} {std:.4f}")
EOF

echo ""
echo "完整结果已保存至: $SUMMARY_FILE"
echo ""
echo "============================================"
echo " 实验说明："
echo " - Accuracy = 在原始dev集上的QNLI分类准确率"
echo " - Accuracy越高 → 脱敏后数据效用越好"
echo " - epsilon越大 → 脱敏程度越低 → Accuracy应越高"
echo "============================================"
