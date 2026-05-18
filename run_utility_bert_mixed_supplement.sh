#!/bin/bash
# ============================================================
# SanText Utility 补充实验脚本 (Mixed 数据版本)
# 用途: 对已有 mixed 数据目录中补充的 epsilon' 进行微调
# 结果追加到已有的汇总 CSV 文件中
#
# 用法:
#   CUDA_VISIBLE_DEVICES=4 bash run_utility_bert_mixed_supplement.sh 24 10.5 11 11.5 12.5 13 13.5
#   参数1: EPS_HIGH（默认 24）
#   参数2+: 需要补充微调的 epsilon' 列表
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置（可通过环境变量覆盖，默认 GPU 5）----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/home/youyaru/SanText-main"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"

# ============================================================
# 第一个参数: EPS_HIGH（默认 24）
# 后续参数: 需要补充的 epsilon' 列表
# ============================================================
EPS_HIGH=${1:-16}
shift  # 移除第一个参数，剩下的都是 epsilon 列表

# 补充的 epsilon 列表
if [ $# -eq 0 ]; then
    # 如果没有传 epsilon 列表，使用默认值
    EPSILON_LIST=(16)
else
    EPSILON_LIST=("$@")
fi

# Mixed 数据根目录
MIXED_DATA_ROOT="$PROJECT_DIR/output_SST2_bert_utility_mixed_eps${EPS_HIGH}"

# 每个 epsilon 跑几轮
NUM_RUNS=5

# 结果汇总文件（追加到已有文件中）
SUMMARY_FILE="$PROJECT_DIR/utility_results_SST2_bert_mixed_eps${EPS_HIGH}.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "epsilon,run,seed,accuracy" > "$SUMMARY_FILE"
fi

# 微调结果保存根目录（与原脚本一致）
FINETUNE_ROOT="$MIXED_DATA_ROOT/finetune"

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " SanText Utility 补充实验 (SST-2, Mixed 数据)"
echo " EPS_HIGH:       $EPS_HIGH"
echo " 数据目录:       $MIXED_DATA_ROOT"
echo " 补充 Epsilon:   ${EPSILON_LIST[*]}"
echo " 每个 epsilon:   $NUM_RUNS 轮"
echo " 结果文件:       $SUMMARY_FILE (追加)"
echo "============================================"
echo ""

# ============================================================
# 遍历补充的 epsilon 值
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

        # Mixed 脱敏数据目录
        SANITIZE_DIR="$MIXED_DATA_ROOT/eps_${EPSILON_DIR}/run_${RUN}/eps_${EPSILON_DIR}"

        # 微调输出目录
        OUTPUT_FINETUNE="$FINETUNE_ROOT/eps_${EPSILON_DIR}/run_${RUN}"

        echo "  随机种子:   $SEED"
        echo "  脱敏数据:   $SANITIZE_DIR"
        echo "  微调输出:   $OUTPUT_FINETUNE"

        # 检查脱敏数据是否存在
        if [ ! -f "$SANITIZE_DIR/train.tsv" ]; then
            echo "  [ERROR] 找不到脱敏数据: $SANITIZE_DIR/train.tsv"
            echo "  请先运行: python mix_sanitize.py --eps_high $EPS_HIGH --target_epsilons ${EPSILON_LIST[*]} --output_dir $MIXED_DATA_ROOT"
            exit 1
        fi

        # ============================================================
        # 微调 BERT
        # ============================================================
        echo ""
        echo "[唯一阶段] 微调 BERT (epsilon=$EPSILON, run=$RUN, seed=$SEED)..."

        mkdir -p "$OUTPUT_FINETUNE"

        python3 run_glue.py \
          --model_name_or_path "$BERT_MODEL" \
          --task_name sst-2 \
          --do_train \
          --do_eval \
          --data_dir "$SANITIZE_DIR" \
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

        # 方式2：从 all_results.json 提取（新版 transformers）
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

        # 记录结果（追加到已有 CSV）
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
# 补充实验完成，输出完整汇总（包含之前的结果）
# ============================================================
echo ""
echo "============================================"
echo " 补充实验完成！完整汇总（含之前结果）："
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
echo " 补充实验说明："
echo " - EPS_HIGH = $EPS_HIGH"
echo " - 补充的 epsilon': ${EPSILON_LIST[*]}"
echo " - 结果已追加到已有 CSV 文件中"
echo " - 微调模型保存在: $FINETUNE_ROOT"
echo "============================================"
