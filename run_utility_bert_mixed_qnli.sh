#!/bin/bash
# ============================================================
# SanText Utility 实验脚本 (Mixed 数据版本)
# 数据集: QNLI
# 流程: 直接读取 mixed 脱敏数据 → 微调 BERT → 评估 Accuracy
# 数据来源: output_QNLI_bert_utility_mixed_eps{EPS_HIGH}/
#           （由 mix_sanitize_qnli.py 生成）
#
# 用法:
#   bash run_utility_bert_mixed_qnli.sh          # 默认 EPS_HIGH=24
#   bash run_utility_bert_mixed_qnli.sh 18       # 指定 EPS_HIGH=18
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置（可通过环境变量覆盖，默认 GPU 1）----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/data/youyaru/youyaru/SanText-main"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"

# ============================================================
# 高质量文档的 MLDP epsilon（可通过第一个命令行参数指定，默认 24）
# ============================================================
EPS_HIGH=${1:-24}

# ============================================================
# Mixed 数据根目录（由 mix_sanitize_qnli.py 生成）
# ============================================================
MIXED_DATA_ROOT="$PROJECT_DIR/output_QNLI_bert_utility_mixed_eps${EPS_HIGH}"

# ============================================================
# epsilon 列表：自动扫描 mixed 目录下所有 eps_* 子目录
# ============================================================
if [ ! -d "$MIXED_DATA_ROOT" ]; then
    echo "ERROR: Mixed 数据目录不存在: $MIXED_DATA_ROOT"
    echo "请先运行 mix_sanitize_qnli.py 生成 mixed 数据！"
    exit 1
fi

# 自动发现所有 epsilon 值，按数值排序
EPSILON_LIST=()
for eps_dir in $(find "$MIXED_DATA_ROOT" -maxdepth 1 -type d -name "eps_*" | sort -V); do
    eps_name=$(basename "$eps_dir")
    eps_val=$(echo "$eps_name" | sed 's/eps_//')
    EPSILON_LIST+=("$eps_val")
done

if [ ${#EPSILON_LIST[@]} -eq 0 ]; then
    echo "ERROR: 未找到任何 eps_* 子目录: $MIXED_DATA_ROOT"
    exit 1
fi

# 每个 epsilon 跑几轮
NUM_RUNS=5

# 结果汇总文件（文件名带上 high epsilon 后缀）
SUMMARY_FILE="$PROJECT_DIR/utility_results_QNLI_bert_mixed_eps${EPS_HIGH}.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "epsilon,run,seed,accuracy" > "$SUMMARY_FILE"
fi

# 微调结果保存根目录
FINETUNE_ROOT="$MIXED_DATA_ROOT/finetune"

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " SanText Utility 实验 (QNLI, Mixed 数据)"
echo " EPS_HIGH:     $EPS_HIGH"
echo " 数据目录:     $MIXED_DATA_ROOT"
echo " Epsilon 列表: ${EPSILON_LIST[*]}"
echo " 每个 epsilon 重复: $NUM_RUNS 轮"
echo " 结果文件:     $SUMMARY_FILE"
echo "============================================"
echo ""

# ============================================================
# 遍历所有 epsilon 值
# ============================================================
for EPSILON in "${EPSILON_LIST[@]}"; do
    EPSILON_DIR=$(python3 -c "print('%.2f' % float('$EPSILON'))")

    echo ""
    echo "============================================"
    echo " 开始 epsilon = $EPSILON 的实验"
    echo "============================================"

    for RUN in $(seq 1 $NUM_RUNS); do
        echo ""
        echo "---- epsilon=$EPSILON, 第 $RUN/$NUM_RUNS 轮 ----"

        SEED=$((42 + RUN - 1))

        # Mixed 脱敏数据目录（直接读取，无需再脱敏）
        SANITIZE_DIR="$MIXED_DATA_ROOT/eps_${EPSILON_DIR}/run_${RUN}/eps_${EPSILON_DIR}"

        # 微调输出目录
        OUTPUT_FINETUNE="$FINETUNE_ROOT/eps_${EPSILON_DIR}/run_${RUN}"

        echo "  随机种子:   $SEED"
        echo "  脱敏数据:   $SANITIZE_DIR"
        echo "  微调输出:   $OUTPUT_FINETUNE"

        # 检查脱敏数据是否存在
        if [ ! -f "$SANITIZE_DIR/train.tsv" ]; then
            echo "  [ERROR] 找不到脱敏数据: $SANITIZE_DIR/train.tsv"
            echo "  请先运行 mix_sanitize_qnli.py 生成 mixed 数据！"
            exit 1
        fi

        # 检查是否已有结果（避免重复运行）
        if [ -f "$SUMMARY_FILE" ]; then
            EXISTING=$(python3 -c "
import csv
with open('$SUMMARY_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['epsilon'] == '$EPSILON' and int(row['run']) == $RUN and row['accuracy'] != 'N/A':
            print('found')
            break
" 2>/dev/null)
            if [ "$EXISTING" = "found" ]; then
                echo "  已有结果，跳过"
                continue
            fi
        fi

        # ============================================================
        # 微调 BERT（直接在 mixed 脱敏数据上）
        # ============================================================
        echo ""
        echo "[唯一阶段] 微调 BERT (epsilon=$EPSILON, run=$RUN, seed=$SEED)..."

        mkdir -p "$OUTPUT_FINETUNE"

        python3 run_glue.py \
          --model_name_or_path "$BERT_MODEL" \
          --task_name qnli \
          --do_train \
          --do_eval \
          --data_dir "$SANITIZE_DIR" \
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
echo " - EPS_HIGH = $EPS_HIGH"
echo " - Accuracy = 在原始 dev 集上的 QNLI 分类准确率"
echo " - Accuracy 越高 → 脱敏后数据效用越好"
echo " - epsilon 越大 → 隐私保护越弱 → Accuracy 应越高"
echo " - 数据由 mix_sanitize_qnli.py 混合采样生成"
echo "   (eps_0 随机文档 + eps_${EPS_HIGH} 高质量文档 按概率 p 混合)"
echo "============================================"
