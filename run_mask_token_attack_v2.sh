#!/bin/bash
# ============================================================
# Mask Token Inference Attack 实验脚本 (优化版)
# 数据集: SST-2
# 直接使用 bert_inference_token.py（内部包含脱敏+攻击）
# ============================================================

set -e  # 遇到错误立即停止

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=3
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/home/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"
THREADS=8
TASK="SST-2"
METHOD="WarmUp"
EMBEDDING="bert"  # 使用 BERT embedding

# ============================================================
# epsilon 列表：论文中 Figure 4 使用的值
# ============================================================
EPSILON_LIST=(0)

# 每个 epsilon 跑几轮（取平均值）
NUM_RUNS=5

# 结果汇总文件
SUMMARY_FILE="$PROJECT_DIR/mask_attack_results_SST2_v2.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "epsilon,run,seed,defense_rate,attack_success_rate" > "$SUMMARY_FILE"
fi

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " Mask Token Inference Attack 实验 (SST-2)"
echo " Epsilon 列表: ${EPSILON_LIST[*]}"
echo " 每个 epsilon 重复: $NUM_RUNS 轮"
echo " Embedding: $EMBEDDING"
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

        # 输出目录
        OUTPUT_DIR="$PROJECT_DIR/output_SST2_bert_attack/eps_${EPSILON_DIR}/run_${RUN}"

        echo "  随机种子: $SEED"
        echo "  输出目录: $OUTPUT_DIR"

        # ============================================================
        # 直接运行 bert_inference_token.py（包含脱敏+攻击）
        # ============================================================
        echo ""
        echo "[运行] Mask Token Inference Attack (epsilon=$EPSILON, seed=$SEED)..."

        # 运行攻击脚本
        ATTACK_OUTPUT=$(python3 bert_inference_token.py \
          --task "$TASK" \
          --method "$METHOD" \
          --embedding_type "$EMBEDDING" \
          --epsilon "$EPSILON" \
          --model_path "$BERT_MODEL" \
          --data_dir "$DATA_DIR" \
          --output_dir "$OUTPUT_DIR" \
          --threads "$THREADS" \
          --seed "$SEED" \
          --batch_size 256 \
          --max_seq_length 128 2>&1 | tee /dev/stderr)

        # 从输出中提取攻击成功率
        ATTACK_SUCCESS_RATE=$(echo "$ATTACK_OUTPUT" | tail -n 1 | grep -oP '[0-9.]+')
        
        # 计算防御率 = 1 - 攻击成功率
        if [ -n "$ATTACK_SUCCESS_RATE" ]; then
            DEFENSE_RATE=$(python3 -c "print(f'{1 - float($ATTACK_SUCCESS_RATE):.4f}')")
        else
            DEFENSE_RATE="N/A"
            ATTACK_SUCCESS_RATE="N/A"
        fi

        # 记录结果
        echo "$EPSILON,$RUN,$SEED,$DEFENSE_RATE,$ATTACK_SUCCESS_RATE" >> "$SUMMARY_FILE"

        echo ""
        echo "本轮结果: epsilon=$EPSILON, run=$RUN, seed=$SEED"
        echo "  攻击成功率: $ATTACK_SUCCESS_RATE"
        echo "  防御率: $DEFENSE_RATE"
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

defense_rates = []
attack_rates = []
with open("$SUMMARY_FILE") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['epsilon'] == '$EPSILON' and row['defense_rate'] != 'N/A':
            defense_rates.append(float(row['defense_rate']))
            attack_rates.append(float(row['attack_success_rate']))

if defense_rates:
    avg_defense = statistics.mean(defense_rates)
    std_defense = statistics.stdev(defense_rates) if len(defense_rates) > 1 else 0.0
    avg_attack = statistics.mean(attack_rates)
    std_attack = statistics.stdev(attack_rates) if len(attack_rates) > 1 else 0.0
    print(f"  epsilon=$EPSILON | 轮数={len(defense_rates)}")
    print(f"  平均防御率={avg_defense:.4f} ± {std_defense:.4f}")
    print(f"  平均攻击成功率={avg_attack:.4f} ± {std_attack:.4f}")
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
        if row['defense_rate'] == 'N/A':
            continue
        if eps not in summary:
            summary[eps] = {'defense': [], 'attack': []}
        summary[eps]['defense'].append(float(row['defense_rate']))
        summary[eps]['attack'].append(float(row['attack_success_rate']))

print(f"{'epsilon':<12} {'轮数':<8} {'平均防御率':<14} {'防御率标准差':<14} {'平均攻击率':<14}")
print("-" * 70)
for eps in sorted(summary.keys(), key=float):
    defense_rates = summary[eps]['defense']
    attack_rates = summary[eps]['attack']
    avg_defense = statistics.mean(defense_rates)
    std_defense = statistics.stdev(defense_rates) if len(defense_rates) > 1 else 0.0
    avg_attack = statistics.mean(attack_rates)
    print(f"{eps:<12} {len(defense_rates):<8} {avg_defense:<14.4f} {std_defense:<14.4f} {avg_attack:<14.4f}")
EOF

echo ""
echo "完整结果已保存至: $SUMMARY_FILE"
echo ""
echo "============================================"
echo " 实验说明："
echo " - 防御率 = 1 - 攻击成功率"
echo " - 防御率越高，表示隐私保护越好"
echo " - 攻击成功率 = BERT正确预测原始token的比例"
echo "============================================"