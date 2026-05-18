#!/bin/bash
# ============================================================
# Mask Token Inference Attack 实验脚本 (Mixed 数据版本)
# 数据集: QNLI
# 流程: 读取 mixed 脱敏数据 → Mask Token Attack → 统计防御率
# 数据来源: output_QNLI_bert_utility_mixed_epsXX/
#           （由 mix_sanitize_qnli.py 生成）
#
# 用法:
#   bash run_mask_token_attack_mixed_qnli.sh          # 默认 EPS_HIGH=24
#   bash run_mask_token_attack_mixed_qnli.sh 20       # 指定 EPS_HIGH=20
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/data/youyaru/youyaru/SanText-main"
ORIGINAL_DATA_DIR="$PROJECT_DIR/data/QNLI"       # 原始数据（ground truth）
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"

# ============================================================
# 高质量文档的 MLDP epsilon（可通过第一个命令行参数指定，默认 24）
# ============================================================
EPS_HIGH=${1:-24}

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

# 结果汇总文件
SUMMARY_FILE="$PROJECT_DIR/mask_attack_results_QNLI_mixed_eps${EPS_HIGH}.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "epsilon,run,seed,defense_rate,attack_success_rate" > "$SUMMARY_FILE"
fi

# 攻击结果输出根目录
ATTACK_OUTPUT_ROOT="$MIXED_DATA_ROOT/attack"

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " Mask Token Inference Attack (QNLI, Mixed 数据)"
echo " EPS_HIGH:     $EPS_HIGH"
echo " 原始数据:     $ORIGINAL_DATA_DIR"
echo " 脱敏数据:     $MIXED_DATA_ROOT"
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
    echo " 开始 epsilon = $EPSILON 的攻击实验"
    echo "============================================"

    for RUN in $(seq 1 $NUM_RUNS); do
        echo ""
        echo "---- epsilon=$EPSILON, 第 $RUN/$NUM_RUNS 轮 ----"

        SEED=$((42 + RUN - 1))

        # 脱敏数据目录
        SANITIZED_DIR="$MIXED_DATA_ROOT/eps_${EPSILON_DIR}/run_${RUN}/eps_${EPSILON_DIR}"

        # 攻击结果输出目录
        OUTPUT_DIR="$ATTACK_OUTPUT_ROOT/eps_${EPSILON_DIR}/run_${RUN}"

        echo "  随机种子:   $SEED"
        echo "  脱敏数据:   $SANITIZED_DIR"
        echo "  输出目录:   $OUTPUT_DIR"

        # 检查脱敏数据是否存在
        if [ ! -f "$SANITIZED_DIR/dev.tsv" ]; then
            echo "  [ERROR] 找不到脱敏数据: $SANITIZED_DIR/dev.tsv"
            echo "  跳过此轮"
            continue
        fi

        # 检查是否已有结果（避免重复运行）
        if [ -f "$SUMMARY_FILE" ]; then
            EXISTING=$(python3 -c "
import csv
with open('$SUMMARY_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['epsilon'] == '$EPSILON' and int(row['run']) == $RUN and row['attack_success_rate'] != 'N/A':
            print('found')
            break
" 2>/dev/null)
            if [ "$EXISTING" = "found" ]; then
                echo "  已有结果，跳过"
                continue
            fi
        fi

        mkdir -p "$OUTPUT_DIR"

        # ============================================================
        # 运行 Mask Token Inference Attack
        # ============================================================
        echo ""
        echo "[运行] Mask Token Inference Attack (epsilon=$EPSILON, run=$RUN, seed=$SEED)..."

        TMP_LOG=$(mktemp /tmp/attack_XXXXXX.log)

        python3 bert_inference_token_from_file_qnli.py \
          --model_path "$BERT_MODEL" \
          --original_data_dir "$ORIGINAL_DATA_DIR" \
          --sanitized_data_dir "$SANITIZED_DIR" \
          --output_dir "$OUTPUT_DIR" \
          --max_seq_length 128 \
          --batch_size 256 \
          --seed "$SEED" 2>&1 | tee "$TMP_LOG"

        # 从最后一行提取攻击成功率
        ATTACK_SUCCESS_RATE=$(tail -n 1 "$TMP_LOG" | grep -oP '[0-9.]+([eE][+-]?[0-9]+)?')
        rm -f "$TMP_LOG"

        # 计算防御率
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
        echo "  防御率:     $DEFENSE_RATE"
        echo "----------------------------------------"
    done

    # ============================================================
    # 计算当前 epsilon 的平均值和标准差
    # ============================================================
    echo ""
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
            try:
                defense_rates.append(float(row['defense_rate']))
                attack_rates.append(float(row['attack_success_rate']))
            except:
                pass

if defense_rates:
    avg_defense = statistics.mean(defense_rates)
    std_defense = statistics.stdev(defense_rates) if len(defense_rates) > 1 else 0.0
    avg_attack  = statistics.mean(attack_rates)
    std_attack  = statistics.stdev(attack_rates) if len(attack_rates) > 1 else 0.0
    print(f"  epsilon=$EPSILON | 轮数={len(defense_rates)}")
    print(f"  平均防御率      = {avg_defense:.4f} ± {std_defense:.4f}")
    print(f"  平均攻击成功率  = {avg_attack:.4f}  ± {std_attack:.4f}")
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
        try:
            summary.setdefault(eps, {'defense': [], 'attack': []})
            summary[eps]['defense'].append(float(row['defense_rate']))
            summary[eps]['attack'].append(float(row['attack_success_rate']))
        except:
            pass

print(f"{'epsilon':<12} {'轮数':<8} {'平均防御率':<14} {'防御率标准差':<14} {'平均攻击率':<14}")
print("-" * 70)
for eps in sorted(summary.keys(), key=float):
    d = summary[eps]['defense']
    a = summary[eps]['attack']
    avg_d = statistics.mean(d)
    std_d = statistics.stdev(d) if len(d) > 1 else 0.0
    avg_a = statistics.mean(a)
    print(f"{eps:<12} {len(d):<8} {avg_d:<14.4f} {std_d:<14.4f} {avg_a:<14.4f}")
EOF

echo ""
echo "完整结果已保存至: $SUMMARY_FILE"
echo ""
echo "============================================"
echo " 实验说明："
echo " - 数据集: QNLI（question + sentence 两个字段分别攻击）"
echo " - EPS_HIGH = $EPS_HIGH"
echo " - 防御率 = 1 - 攻击成功率"
echo " - 防御率越高 → 隐私保护越好"
echo " - 攻击成功率 = BERT 正确预测原始 token 的比例"
echo " - 数据由 mix_sanitize_qnli.py 混合采样生成"
echo "   (eps_0 随机文档 + eps_${EPS_HIGH} 高质量文档 按概率 p 混合)"
echo "============================================"
