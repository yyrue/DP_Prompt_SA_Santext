#!/bin/bash
# ============================================================
# KNN Attack for Mixed (Sample Amplification) 脱敏文件
#
# 对 output_SST2_bert_utility_mixed_eps{16,18,20,24} 目录下
# 所有混合后的 epsilon' 数据执行 KNN Attack
#
# 脱敏文件路径格式：
#   output_SST2_bert_utility_mixed_eps{EPS_HIGH}/eps_{EPS'}/run_{RUN}/eps_{EPS'}/dev.tsv
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=1

echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 路径配置 ----
PROJECT_DIR="/data/youyaru/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"
MIXED_BASE="/home/youyaru/SanText-main"

# ---- 实验配置 ----
TOPK=10
NUM_RUNS=5
KNN_BATCH_SIZE=2048

# ---- 要攻击的 eps_high 列表 ----

EPS_HIGH_LIST=(16 18 20 22 24)

export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " KNN Attack for Mixed (Sample Amplification)"
echo " Top-K       : $TOPK"
echo " eps_high 列表: ${EPS_HIGH_LIST[*]}"
echo " 每个 epsilon: $NUM_RUNS 轮"
echo "============================================"
echo ""

# ============================================================
# 遍历所有 eps_high
# ============================================================
for EPS_HIGH in "${EPS_HIGH_LIST[@]}"; do

    MIXED_DIR="$MIXED_BASE/output_SST2_bert_utility_mixed_eps${EPS_HIGH}"
    RESULT_FILE="$PROJECT_DIR/knn_attack_results_SST2_mixed_eps${EPS_HIGH}_Top${TOPK}.csv"

    echo ""
    echo "============================================================"
    echo " eps_high = $EPS_HIGH"
    echo " 混合数据目录: $MIXED_DIR"
    echo " 结果文件    : $RESULT_FILE"
    echo "============================================================"

    # 检查目录是否存在
    if [ ! -d "$MIXED_DIR" ]; then
        echo "  ⚠️  目录不存在，跳过: $MIXED_DIR"
        continue
    fi

    # 遍历该 mixed 目录下所有 eps_* 子目录
    EPSILON_DIRS=$(find "$MIXED_DIR" -maxdepth 1 -type d -name "eps_*" | sort)

    if [ -z "$EPSILON_DIRS" ]; then
        echo "  ⚠️  未找到任何 eps_* 子目录，跳过: $MIXED_DIR"
        continue
    fi

    # 提取 epsilon 值列表
    echo "  发现的 epsilon' 值："
    for eps_dir in $EPSILON_DIRS; do
        eps_name=$(basename "$eps_dir")
        eps_val=$(echo "$eps_name" | sed 's/eps_//')
        echo "    $eps_val"
    done
    echo ""

    # ---- 遍历所有 epsilon' 和 run ----
    for eps_dir in $EPSILON_DIRS; do
        eps_name=$(basename "$eps_dir")
        EPS_VAL=$(echo "$eps_name" | sed 's/eps_//')

        echo ""
        echo "--------------------------------------------"
        echo " eps_high=$EPS_HIGH, epsilon'=$EPS_VAL"
        echo "--------------------------------------------"

        for RUN in $(seq 1 $NUM_RUNS); do
            SEED=$((42 + RUN - 1))

            # 脱敏文件目录
            SANITIZED_DIR="$MIXED_DIR/${eps_name}/run_${RUN}/${eps_name}"

            echo ""
            echo "  ---- run=$RUN/$NUM_RUNS (seed=$SEED) ----"
            echo "    脱敏文件: $SANITIZED_DIR/dev.tsv"

            # 检查文件是否存在
            if [ ! -f "$SANITIZED_DIR/dev.tsv" ]; then
                echo "    ⚠️  文件不存在，跳过"
                continue
            fi

            # 检查是否已有结果（避免重复运行）
            if [ -f "$RESULT_FILE" ]; then
                EXISTING=$(python3 -c "
import csv
with open('$RESULT_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if abs(float(row['epsilon']) - $EPS_VAL) < 0.001 and int(row['run']) == $RUN and int(row['topk']) == $TOPK:
            print('found')
            break
" 2>/dev/null)
                if [ "$EXISTING" = "found" ]; then
                    echo "    ⏭️  已有结果，跳过"
                    continue
                fi
            fi

            python3 "$PROJECT_DIR/knn_attack_v2.py" \
                --bert_model_path   "$BERT_MODEL" \
                --original_data_dir "$DATA_DIR" \
                --sanitized_dir     "$SANITIZED_DIR" \
                --result_file       "$RESULT_FILE" \
                --epsilon           "$EPS_VAL" \
                --run               "$RUN" \
                --seed              "$SEED" \
                --topk              "$TOPK" \
                --knn_batch_size    "$KNN_BATCH_SIZE"

            echo "    ✅ 完成"
        done

        # 当前 epsilon' 的统计
        echo ""
        python3 - <<EOF
import csv, statistics
results = []
if __import__('os').path.exists("$RESULT_FILE"):
    with open("$RESULT_FILE") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if abs(float(row['epsilon']) - $EPS_VAL) < 0.001:
                try:
                    results.append(float(row['defense_rate']))
                except:
                    pass
if results:
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"  eps_high=$EPS_HIGH, epsilon'=$EPS_VAL | 轮数={len(results)} | Defense Rate={avg:.4f} ± {std:.4f}")
else:
    print("  暂无有效结果")
EOF

    done

    # ---- 当前 eps_high 的完整汇总 ----
    echo ""
    echo "============================================"
    echo " eps_high=$EPS_HIGH 汇总"
    echo "============================================"

    python3 - <<EOF
import csv, statistics, os

if not os.path.exists("$RESULT_FILE"):
    print("  结果文件不存在")
else:
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

    print(f"\n  KNN Attack (Top-$TOPK) — Mixed eps_high=$EPS_HIGH")
    print(f"  {'epsilon':>10} {'轮数':>6} {'Defense Rate':>20} {'Attack Success Rate':>22}")
    print("  " + "-" * 62)
    for eps in sorted(summary.keys()):
        d = summary[eps]['def']
        a = summary[eps]['asr']
        d_avg = statistics.mean(d)
        d_std = statistics.stdev(d) if len(d) > 1 else 0.0
        a_avg = statistics.mean(a)
        a_std = statistics.stdev(a) if len(a) > 1 else 0.0
        print(f"  {eps:>10.2f} {len(d):>6} {d_avg:>12.4f} ± {d_std:.4f}   {a_avg:>12.4f} ± {a_std:.4f}")
EOF

done

# ============================================================
# 全部完成 — 总汇总
# ============================================================
echo ""
echo "============================================================"
echo " 所有实验完成！"
echo "============================================================"
echo ""
echo "结果文件列表："
for EPS_HIGH in "${EPS_HIGH_LIST[@]}"; do
    RESULT_FILE="$PROJECT_DIR/knn_attack_results_SST2_mixed_eps${EPS_HIGH}_Top${TOPK}.csv"
    if [ -f "$RESULT_FILE" ]; then
        LINES=$(wc -l < "$RESULT_FILE")
        echo "  $RESULT_FILE  ($((LINES - 1)) 条记录)"
    fi
done
echo ""
echo "============================================================"
