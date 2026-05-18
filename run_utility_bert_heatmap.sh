#!/bin/bash
# ============================================================
# 热力图实验：效用评估脚本
# 固定 eps_target，对所有 (eps_low, eps_high) 组合的混合数据微调 BERT
# ============================================================

set -e

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数 ----
PROJECT_DIR="/data/youyaru/youyaru/SanText-main"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"

# 目标 epsilon（可通过第一个命令行参数指定，默认 10）
EPS_TARGET=${1:-10}

# 可选：第二个参数指定只跑哪些 eps_low（空格分隔的字符串），用于多卡并行
# 用法: bash run_utility_bert_heatmap.sh 10 "0 2"   # 只跑 eps_low=0 和 eps_low=2
#       bash run_utility_bert_heatmap.sh 10          # 跑所有组合
EPS_LOW_FILTER="${2:-}"

# 混合数据根目录（由 mix_sanitize_heatmap.py 生成）
HEATMAP_DATA_ROOT="$PROJECT_DIR/output_SST2_bert_utility_heatmap_target${EPS_TARGET}"

# 每个组合跑几轮
NUM_RUNS=5

# 结果汇总文件
SUMMARY_FILE="$PROJECT_DIR/utility_results_SST2_bert_heatmap_target${EPS_TARGET}.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "eps_low,eps_high,run,seed,accuracy" > "$SUMMARY_FILE"
fi

# 微调结果保存根目录
FINETUNE_ROOT="$HEATMAP_DATA_ROOT/finetune"

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " 热力图实验：效用评估 (SST-2)"
echo " EPS_TARGET:  $EPS_TARGET"
echo " 数据目录:    $HEATMAP_DATA_ROOT"
echo " 结果文件:    $SUMMARY_FILE"
echo "============================================"
echo ""

# ============================================================
# 检查数据目录是否存在
# ============================================================
if [ ! -d "$HEATMAP_DATA_ROOT" ]; then
    echo "[ERROR] 数据目录不存在: $HEATMAP_DATA_ROOT"
    echo "请先运行: python3 mix_sanitize_heatmap.py --eps_target $EPS_TARGET"
    exit 1
fi

# ============================================================
# 遍历所有 (eps_low, eps_high) 组合目录
# ============================================================
for COMBO_DIR in $(ls -d "$HEATMAP_DATA_ROOT"/low*_high* 2>/dev/null | sort); do
    COMBO_NAME=$(basename "$COMBO_DIR")

    # 从目录名解析 eps_low 和 eps_high
    EPS_LOW=$(echo "$COMBO_NAME" | sed 's/low\([0-9]*\)_high.*/\1/')
    EPS_HIGH=$(echo "$COMBO_NAME" | sed 's/low[0-9]*_high\([0-9]*\)/\1/')

    # 如果指定了 EPS_LOW_FILTER，只跑指定的 eps_low
    if [ -n "$EPS_LOW_FILTER" ]; then
        MATCH=0
        for ALLOWED in $EPS_LOW_FILTER; do
            if [ "$EPS_LOW" = "$ALLOWED" ]; then
                MATCH=1
                break
            fi
        done
        if [ "$MATCH" -eq 0 ]; then
            continue
        fi
    fi

    echo ""
    echo "============================================"
    echo " 组合: eps_low=$EPS_LOW, eps_high=$EPS_HIGH"
    echo "============================================"

    for RUN in $(seq 1 $NUM_RUNS); do
        echo ""
        echo "---- eps_low=$EPS_LOW, eps_high=$EPS_HIGH, 第 $RUN/$NUM_RUNS 轮 ----"

        SEED=$((42 + RUN - 1))

        # 混合脱敏数据目录
        SANITIZE_DIR="$COMBO_DIR/run_${RUN}"

        # 微调输出目录
        OUTPUT_FINETUNE="$FINETUNE_ROOT/${COMBO_NAME}/run_${RUN}"

        echo "  随机种子:   $SEED"
        echo "  脱敏数据:   $SANITIZE_DIR"
        echo "  微调输出:   $OUTPUT_FINETUNE"

        # 检查脱敏数据是否存在
        if [ ! -f "$SANITIZE_DIR/train.tsv" ]; then
            echo "  [WARN] 找不到脱敏数据: $SANITIZE_DIR/train.tsv，跳过"
            continue
        fi

        # 检查是否已经跑过（避免重复）
        if grep -q "^${EPS_LOW},${EPS_HIGH},${RUN}," "$SUMMARY_FILE" 2>/dev/null; then
            echo "  [SKIP] 已有结果，跳过"
            continue
        fi

        # ============================================================
        # 微调 BERT
        # ============================================================
        echo ""
        echo "[微调] BERT (eps_low=$EPS_LOW, eps_high=$EPS_HIGH, run=$RUN, seed=$SEED)..."

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

        # 方式2：从 all_results.json 提取
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
        echo "$EPS_LOW,$EPS_HIGH,$RUN,$SEED,$ACCURACY" >> "$SUMMARY_FILE"

        echo ""
        echo "  结果: eps_low=$EPS_LOW, eps_high=$EPS_HIGH, run=$RUN, accuracy=$ACCURACY"
        echo "  ----------------------------------------"
    done
done

# ============================================================
# 最终汇总
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
        key = (row['eps_low'], row['eps_high'])
        if row['accuracy'] == 'N/A':
            continue
        try:
            summary.setdefault(key, []).append(float(row['accuracy']))
        except:
            pass

print(f"{'eps_low':<10} {'eps_high':<10} {'轮数':<6} {'平均Accuracy':<16} {'标准差'}")
print("-" * 60)
for key in sorted(summary.keys(), key=lambda x: (float(x[0]), float(x[1]))):
    results = summary[key]
    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"{key[0]:<10} {key[1]:<10} {len(results):<6} {avg:<16.4f} {std:.4f}")
EOF

echo ""
echo "完整结果已保存至: $SUMMARY_FILE"
echo ""
echo "============================================"
echo " 下一步：运行绘图脚本生成热力图"
echo " python3 plot_heatmap_utility.py --eps_target $EPS_TARGET"
echo "============================================"
