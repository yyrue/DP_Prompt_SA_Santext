#!/bin/bash

# =============================================================================
# Mask Token Inference Attack - 无脱敏版本（Baseline）
# =============================================================================
# 此脚本直接对原始数据进行Mask Token Inference Attack
# 不使用SanText进行脱敏，用于对比实验
# =============================================================================

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: santext"

# GPU设置
export CUDA_VISIBLE_DEVICES=5
echo "使用 GPU: 5"

# 基础路径配置
BASE_DIR="/home/youyaru/SanText-main"
DATA_DIR="$BASE_DIR/data/SST-2"
BERT_MODEL="$BASE_DIR/bert-base-uncased"
OUTPUT_DIR="$BASE_DIR/output_SST2_no_sanitization"
BATCH_SIZE=256

# 实验配置
NUM_RUNS=5
TASK="SST-2"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 结果文件 (保存在输出目录中，由Python脚本自动写入)
# Shell脚本不再重复创建

echo ""
echo "============================================"
echo " Mask Token Inference Attack - 无脱敏Baseline"
echo " 数据集: $TASK"
echo " 重复次数: $NUM_RUNS 轮"
echo "============================================"
echo ""

# 多轮实验
for run in $(seq 1 $NUM_RUNS); do
    seed=$((41 + run))
    
    echo "---- 第 $run/$NUM_RUNS 轮 ----"
    echo "  随机种子: $seed"
    echo "  输出目录: $OUTPUT_DIR/run_$run"
    
    # 创建当前轮次的输出目录
    mkdir -p "$OUTPUT_DIR/run_$run"
    
    echo ""
    echo "[阶段 1/2] 准备原始数据 (无脱敏)..."
    
    # 直接复制原始数据到输出目录（模拟无脱敏）
    cp "$DATA_DIR/dev.tsv" "$OUTPUT_DIR/run_$run/dev.tsv"
    cp "$DATA_DIR/train.tsv" "$OUTPUT_DIR/run_$run/train.tsv"
    
    echo "原始数据已复制到: $OUTPUT_DIR/run_$run"
    
    echo ""
    echo "[阶段 2/2] Mask Token Inference Attack (seed=$seed)..."
    
    # 运行攻击脚本
    # 使用非常大的epsilon (如10000) 表示不进行有意义的脱敏
    python3 "$BASE_DIR/bert_inference_token_no_sanitization.py" \
        --task "$TASK" \
        --model_path "$BERT_MODEL" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/run_$run" \
        --batch_size "$BATCH_SIZE" \
        --seed "$seed" \
        --max_seq_length 64
    
    echo ""
    echo "第 $run 轮完成"
    echo "----------------------------------------"
done

echo ""
echo "============================================"
echo " 所有实验完成！"
echo "============================================"
echo ""
echo "结果已保存到: $OUTPUT_DIR/mask_attack_results_no_sanitization.csv"
echo ""

# 显示结果汇总
echo "========== 结果汇总 =========="
cat "$OUTPUT_DIR/mask_attack_results_no_sanitization.csv"