#!/bin/bash
# ============================================================
# SanText 完整运行脚本 (CPU 版本)
# 包含三个阶段：文本脱敏 → BERT微调 → 查看结果
# ============================================================

set -e  # 遇到错误立即停止

# ---- 配置参数（可按需修改）----
PROJECT_DIR="/home/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
BERT_MODEL="bert-base-uncased"
EPSILON=14.0
THREADS=4
TASK="SST-2"
METHOD="SanText"
EMBEDDING="bert"

OUTPUT_SANITIZE="$PROJECT_DIR/output_bert/SST-2"
OUTPUT_FINETUNE="$PROJECT_DIR/output_bert/finetune/SST-2"

# 格式化 epsilon 为两位小数（与 run_SanText.py 输出目录保持一致）
EPSILON_DIR=$(python3 -c "print('%.2f' % $EPSILON)")

# 设置 HuggingFace 镜像（国内网络）
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " SanText 运行脚本 (CPU 版本)"
echo " 任务: $TASK | 方法: $METHOD | epsilon: $EPSILON"
echo "============================================"
echo ""

# ============================================================
# 阶段一：文本脱敏
# ============================================================
echo "[阶段1/2] 开始文本脱敏..."
echo "  输入数据: $DATA_DIR"
echo "  输出目录: $OUTPUT_SANITIZE/eps_${EPSILON_DIR}/"
echo ""

# 强制使用 CPU
export CUDA_VISIBLE_DEVICES=""

python run_SanText.py \
  --task "$TASK" \
  --method "$METHOD" \
  --embedding_type "$EMBEDDING" \
  --epsilon "$EPSILON" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_SANITIZE" \
  --bert_model_path "$BERT_MODEL" \
  --threads "$THREADS"

echo ""
echo "[阶段1 完成] 脱敏文本已保存到: $OUTPUT_SANITIZE/eps_${EPSILON_DIR}/"
echo ""

# 预览脱敏效果
echo "---- 原始文本（前3条）----"
head -4 "$DATA_DIR/train.tsv" | tail -3
echo ""
echo "---- 脱敏后文本（前3条）----"
head -4 "$OUTPUT_SANITIZE/eps_${EPSILON_DIR}/train.tsv" | tail -3
echo ""

# ============================================================
# 阶段二：脱敏感知微调 BERT (CPU 版本)
# ============================================================
echo "[阶段2/2] 开始微调 BERT (CPU 模式)..."
echo "  训练数据: $OUTPUT_SANITIZE/eps_${EPSILON_DIR}/"
echo "  输出目录: $OUTPUT_FINETUNE"
echo "  注意: CPU 模式下训练会较慢"
echo ""

mkdir -p "$OUTPUT_FINETUNE"

python run_glue.py \
  --model_name_or_path "$BERT_MODEL" \
  --task_name sst-2 \
  --do_train \
  --do_eval \
  --data_dir "$OUTPUT_SANITIZE/eps_${EPSILON_DIR}/" \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir "$OUTPUT_FINETUNE" \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_steps 2000 \
  --no_cuda

echo ""
echo "============================================"
echo " 全部完成！"
echo " 脱敏文本: $OUTPUT_SANITIZE/eps_${EPSILON_DIR}/"
echo " 微调模型: $OUTPUT_FINETUNE"
echo "============================================"