#!/bin/bash
# ============================================================
# SanText 完整运行脚本 - GloVe 版本
# 使用 GloVe 词向量进行文本脱敏
# ============================================================

set -e  # 遇到错误立即停止

# ---- 初始化 conda ----
source /home/youyaru/miniconda3/etc/profile.d/conda.sh
conda activate santext
echo "已激活 conda 环境: $CONDA_DEFAULT_ENV"

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=5
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ---- 配置参数（可按需修改）----
PROJECT_DIR="/home/youyaru/SanText-main"
DATA_DIR="$PROJECT_DIR/data/SST-2"
GLOVE_EMBEDDING="$PROJECT_DIR/data/glove.840B.300d.txt"
BERT_MODEL="$PROJECT_DIR/bert-base-uncased"
EPSILON=14.0
THREADS=4
TASK="SST-2"
METHOD="SanText"
EMBEDDING="glove"          # 使用 GloVe 词向量
EMBEDDING_SIZE=300         # GloVe 词向量维度

OUTPUT_SANITIZE="$PROJECT_DIR/output_glove/SST-2"
OUTPUT_FINETUNE="$PROJECT_DIR/output_glove/finetune/SST-2"

# 格式化 epsilon 为两位小数（与 run_SanText.py 输出目录保持一致）
EPSILON_DIR=$(python3 -c "print('%.2f' % $EPSILON)")

# 设置 HuggingFace 镜像（国内网络）
export HF_ENDPOINT=https://hf-mirror.com

cd "$PROJECT_DIR"

echo "============================================"
echo " SanText 运行脚本 (GloVe 版本)"
echo " 任务: $TASK | 方法: $METHOD | epsilon: $EPSILON"
echo " 词向量: GloVe (300d)"
echo "============================================"
echo ""

# ============================================================
# 阶段一：文本脱敏 (使用 GloVe 词向量)
# ============================================================
echo "[阶段1/2] 开始文本脱敏 (GloVe)..."
echo "  输入数据: $DATA_DIR"
echo "  GloVe词向量: $GLOVE_EMBEDDING"
echo "  输出目录: $OUTPUT_SANITIZE/eps_${EPSILON_DIR}/"
echo ""

python3 run_SanText.py \
  --task "$TASK" \
  --method "$METHOD" \
  --embedding_type "$EMBEDDING" \
  --word_embedding_path "$GLOVE_EMBEDDING" \
  --word_embedding_size "$EMBEDDING_SIZE" \
  --epsilon "$EPSILON" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_SANITIZE" \
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
# 阶段二：脱敏感知微调 BERT
# 注：即使脱敏用GloVe，微调仍使用BERT模型
# ============================================================
echo "[阶段2/2] 开始微调 BERT..."
echo "  训练数据: $OUTPUT_SANITIZE/eps_${EPSILON_DIR}/"
echo "  输出目录: $OUTPUT_FINETUNE"
echo ""

mkdir -p "$OUTPUT_FINETUNE"

python3 run_glue.py \
  --model_name_or_path "$BERT_MODEL" \
  --task_name sst-2 \
  --do_train \
  --do_eval \
  --data_dir "$OUTPUT_SANITIZE/eps_${EPSILON_DIR}/" \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir "$OUTPUT_FINETUNE" \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_steps 2000

echo ""
echo "============================================"
echo " 全部完成！"
echo " 脱敏文本: $OUTPUT_SANITIZE/eps_${EPSILON_DIR}/"
echo " 微调模型: $OUTPUT_FINETUNE"
echo "============================================"