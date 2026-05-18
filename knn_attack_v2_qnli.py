#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Attack v2 —— QNLI 数据集版本（直接读取已有脱敏文件）

与 SST-2 版本的区别：
  QNLI 有两个文本字段（question + sentence），
  本脚本对两个字段的 token 合并后统一做 KNN Attack。

QNLI 数据格式：
  原始:   index\tquestion\tsentence\tlabel
  脱敏后: index\tquestion(subword)\tsentence(subword)\tlabel
"""

import argparse
import os
import csv
import logging
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

logger = logging.getLogger(__name__)


# ============================================================
# KNN Attack 核心（GPU 加速）—— 与 SST-2 版本完全相同
# ============================================================

def knn_attack_gpu(original_docs, sanitized_docs, word2id,
                   all_word_embed, topk=10, batch_size=2048, device=None):
    """
    KNN Attack：对每个脱敏词，在 embedding 空间找 Top-K 最近邻，
    检查原始词是否在其中。

    返回：(topk_accuracy, defense_rate)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_ids = []
    san_ids  = []
    skip_count = 0

    for orig_doc, san_doc in zip(original_docs, sanitized_docs):
        for orig_tok, san_tok in zip(orig_doc, san_doc):
            if orig_tok in word2id and san_tok in word2id:
                orig_ids.append(word2id[orig_tok])
                san_ids.append(word2id[san_tok])
            else:
                skip_count += 1

    total = len(orig_ids)
    if total == 0:
        logger.warning("没有有效的 token 对！")
        return 0.0, 1.0

    logger.info(f"  有效 token 对: {total}，跳过 OOV: {skip_count}，Top-K={topk}，device={device}")

    orig_ids = np.array(orig_ids, dtype=np.int64)
    san_ids  = np.array(san_ids,  dtype=np.int64)

    # 全词表 embedding → GPU
    embed_t = torch.tensor(all_word_embed, dtype=torch.float32, device=device)  # (N, D)
    norm_sq = (embed_t ** 2).sum(dim=1)  # (N,)

    hit = 0
    n_batches = (total + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="  KNN 查找", leave=False):
        s = b * batch_size
        e = min(s + batch_size, total)

        san_emb = embed_t[san_ids[s:e]]           # (B, D)
        san_nsq = norm_sq[san_ids[s:e]]            # (B,)

        # 欧氏距离平方 (B, N)
        dot     = san_emb @ embed_t.T              # (B, N)
        dist_sq = san_nsq.unsqueeze(1) + norm_sq.unsqueeze(0) - 2 * dot
        dist_sq = dist_sq.clamp(min=0.0)

        # Top-K 最小距离索引
        topk_idx = torch.topk(dist_sq, k=topk, dim=1, largest=False).indices  # (B, K)
        topk_np  = topk_idx.cpu().numpy()

        orig_batch = orig_ids[s:e]
        for i in range(e - s):
            if orig_batch[i] in topk_np[i]:
                hit += 1

    acc          = hit / total
    defense_rate = 1.0 - acc
    return acc, defense_rate


# ============================================================
# 读取 QNLI TSV 文件
# ============================================================

def read_qnli_tsv_sanitized(tsv_path):
    """
    读取脱敏后的 QNLI tsv，返回 token 列表的列表
    每条数据的 question 和 sentence 分别作为独立的 doc
    """
    docs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 跳过 header
    if lines and ('question' in lines[0] or 'sentence' in lines[0] or 'label' in lines[0]):
        lines = lines[1:]

    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 4:
            question_tokens = parts[1].split()
            sentence_tokens = parts[2].split()
        elif len(parts) == 3:
            question_tokens = parts[0].split()
            sentence_tokens = parts[1].split()
        else:
            continue
        docs.append(question_tokens)
        docs.append(sentence_tokens)
    return docs


def read_qnli_tsv_tokenize(tsv_path, tokenizer):
    """
    读取原始 QNLI tsv，用 BERT tokenizer 分词
    每条数据的 question 和 sentence 分别作为独立的 doc
    """
    docs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 跳过 header
    if lines and ('question' in lines[0] or 'sentence' in lines[0] or 'label' in lines[0]):
        lines = lines[1:]

    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 4:
            question = parts[1]
            sentence = parts[2]
        elif len(parts) == 3:
            question = parts[0]
            sentence = parts[1]
        else:
            continue
        question_tokens = tokenizer.tokenize(question)
        sentence_tokens = tokenizer.tokenize(sentence)
        docs.append(question_tokens)
        docs.append(sentence_tokens)
    return docs


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="KNN Attack v2 —— QNLI 数据集版本（直接读取已有脱敏文件）"
    )

    parser.add_argument("--bert_model_path",  type=str, default="./bert-base-uncased")
    parser.add_argument("--original_data_dir",type=str, default="./data/QNLI",
                        help="原始 QNLI 数据目录（含 dev.tsv）")
    parser.add_argument("--sanitized_dir",    type=str, required=True,
                        help="脱敏后数据目录（含 dev.tsv）")
    parser.add_argument("--result_file",      type=str, default="./knn_attack_results_QNLI.csv")
    parser.add_argument("--epsilon",          type=float, default=2.0)
    parser.add_argument("--run",              type=int,   default=1)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--topk",             type=int,   default=10)
    parser.add_argument("--knn_batch_size",   type=int,   default=2048)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"KNN Attack v2 (QNLI) | ε={args.epsilon} | run={args.run} | "
                f"seed={args.seed} | Top-K={args.topk} | device={device}")
    logger.info(f"脱敏文件目录: {args.sanitized_dir}")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # 1. 加载 BERT Tokenizer & Embedding
    # ----------------------------------------------------------
    logger.info("[1/3] 加载 BERT Tokenizer 和 Embedding...")
    tokenizer  = BertTokenizer.from_pretrained(args.bert_model_path)
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model_path)
    embedding_matrix = bert_model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    del bert_model
    logger.info(f"  BERT Embedding: {embedding_matrix.shape}")

    # 构建 word2id 和 all_word_embed（直接用 BERT 全词表）
    word2id        = {}
    all_word_embed = []
    for word, idx in tokenizer.vocab.items():
        word2id[word] = len(word2id)
        all_word_embed.append(embedding_matrix[idx])

    all_word_embed = np.array(all_word_embed, dtype=np.float32)
    del embedding_matrix
    logger.info(f"  词表大小: {len(word2id)}，Embedding: {all_word_embed.shape}")

    # ----------------------------------------------------------
    # 2. 读取原始文件 & 脱敏文件
    # ----------------------------------------------------------
    logger.info("[2/3] 读取原始文件和脱敏文件...")

    orig_path = os.path.join(args.original_data_dir, "dev.tsv")
    san_path  = os.path.join(args.sanitized_dir, "dev.tsv")

    if not os.path.exists(san_path):
        logger.error(f"脱敏文件不存在: {san_path}")
        return

    # 原始文件：需要 BERT tokenize
    original_docs  = read_qnli_tsv_tokenize(orig_path, tokenizer)
    # 脱敏文件：token 已经是 subword，直接 split
    sanitized_docs = read_qnli_tsv_sanitized(san_path)

    logger.info(f"  原始文档数（question+sentence）: {len(original_docs)}")
    logger.info(f"  脱敏文档数（question+sentence）: {len(sanitized_docs)}")

    # 对齐长度
    n = min(len(original_docs), len(sanitized_docs))
    original_docs  = original_docs[:n]
    sanitized_docs = sanitized_docs[:n]

    # 对齐每条文档的 token 数
    aligned_orig = []
    aligned_san  = []
    for orig_doc, san_doc in zip(original_docs, sanitized_docs):
        min_len = min(len(orig_doc), len(san_doc))
        if min_len > 0:
            aligned_orig.append(orig_doc[:min_len])
            aligned_san.append(san_doc[:min_len])

    total_tokens = sum(len(d) for d in aligned_orig)
    logger.info(f"  对齐后文档数: {len(aligned_orig)}，总 token 数: {total_tokens}")

    # ----------------------------------------------------------
    # 3. KNN Attack
    # ----------------------------------------------------------
    logger.info(f"[3/3] 执行 KNN Attack (Top-{args.topk})...")

    topk_acc, defense_rate = knn_attack_gpu(
        original_docs  = aligned_orig,
        sanitized_docs = aligned_san,
        word2id        = word2id,
        all_word_embed = all_word_embed,
        topk           = args.topk,
        batch_size     = args.knn_batch_size,
        device         = device,
    )

    # ----------------------------------------------------------
    # 输出结果
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"  epsilon            : {args.epsilon}")
    logger.info(f"  run / seed         : {args.run} / {args.seed}")
    logger.info(f"  Top-{args.topk} Accuracy  : {topk_acc:.4f}  ({topk_acc*100:.2f}%)")
    logger.info(f"  Defense Rate       : {defense_rate:.4f}  ({defense_rate*100:.2f}%)")
    logger.info("=" * 60)

    # 写入汇总 CSV
    write_header = not os.path.exists(args.result_file)
    with open(args.result_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epsilon", "run", "seed", "topk",
                             "attack_success_rate", "defense_rate"])
        writer.writerow([args.epsilon, args.run, args.seed, args.topk,
                         round(topk_acc, 6), round(defense_rate, 6)])

    logger.info(f"结果已追加至: {args.result_file}")
    print(f"epsilon={args.epsilon} run={args.run} seed={args.seed} "
          f"top{args.topk}_acc={topk_acc:.4f} defense_rate={defense_rate:.4f}")


if __name__ == "__main__":
    main()
