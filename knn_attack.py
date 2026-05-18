#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Attack (Song & Raghunathan, 2020) 针对 SanText 脱敏系统

攻击原理：
  给定脱敏后的词 w'，计算 w' 的 embedding 与词表中所有词的欧氏距离，
  取距离最小的 Top-K 个词作为候选，若原始词 w 在 Top-K 中则攻击成功。

评估指标：
  Top-K Accuracy = 原始词出现在 Top-K 候选中的比例
  Defense Rate   = 1 - Top-K Accuracy

性能优化：
  - 概率矩阵：GPU 分块流式计算 + 直接采样，避免存储 30522×30522 大矩阵
  - KNN 查找：GPU 矩阵运算加速
"""

import argparse
import os
import csv
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import euclidean_distances
from transformers import BertTokenizer, BertForMaskedLM
from utils import get_vocab_SST2, get_vocab_QNLI, get_vocab_CliniSTS

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# SanText 多进程脱敏（GPU 流式概率矩阵版本）
# ============================================================

# 全局变量（多进程共享）
_g_prob_matrix  = None
_g_word2id      = None
_g_sword2id     = None
_g_id2sword     = None
_g_all_words    = None
_g_p            = None
_g_tokenizer    = None


def SanText_init_global(prob_matrix, word2id, sword2id, all_words, p, tokenizer):
    global _g_prob_matrix, _g_word2id, _g_sword2id, _g_id2sword
    global _g_all_words, _g_p, _g_tokenizer
    _g_prob_matrix = prob_matrix
    _g_word2id     = word2id
    _g_sword2id    = sword2id
    _g_id2sword    = {v: k for k, v in sword2id.items()}
    _g_all_words   = all_words
    _g_p           = p
    _g_tokenizer   = tokenizer


def SanText_worker(doc):
    """对单个文档进行 SanText 脱敏"""
    new_doc = []
    for word in doc:
        if word in _g_word2id:
            if word in _g_sword2id:
                idx = _g_word2id[word]
                prob = _g_prob_matrix[idx]
                sampled = np.random.choice(len(prob), 1, p=prob)[0]
                new_doc.append(_g_id2sword[sampled])
            else:
                if random.random() <= _g_p:
                    idx = _g_word2id[word]
                    prob = _g_prob_matrix[idx]
                    sampled = np.random.choice(len(prob), 1, p=prob)[0]
                    new_doc.append(_g_id2sword[sampled])
                else:
                    new_doc.append(word)
        else:
            # OOV：随机替换
            sampled = np.random.randint(len(_g_all_words))
            new_doc.append(_g_all_words[sampled])
    return new_doc


def build_prob_matrix_gpu(all_word_embed, sensitive_word_embed, epsilon,
                           block_size=1000, device=None):
    """
    用 GPU 分块计算概率矩阵，避免一次性构建 N×S 大矩阵。
    返回 numpy float32 概率矩阵 (N, S)。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = all_word_embed.shape[0]
    S = sensitive_word_embed.shape[0]
    logger.info(f"  概率矩阵: ({N}, {S})，使用 GPU 分块计算 (block={block_size}, device={device})")

    # 将敏感词 embedding 放到 GPU
    sen_t   = torch.tensor(sensitive_word_embed, dtype=torch.float32, device=device)  # (S, D)
    sen_nsq = (sen_t ** 2).sum(dim=1)  # (S,)

    prob_matrix = np.empty((N, S), dtype=np.float32)

    for i in tqdm(range(0, N, block_size), desc="  计算概率矩阵", leave=False):
        i_end = min(i + block_size, N)
        blk   = torch.tensor(all_word_embed[i:i_end], dtype=torch.float32, device=device)  # (B, D)
        blk_nsq = (blk ** 2).sum(dim=1)  # (B,)

        # 欧氏距离平方: (B, S)
        dot    = blk @ sen_t.T                                          # (B, S)
        dist_sq = blk_nsq.unsqueeze(1) + sen_nsq.unsqueeze(0) - 2 * dot
        dist_sq = dist_sq.clamp(min=0.0).sqrt()                        # 欧氏距离 (B, S)

        # 指数机制概率: softmax(epsilon * (-dist) / 2)
        logits = -dist_sq * (epsilon / 2.0)                            # (B, S)
        prob_blk = F.softmax(logits, dim=1)                            # (B, S)

        prob_matrix[i:i_end] = prob_blk.cpu().numpy()

    return prob_matrix


# ============================================================
# KNN Attack 核心（GPU 加速）
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

    # 收集 (原始词id, 脱敏词id) 对
    orig_ids = []
    san_ids  = []
    for orig_doc, san_doc in zip(original_docs, sanitized_docs):
        for orig_tok, san_tok in zip(orig_doc, san_doc):
            if orig_tok in word2id and san_tok in word2id:
                orig_ids.append(word2id[orig_tok])
                san_ids.append(word2id[san_tok])

    total = len(orig_ids)
    if total == 0:
        return 0.0, 1.0

    logger.info(f"  KNN Attack: {total} 个有效 token 对，Top-K={topk}，device={device}")

    orig_ids = np.array(orig_ids, dtype=np.int64)
    san_ids  = np.array(san_ids,  dtype=np.int64)

    # 全词表 embedding → GPU
    embed_t  = torch.tensor(all_word_embed, dtype=torch.float32, device=device)  # (N, D)
    norm_sq  = (embed_t ** 2).sum(dim=1)  # (N,)

    hit = 0
    n_batches = (total + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="  KNN 查找", leave=False):
        s = b * batch_size
        e = min(s + batch_size, total)

        san_emb  = embed_t[san_ids[s:e]]          # (B, D)
        san_nsq  = norm_sq[san_ids[s:e]]           # (B,)

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
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KNN Attack on SanText (BERT Embedding)")

    parser.add_argument("--task",            type=str, default="SST-2",
                        choices=["SST-2", "QNLI", "CliniSTS"])
    parser.add_argument("--data_dir",        type=str, default="./data/SST-2/")
    parser.add_argument("--bert_model_path", type=str, default="./bert-base-uncased")
    parser.add_argument("--output_dir",      type=str, default="./output_SST2_knn_attack")
    parser.add_argument("--result_file",     type=str, default="./knn_attack_results_SST2.csv")
    parser.add_argument("--epsilon",         type=float, default=10.0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--run",             type=int,   default=1)
    parser.add_argument("--topk",            type=int,   default=10)
    parser.add_argument("--p",               type=float, default=0.2)
    parser.add_argument("--sensitive_word_percentage", type=float, default=1.0)
    parser.add_argument("--threads",         type=int,   default=8)
    parser.add_argument("--prob_block_size", type=int,   default=1000,
                        help="概率矩阵 GPU 分块大小")
    parser.add_argument("--knn_batch_size",  type=int,   default=2048,
                        help="KNN GPU 批大小")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"KNN Attack | task={args.task} | ε={args.epsilon} "
                f"| seed={args.seed} | run={args.run} | Top-K={args.topk} | device={device}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. 加载 BERT Tokenizer & Embedding
    # ----------------------------------------------------------
    logger.info("[1/4] 加载 BERT Tokenizer 和 Embedding...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model_path)
    embedding_matrix = bert_model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    del bert_model  # 释放内存
    logger.info(f"  BERT Embedding: {embedding_matrix.shape}")

    # ----------------------------------------------------------
    # 2. 构建词表 & 词向量
    # ----------------------------------------------------------
    logger.info("[2/4] 构建词表和词向量...")

    if args.task == "SST-2":
        vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type="subword")
    elif args.task == "QNLI":
        vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type="subword")
    elif args.task == "CliniSTS":
        vocab = get_vocab_CliniSTS(args.data_dir, tokenizer, tokenizer_type="subword")
    else:
        raise NotImplementedError

    sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
    words             = [key for key, _ in vocab.most_common()]
    sensitive_words   = words[-sensitive_word_count - 1:]
    sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}

    word2id  = {}
    sword2id = {}
    all_word_embed       = []
    sensitive_word_embed = []
    all_count = sensitive_count = 0

    for cur_word in tokenizer.vocab:
        if cur_word in vocab and cur_word not in word2id:
            word2id[cur_word] = all_count
            emb = embedding_matrix[tokenizer.convert_tokens_to_ids(cur_word)]
            all_word_embed.append(emb)
            all_count += 1
            if cur_word in sensitive_words2id:
                sword2id[cur_word] = sensitive_count
                sensitive_word_embed.append(emb)
                sensitive_count += 1

    all_word_embed       = np.array(all_word_embed,       dtype=np.float32)
    sensitive_word_embed = np.array(sensitive_word_embed, dtype=np.float32)
    del embedding_matrix

    logger.info(f"  all_word_embed: {all_word_embed.shape}")
    logger.info(f"  sensitive_word_embed: {sensitive_word_embed.shape}")

    # ----------------------------------------------------------
    # 3. 读取数据 & SanText 脱敏
    # ----------------------------------------------------------
    logger.info("[3/4] 读取数据并执行 SanText 脱敏...")

    # GPU 分块计算概率矩阵
    prob_matrix = build_prob_matrix_gpu(
        all_word_embed, sensitive_word_embed,
        epsilon=args.epsilon,
        block_size=args.prob_block_size,
        device=device
    )
    logger.info(f"  概率矩阵完成: {prob_matrix.shape}")

    # 读取 dev.tsv
    original_docs = []
    data_file = os.path.join(args.data_dir, "dev.tsv")
    num_lines = sum(1 for _ in open(data_file))
    with open(data_file, "r") as rf:
        next(rf)
        for line in tqdm(rf, total=num_lines - 1, desc="  读取 dev.tsv"):
            content = line.strip().split("\t")
            if args.task == "SST-2":
                doc = tokenizer.tokenize(content[0])
            elif args.task == "QNLI":
                doc = tokenizer.tokenize(content[1])
            else:
                doc = tokenizer.tokenize(content[0])
            original_docs.append(doc)

    logger.info(f"  共 {len(original_docs)} 条数据")

    # 多进程 SanText 脱敏
    threads = min(args.threads, cpu_count())
    with Pool(threads,
              initializer=SanText_init_global,
              initargs=(prob_matrix, word2id, sword2id, words, args.p, tokenizer)) as pool:
        sanitized_docs = list(tqdm(
            pool.imap(SanText_worker, original_docs, chunksize=32),
            total=len(original_docs),
            desc="  SanText 脱敏"
        ))
        pool.close()

    # ----------------------------------------------------------
    # 4. KNN Attack（GPU 加速）
    # ----------------------------------------------------------
    logger.info(f"[4/4] 执行 KNN Attack (Top-{args.topk})...")

    topk_acc, defense_rate = knn_attack_gpu(
        original_docs  = original_docs,
        sanitized_docs = sanitized_docs,
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
    logger.info(f"  epsilon              : {args.epsilon}")
    logger.info(f"  run / seed           : {args.run} / {args.seed}")
    logger.info(f"  Top-{args.topk} Accuracy    : {topk_acc:.4f}  ({topk_acc*100:.2f}%)")
    logger.info(f"  Defense Rate         : {defense_rate:.4f}  ({defense_rate*100:.2f}%)")
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
